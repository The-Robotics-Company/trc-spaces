"""Data generation config + task sampler for the PiPER-X "cubes-in-cup" task.

Mirrors ``FrankaPickAndPlaceDataGenConfig`` (single-arm, fixed-base): it reuses
the arm-agnostic ``PickAndPlacePlannerPolicy`` phase machine and only swaps the
IK backend to cuRobo (PiPER's local IK solvers diverge on the low-shelf 45° grasp
poses). The scene ``asset_library/cubes_in_cup_scene.xml`` already contains the
cup receptacle and one cube on the upper shelf (board2, top z=0.13). This module:

  * registers the user asset + grasp libraries at import (idempotent),
  * defines ``PiperXCubesInCupTaskSampler`` — keeps the fixed-base arm at the
    world origin and re-samples the cube + cup on the shelf each episode, using
    the in-scene cup as the place receptacle (no external procthor receptacles),
  * defines ``PiperXCuroboIKPickAndPlacePlannerPolicy`` — solves IK with cuRobo
    per waypoint and executes in joint space,
  * registers ``PiperXCubesInCupDataGenConfig`` for
    ``python -m molmo_spaces.data_generation.main PiperXCubesInCupDataGenConfig``.

Reference: examples/custom_assets/datagen.py (red_block custom scene) and
molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py.
"""

import logging
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig
from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceTaskSamplerConfig
from molmo_spaces.data_generation.config.piper_x_datagen_configs import PiperXCameraSystem
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import (
    USER_ASSET_LIBRARIES,
    USER_GRASP_LIBRARIES,
    register_user_asset_library,
    register_user_grasp_library,
)
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
    JointMoveSegment,
    JointMoveSequence,
    JointTrajectorySegment,
    NoopAction,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.robots.piper_x_config import PiperXRobotConfig
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)

# --- paths ------------------------------------------------------------------
# repo root: molmo_spaces/data_generation/config/<this file> -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSET_LIB = _REPO_ROOT / "asset_library"
_SCENE_XML = _ASSET_LIB / "cubes_in_cup_scene.xml"

# cuRobo IK assets. The URDF/mesh/sphere paths must be passed explicitly to
# CuroboPlannerConfig; otherwise CuroboPlanner defaults them to RBY1's assets.
_CUROBO_YAML = _REPO_ROOT / "assets" / "piper_x" / "curobo_config" / "piper_x.yml"
_CUROBO_URDF = _REPO_ROOT / "external" / "piper-x-arm" / "assets" / "piper_x" / "urdf" / "piper_x.urdf"
_CUROBO_MESHES = _REPO_ROOT / "external" / "piper-x-arm" / "assets" / "piper_x" / "meshes"
_CUROBO_SPHERES = _REPO_ROOT / "assets" / "piper_x" / "curobo_config" / "piper_x_spheres.yml"

# --- user asset + grasp library registration (idempotent; config/ modules are
# all auto-imported by data_generation.main) --------------------------------
_ASSET_LIB_NAME = "piper_cubes_assets"
_GRASP_ROOT = "piper_cubes_grasps"
# grasps_index.json has grasp_paths["piper_x"] -> resolved library name below.
GRASP_LIBRARY = f"{_GRASP_ROOT}/piper_x"

if _ASSET_LIB_NAME not in USER_ASSET_LIBRARIES:
    register_user_asset_library(_ASSET_LIB_NAME, _ASSET_LIB)
if GRASP_LIBRARY not in USER_GRASP_LIBRARIES:
    register_user_grasp_library(_GRASP_ROOT, _ASSET_LIB, _ASSET_LIB_NAME)


class PiperXCubesInCupTaskSampler(PickAndPlaceTaskSampler):
    """Custom-scene pick-and-place sampler for the cubes-in-cup task.

    Differs from the default ``PickAndPlaceTaskSampler`` (built for procthor
    scenes with a mobile robot and externally-added receptacles):

      * the cup is already in the scene, so ``add_auxiliary_objects`` skips
        ``_add_receptacles_to_scene`` and registers ``"cup"`` as the receptacle;
      * ``_prepare_place_target`` / ``_filter_place_target`` are no-ops;
      * ``_sample_and_place_robot`` keeps the fixed-base arm at the world origin
        and samples the cube + cup on the shelf.
    """

    RECEPTACLE_NAME = "cup"
    # all cube bodies in the scene XML; 1-4 are activated per episode
    CUBE_NAMES = ("cube", "cube2_cube", "cube3_cube", "cube4_cube")

    # Shelf sampling regions in the robot base frame, constrained to the PiPER
    # reach sweet spot: board2 top spans x in [0.27, 0.62], y in [-0.35, 0.35] at
    # z=0.13; cuRobo IK at the 45° grasp orientation solves reliably for
    # x in ~[0.32, 0.40], |y| <= ~0.13, z up to ~0.30 m.
    # Sampling region for BOTH the cup and the cubes: the ENTIRE upper board
    # (board2: x 0.27..0.62, y +/-0.35) minus a 2 cm edge margin, capped at the
    # MEASURED reach radius. 2 cm-grid IK sweeps showed: grasp+pregrasp solve
    # everywhere up to r~0.67 m (638/648 cells, far corners fail), and the
    # level-wrist place poses solve on ALL 648/648 cells (r up to 0.70).
    BOARD_X = (0.29, 0.60)
    BOARD_Y = (-0.33, 0.33)
    MAX_REACH_XY = 0.65
    CUBE_Z = 0.144  # cube center; seated ~1 mm into shelf so a contact registers
    CUP_Z = 0.129  # cup body origin; base seated ~1 mm into shelf
    # Layout rules: the cup is sampled first; cubes may not lie between the two
    # front-direction (+x) lines 5 cm to either side of the cup's wall — the
    # cup's front-back lane stays clear of cubes. Cubes are >= 10 cm apart.
    CUP_RADIUS = 0.045  # cup outer radius (AABB/2)
    CUP_LANE_CLEAR = 0.05  # lateral clearance from the cup wall to the lines
    MIN_CUBE_CUBE = 0.10  # min XY center-center distance cube<->cube
    _MAX_PLACE_TRIES = 200
    # parking spots for inactive cubes: on the floor behind the shelf legs,
    # far outside the workspace and the wrist camera's view of the shelf
    _PARK_XY = ((0.9, -0.6), (0.9, -0.45), (0.9, 0.45), (0.9, 0.6))
    _PARK_Z = -0.725  # floor top is -0.74; cube half-size 0.015

    def __init__(self, config) -> None:
        super().__init__(config)
        self._register_in_scene_cup()

    def _register_in_scene_cup(self) -> None:
        """The in-scene cup is the (only, fixed) place receptacle."""
        self._receptacle_names = [self.RECEPTACLE_NAME]
        self._receptacle_uids = [self.RECEPTACLE_NAME]
        self._current_receptacle_index = 0
        self.place_receptacle_name = self.RECEPTACLE_NAME

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add only the policy's aux objects (grasp-collision bodies); the cup is
        already in the scene, so skip ``_add_receptacles_to_scene``."""
        PickTaskSampler.add_auxiliary_objects(self, spec)
        self._register_in_scene_cup()

    # cup already placed by _sample_and_place_robot -> nothing to prepare/filter
    def _prepare_place_target(self, env, place_target_name, pickup_obj_name,
                              pickup_obj_pos, supporting_geom_id) -> bool:
        return True

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        return True

    def _sample_cup_xy(self) -> np.ndarray:
        return self._sample_cube_xy()  # measured place-reach covers the board

    def _sample_cube_xy(self) -> np.ndarray:
        while True:
            xy = np.array([np.random.uniform(*self.BOARD_X),
                           np.random.uniform(*self.BOARD_Y)])
            if np.linalg.norm(xy) <= self.MAX_REACH_XY:
                return xy

    @staticmethod
    def _pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
        return pose

    def _try_place_cubes(self, env, cup_xy: np.ndarray, n_cubes: int) -> list[str]:
        """Place up to n_cubes on the shelf (rest parked on the floor).

        Constraints: outside the cup's front-direction lane (|y - cup_y| >=
        CUP_RADIUS + CUP_LANE_CLEAR) and >= MIN_CUBE_CUBE pairwise.
        Returns the names of the cubes actually placed.
        """
        lane_half = self.CUP_RADIUS + self.CUP_LANE_CLEAR
        placed_xy: list[np.ndarray] = []
        active: list[str] = []
        for i, name in enumerate(self.CUBE_NAMES):
            body = create_mlspaces_body(env.current_data, name)
            if len(active) < n_cubes:
                for _ in range(self._MAX_PLACE_TRIES):
                    xy = self._sample_cube_xy()
                    if abs(xy[1] - cup_xy[1]) < lane_half:
                        continue
                    if any(np.linalg.norm(xy - p) < self.MIN_CUBE_CUBE for p in placed_xy):
                        continue
                    body.pose = self._pose(xy[0], xy[1], self.CUBE_Z,
                                           yaw=np.random.uniform(-np.pi, np.pi))
                    placed_xy.append(xy)
                    active.append(name)
                    break
            if name not in active:
                px, py = self._PARK_XY[i]
                body.pose = self._pose(px, py, self._PARK_Z, yaw=0.0)
        return active

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Keep the fixed-base arm at the origin; sample 1-4 cubes + the cup on
        the shelf (surplus cubes parked on the floor, out of the workspace)."""
        task_cfg = self.config.task_config
        robot_view = env.current_robot.robot_view

        # Fixed-base tabletop arm (base_size=None => arm base at z=0). Not moved;
        # just record the base pose.
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        n_cubes = int(np.random.randint(1, len(self.CUBE_NAMES) + 1))
        cup = create_mlspaces_body(env.current_data, self.place_receptacle_name)

        # cup first; the lane rule always leaves feasible cube space on the board
        cup_xy = self._sample_cup_xy()
        cup.pose = self._pose(cup_xy[0], cup_xy[1], self.CUP_Z,
                              yaw=np.random.uniform(-np.pi, np.pi))
        active = self._try_place_cubes(env, cup_xy, n_cubes)
        if len(active) < n_cubes:
            log.info(f"placed {len(active)}/{n_cubes} cubes (region too crowded)")

        mujoco.mj_fwdPosition(env.current_model, env.current_data)

        # The policy queues shelf cubes dynamically and reads poses live at plan
        # time; pickup_obj_name / these poses are the first target + bookkeeping.
        task_cfg.pickup_obj_name = active[0]
        cube = create_mlspaces_body(env.current_data, active[0])
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(cube.pose).tolist()
        goal_pose = pose_mat_to_7d(cube.pose)
        goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = goal_pose.tolist()
        task_cfg.place_receptacle_start_pose = pose_mat_to_7d(cup.pose).tolist()


class _PlaceCorrectingSequence(JointMoveSequence):
    """Lift/carry/place sequence that re-aims the final descent as it starts.

    The precomputed place qpos assumes the cube sits a fixed distance ahead of
    the TCP; the real in-hand pose differs (the cube seats during the close and
    sags during the carry). When the "place" segment becomes active — the arm
    is at preplace, wrist already level, cube settled — ``correct_fn`` (called
    with this sequence) measures the cube in the TCP frame and re-plans the
    descent segment(s) so the CUBE center descends onto the cup axis.
    """

    def __init__(self, *args, correct_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._correct_fn = correct_fn
        self._corrected = False

    def execute(self) -> bool:
        done = super().execute()
        if (
            not self._corrected
            and self._correct_fn is not None
            and self.move_seg_idx is not None
            and self.move_seg_idx < len(self.move_segments)
            and self.move_segments[self.move_seg_idx].name == "place"
        ):
            self._corrected = True
            self._correct_fn(self)
        return done

    def reset(self):
        super().reset()
        self._corrected = False


class PiperXCuroboIKPickAndPlacePlannerPolicy(PickAndPlacePlannerPolicy):
    """Pick-and-place policy that solves IK with cuRobo and moves in joint space.

    Keeps the arm-agnostic ``PickAndPlacePlannerPolicy`` phase machine but swaps
    the IK backend to a cuRobo ``CuroboPlanner`` built from
    ``assets/piper_x/curobo_config/piper_x.yml``. PiPER's serial / parallel-warp
    local solvers diverge on the 45° low-shelf poses, and per-step TCP->IK jumps
    IK branches, so ``_compute_trajectory`` solves IK ONCE per waypoint
    (seed-chained for continuity) and interpolates in joint space instead of
    re-solving every control step. IK is collision-disabled (pure kinematics);
    the trajectory clears the cup by construction.

    The CuroboPlanner is cached per-process (build + warmup cost seconds).
    """

    _planner = None  # per-process singleton (num_workers=1)

    @classmethod
    def _get_planner(cls):
        if cls._planner is None:
            from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig

            cls._planner = CuroboPlanner(
                CuroboPlannerConfig(
                    curobo_robot_config_path=str(_CUROBO_YAML),
                    urdf_path=str(_CUROBO_URDF),
                    asset_root_path=str(_CUROBO_MESHES),
                    collision_spheres_path=str(_CUROBO_SPHERES),
                    num_ik_seeds=64,
                    # default 0.2 m makes the cup repel trajectories across most
                    # of this shelf workspace -> huge detour arcs; 3 cm keeps the
                    # penalty local to actual near-collisions
                    collision_activation_distance=0.03,
                    # retime transfers to 40% of the joint velocity/accel limits
                    # (0.5 still read as too fast); result.motion_time reflects
                    # this, so replay slows with it
                    time_dilation_factor=0.4,
                )
            )
        return cls._planner

    def reset(self, reset_retries: bool = True):
        if self.task.episode_step_count == 0:
            self._skip_cubes: set[str] = set()  # fresh episode, no blacklist
            self._homing = False  # final go_home not started
            om = self.task.env.object_managers[self.task.env.current_batch_index]
            self._episode_cubes = [
                n for n in PiperXCubesInCupTaskSampler.CUBE_NAMES
                if 0.135 < om.get_object_by_name(n).position[2] < 0.20
            ]  # cubes that start on the shelf (surplus ones are parked below)
        super().reset(reset_retries)

    def _next_shelf_cube(self) -> str | None:
        """Nearest not-yet-placed cube still standing on the shelf, or None.

        A cube counts as done/unpickable when it is within 7 cm of the cup axis
        (in or against the cup), off shelf height (parked or knocked down), or
        blacklisted after repeated planning failures.
        """
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        cup = om.get_object_by_name(self.config.task_config.place_receptacle_name)
        cup_xy = np.asarray(cup.position[:2])
        best, best_d = None, np.inf
        for name in PiperXCubesInCupTaskSampler.CUBE_NAMES:
            if name in self._skip_cubes:
                continue
            try:
                obj = om.get_object_by_name(name)
            except Exception:  # noqa: BLE001
                continue
            p = np.asarray(obj.position)
            if not (0.135 < p[2] < 0.20):
                continue
            if np.linalg.norm(p[:2] - cup_xy) < 0.07:
                continue
            d = float(np.linalg.norm(p[:2]))
            if d < best_d:
                best, best_d = name, d
        return best

    def _all_cubes_in_cup(self) -> bool:
        """True when every cube that started on the shelf ended up in the cup
        (same in-cup classification as _next_shelf_cube: on the cup axis at
        shelf height or above)."""
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        cup = om.get_object_by_name(self.config.task_config.place_receptacle_name)
        cup_xy = np.asarray(cup.position[:2])
        for name in self._episode_cubes:
            p = np.asarray(om.get_object_by_name(name).position)
            if np.linalg.norm(p[:2] - cup_xy) > 0.07 or p[2] < 0.13:
                return False
        return True

    def _advance_to_next_cube(self) -> bool:
        """Retarget to the next shelf cube and replan; False when none are left."""
        while True:
            nxt = self._next_shelf_cube()
            if nxt is None:
                return False
            self.config.task_config.pickup_obj_name = nxt
            log.info(f"Next cube target: {nxt}")
            try:
                self.reset(reset_retries=True)
                return True
            except ValueError as e:
                log.warning(f"Planning for {nxt} failed ({e}); skipping it")
                self._skip_cubes.add(nxt)

    def get_action(self, info):
        action = super().get_action(info)
        if action.get("done") and not self._homing:
            # current cube's script finished (or its retries ran out) — chain
            # straight from the retreat pose to the next shelf cube; park the
            # arm at home only when none remain
            if action.get("success") is False:
                self._skip_cubes.add(self.config.task_config.pickup_obj_name)
            if self._advance_to_next_cube():
                action = super().get_action(info)
            elif self._all_cubes_in_cup():
                # full success: park the arm back at home
                self._homing = True
                self.action_primitives = self._go_home_primitives()
                self.action_idx = 0
                action = super().get_action(info)
            # else: some cube was skipped/lost — end at the retreat pose
        # Latch the gripper command. JointMoveSequence.get_current_action drops
        # gripper move groups, so without this the gripper controller reverts to
        # its reset (closed) target during joint moves and never opens for the
        # approach. The default TCP path avoids this via get_ctrl_dict(); mirror it.
        gmg = self.robot_view.get_gripper_movegroup_ids()[0]
        if gmg not in action:
            ctrl = self.robot_view.get_ctrl_dict()
            if gmg in ctrl:
                action[gmg] = ctrl[gmg]
        return action

    def _arm_seed(self) -> list:
        return self.robot_view.get_move_group("arm").joint_pos.tolist()

    def _ik_world(self, pose_world: np.ndarray, seed: list | None,
                  check_collision: bool = False) -> list | None:
        """Solve IK for a single 4x4 world-frame EE pose; returns arm joints or None.

        check_collision=True makes cuRobo reject configs whose collision spheres
        hit registered world obstacles (see _sync_cup_obstacle) or self-collide.
        """
        base = self.task.env.current_robot.robot_view.base.pose
        pose_base = np.linalg.inv(base) @ pose_world
        goal7 = pose_mat_to_7d(pose_base)  # [x, y, z, qw, qx, qy, qz]
        joint_config, _ = self._get_planner().ik_solve(
            goal7.tolist(), seed_config=seed, disable_collision=not check_collision
        )
        return joint_config

    def _ik_world_robust(self, pose_world: np.ndarray,
                         check_collision: bool = False) -> list | None:
        """IK seeded from the current arm config, falling back to cuRobo's
        64-random-seed solve (seed_config=None). ik_solve uses ONLY the given seed
        when one is passed, so a single seed spuriously fails reachable poses; the
        seed=None solve is what actually determines feasibility."""
        return (self._ik_world(pose_world, self._arm_seed(), check_collision)
                or self._ik_world(pose_world, None, check_collision))

    def check_feasible_ik(self, pose, check_collision: bool = False):
        single = pose.ndim == 2
        poses = [pose] if single else list(pose)
        mask = np.array([self._ik_world_robust(p, check_collision) is not None
                         for p in poses], dtype=bool)
        return bool(mask[0]) if single else mask

    def _sync_cup_obstacle(self, place_receptacle) -> list[str]:
        """Register/refresh the cup as a HOLLOW cuRobo obstacle set.

        A single solid AABB makes every over-the-rim pose "in collision", which
        forced collision checking OFF for the whole place phase. Instead the cup
        is approximated by a ring of thin wall cuboids around its AABB cylinder
        plus a bottom slab — the mouth stays free space, so cup collision can
        stay enabled for EVERY IK solve and motion plan, place phase included.

        Built from the live AABB (object.pose is the base-frame origin, half a
        cup-height too low for a volume-centered cuboid). Poses go to cuRobo in
        the robot base frame; the PiPER-X base sits at the world origin in this
        task (base_size=None), so world poses pass through unchanged. Revisit if
        the robot base ever moves.
        """
        from curobo.geom.types import Cuboid, WorldConfig

        from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

        data = self.task.env.current_data
        center, size = body_aabb(data.model, data, place_receptacle.object_id)
        cx, cy, cz = np.asarray(center).tolist()
        radius = float(min(size[0], size[1])) / 2.0
        height = float(size[2])
        thickness = 0.012
        n_walls = 8
        # bottom slab: inscribed square (side 2R/sqrt(2)) — a circumscribed 2Rx2R
        # square pokes phantom corners ~0.4R beyond the round cup; the wall ring
        # covers the perimeter anyway
        bottom_side = 2.0 * radius / np.sqrt(2.0)
        cuboids = [
            Cuboid(
                name=f"{place_receptacle.name}_bottom",
                pose=[cx, cy, cz - height / 2 + thickness / 2, 1.0, 0.0, 0.0, 0.0],
                dims=[bottom_side, bottom_side, thickness],
            )
        ]
        chord = 2.0 * radius * np.tan(np.pi / n_walls) * 1.15  # slight overlap
        for i in range(n_walls):
            ang = 2.0 * np.pi * i / n_walls
            r_mid = radius - thickness / 2.0
            yaw = ang + np.pi / 2.0  # box long axis tangent to the ring
            cuboids.append(
                Cuboid(
                    name=f"{place_receptacle.name}_wall{i}",
                    pose=[cx + r_mid * np.cos(ang), cy + r_mid * np.sin(ang), cz,
                          np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    dims=[chord, thickness, height],
                )
            )
        # every cube EXCEPT the current pickup target is an obstacle too (its
        # live AABB): neighbours on the shelf and cubes already in the cup
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        for name in PiperXCubesInCupTaskSampler.CUBE_NAMES:
            if name == self.config.task_config.pickup_obj_name:
                continue
            try:
                obj = om.get_object_by_name(name)
            except Exception:  # noqa: BLE001
                continue
            if obj.position[2] < 0.0:
                continue  # parked on the floor, far out of the workspace
            c, s = body_aabb(data.model, data, obj.object_id)
            cuboids.append(Cuboid(name=f"obs_{name}",
                                  pose=[*np.asarray(c).tolist(), 1.0, 0.0, 0.0, 0.0],
                                  dims=np.asarray(s).tolist()))

        self._get_planner().motion_gen.update_world(WorldConfig(cuboid=cuboids))
        self._cup_obstacle_cuboids = [(c.pose, c.dims) for c in cuboids]  # for viz
        return [c.name for c in cuboids]

    # Nominal in-hand offset: cube ~0.03 m along tool +z beyond the TCP/grasp_site.
    # Used only for the precomputed (feasibility-checked) place poses; the actual
    # descent is re-aimed from the measured offset by _correct_place_descent.
    _GRIP_ALONG_APPROACH = 0.03

    # Cube-center release height above the cup rim.
    _DROP_ABOVE_RIM = 0.04

    # Cup closer than this (XY center-center) steers the grasp yaw toward it.
    _NEAR_CUP_XY = 0.15

    def _analytic_grasp_pose(self, pickup_obj, place_receptacle) -> np.ndarray:
        """Grasp TCP pose from the cube pose, replacing the grasp-library +
        heuristic-scoring pipeline (whose 45deg poses gripped 2.8 cm off-center
        and pushed the cube across the shelf during the close).

        pos:   cube center.
        pitch: fixed 45 deg downward approach.
        yaw:   0 = the base's forward (+x) axis — the direction the arm faces —
               independent of where the cube is. When the cup is within
               _NEAR_CUP_XY, yaw offsets left/right by half the cup's bearing
               off that axis, capped at +/-45 deg (cup at 90deg right -> yaw 45
               right); pointing the approach toward the cup keeps the
               wrist/forearm on its far side.
        Tool axes: +z = approach, +y = finger-close axis (kept horizontal).
        """
        cube = np.asarray(pickup_obj.position)
        # yaw-0 reference: the base's forward axis, regardless of the cube
        fwd = self.task.env.current_robot.robot_view.base.pose[:2, 0]
        fwd = fwd / np.linalg.norm(fwd)
        to_cup = np.asarray(place_receptacle.position[:2]) - cube[:2]
        dist = np.linalg.norm(to_cup)
        yaw = 0.0
        if dist < self._NEAR_CUP_XY:
            to_cup /= dist
            bearing = np.arctan2(fwd[0] * to_cup[1] - fwd[1] * to_cup[0], fwd @ to_cup)
            yaw = np.clip(bearing, -np.pi / 2, np.pi / 2) / 2
        c, s = np.cos(yaw), np.sin(yaw)
        heading = np.array([c * fwd[0] - s * fwd[1], s * fwd[0] + c * fwd[1]])
        c45 = np.sqrt(0.5)
        grasp = np.eye(4)
        grasp[:3, 2] = [heading[0] * c45, heading[1] * c45, -c45]
        grasp[:3, 1] = [-heading[1], heading[0], 0.0]
        grasp[:3, 0] = np.cross(grasp[:3, 1], grasp[:3, 2])
        grasp[:3, 3] = cube
        return grasp

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        """Like the base, but the grasp pose is analytic (_analytic_grasp_pose)
        instead of selected from a pre-authored grasp library."""
        task_config = self.config.task_config
        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle = om.get_object_by_name(task_config.place_receptacle_name)

        grasp_pose_world = self._analytic_grasp_pose(pickup_obj, place_receptacle)
        self._pickup_obj = pickup_obj  # live handle for the place-descent correction

        target_poses = {}
        (target_poses["pregrasp"], target_poses["grasp"],
         target_poses["lift"]) = self._get_grasp_poses(
            grasp_pose_world=grasp_pose_world, pickup_obj=pickup_obj,
            place_receptacle=place_receptacle, robot_view=robot_view,
            task_config=task_config)
        (target_poses["preplace"], target_poses["place"],
         target_poses["postplace"]) = self._get_placement_poses(
            grasp_pose_world=grasp_pose_world, pickup_obj=pickup_obj,
            place_receptacle=place_receptacle)

        if self.task.viewer is not None:
            # only the two poses that matter: grasp (magenta) and place (cyan).
            # Colors deliberately avoid red/yellow/green, which the collision
            # overlay uses for penetration/proximity/clear.
            self._show_poses(target_poses["grasp"][None], style="tcp", color=(1, 0, 1, 1))
            self._show_poses(target_poses["place"][None], style="tcp", color=(0, 1, 1, 1))
            self.task.viewer.sync()
        return target_poses

    def _get_grasp_poses(self, grasp_pose_world, pickup_obj, place_receptacle,
                         robot_view, task_config):
        """pregrasp / grasp / lift. Unlike the base (which drives lift all the way
        to receptacle_top + clearance, ~0.32 m for the tall cup and past PiPER's
        reach), lift is just a modest straight-up raise to clear the shelf while
        keeping the grasp orientation; the level place poses handle the cup height.
        """
        # hollow cup registered once per plan; cup collision stays enabled for
        # EVERY IK solve and motion plan in the episode (the world contains
        # only the cup, so near-contact cube/shelf configs are unaffected)
        self._get_planner().enable_obstacles(self._sync_cup_obstacle(place_receptacle), True)
        def _require_ik(pose, name):
            if self.check_feasible_ik(pose, check_collision=True):
                return
            # split the failure: pure reachability vs the collision constraint
            reachable = self.check_feasible_ik(pose)
            raise ValueError(
                f"IK failed for {name} pose "
                f"({'collision-blocked' if reachable else 'unreachable'}; "
                f"pos {pose[:3, 3].round(3)})"
            )

        pregrasp = grasp_pose_world.copy()
        pregrasp[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp[:3, 2]
        _require_ik(pregrasp, "pregrasp")
        _require_ik(grasp_pose_world, "grasp")
        lift = grasp_pose_world.copy()
        lift[2, 3] += 0.08  # raise the cube ~8 cm straight up, clear of the shelf
        _require_ik(lift, "lift")
        return pregrasp, grasp_pose_world, lift

    def _get_placement_poses(self, grasp_pose_world, pickup_obj, place_receptacle):
        """Place with a LEVEL wrist instead of the 45deg grasp orientation.

        Carrying the down-tilted grasp orientation into the place makes the cube
        hang below+forward of the wrist, so reaching the tall cup's rim (~0.267 m)
        needs the wrist at ~0.30 m+ -- past PiPER's reach. A level wrist (approach
        axis horizontal, fingers opening horizontally) holds the cube at wrist
        height directly over the cup, so the wrist only needs ~rim height, and
        opening the gripper drops the cube straight down into the cup.
        """
        from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

        data = self.task.env.current_data
        center, size = body_aabb(data.model, data, place_receptacle.object_id)
        rim_z = center[2] + size[2] / 2.0
        cup_xy = np.asarray(place_receptacle.position[:2])

        # Level frame: tool +z (approach) horizontal base->cup; tool +y (finger
        # axis) horizontal & perpendicular so opening drops the cube; +x completes.
        a = np.array([cup_xy[0], cup_xy[1], 0.0])
        a = a / (np.linalg.norm(a) + 1e-9)
        y = np.cross(np.array([0.0, 0.0, 1.0]), a)
        y = y / (np.linalg.norm(y) + 1e-9)
        x = np.cross(y, a)
        R_level = np.column_stack([x, y, a])

        def tcp_for_cube(cx, cy, cz):
            T = np.eye(4)
            T[:3, :3] = R_level
            T[:3, 3] = np.array([cx, cy, cz]) - self._GRIP_ALONG_APPROACH * a
            return T

        # stashed for the closed-loop descent correction (_correct_place_descent)
        self._place_R_level = R_level
        self._place_cube_target = np.array(
            [cup_xy[0], cup_xy[1], rim_z + self._DROP_ABOVE_RIM]
        )

        preplace = tcp_for_cube(cup_xy[0], cup_xy[1], rim_z + 0.06)
        if not self.check_feasible_ik(preplace, check_collision=True):
            raise ValueError("IK failed for preplace pose")
        place = tcp_for_cube(cup_xy[0], cup_xy[1], rim_z + self._DROP_ABOVE_RIM)
        if not self.check_feasible_ik(place, check_collision=True):
            raise ValueError("IK failed for place pose")
        postplace = place.copy()
        postplace[2, 3] += 0.06  # retreat straight up
        return preplace, place, postplace

    def _tcp_to_jp_fn(self, mg_id, target_pose):
        robot_view = self.robot_view
        joint_config = self._ik_world(target_pose, seed=self._arm_seed())
        action = robot_view.get_ctrl_dict()
        if joint_config is not None:
            self.sequential_ik_failures = 0
            action["arm"] = np.asarray(joint_config)
        else:
            self.sequential_ik_failures += 1
            if self.sequential_ik_failures >= self.policy_config.max_sequential_ik_failures:
                return self._handle_failure()
        return action

    def _plan_transfer_seg(self, name, start_q, pose_world):
        """Whole collision-free trajectory to a world pose via motion_gen,
        executed on cuRobo's own timing. The cup obstacle set is always active.

        Returns (JointTrajectorySegment tracking the plan, final joints).
        The segment carries cuRobo's native interpolated plan (dense,
        jerk-limited, ``time_dilation_factor`` applied) spread over
        ``result.motion_time``; the executor samples it at the executed time,
        so the commanded motion is exactly the planner's smooth profile.
        """
        base_inv = np.linalg.inv(self.robot_view.base.pose)
        goal7 = pose_mat_to_7d(base_inv @ pose_world)
        traj, result = self._get_planner().motion_plan(list(start_q), [goal7.tolist()])
        if not bool(result.success.item()):
            raise ValueError(f"cuRobo motion plan failed for {name}")
        traj = np.asarray(traj, dtype=float)
        motion_time = max(float(result.motion_time), 0.5)
        seg = JointTrajectorySegment(
            name=name,
            times=np.linspace(0.0, motion_time, len(traj)),
            waypoints={"arm": traj},
        )
        return seg, list(traj[-1])

    def _correct_place_descent(self, seq: "_PlaceCorrectingSequence") -> None:
        """Re-aim the place descent from the measured in-hand cube pose.

        Runs once, when the descent segment becomes active (arm at preplace,
        wrist level, cube settled in the closed gripper). The cube center is
        measured in the TCP frame and the place TCP pose is recomputed so the
        cube — not the nominal _GRIP_ALONG_APPROACH point — ends up centered
        on the cup axis. The descent stays a straight-line joint move; only
        its endpoint is re-solved. Keeps the precomputed qpos if IK fails.
        """
        gripper = self.robot_view.get_gripper(self.robot_view.get_gripper_movegroup_ids()[0])
        T_tcp = gripper.leaf_frame_to_world
        cube = np.asarray(self._pickup_obj.position)
        offset_tcp = T_tcp[:3, :3].T @ (cube - T_tcp[:3, 3])

        R_level = self._place_R_level
        place = np.eye(4)
        place[:3, :3] = R_level
        place[:3, 3] = self._place_cube_target - R_level @ offset_tcp
        log.info(
            f"Place descent corrected: in-hand cube offset (TCP frame) {offset_tcp.round(4)}"
        )

        # The descent is a few cm straight down over the cup mouth: keep it a
        # straight-line joint move (seeded IK stays on the preplace IK branch).
        # motion_gen is deliberately NOT used here — for near-zero Cartesian
        # gaps it may pick a distant IK branch and swing the whole arm.
        place_seg = seq.move_segments[-1]
        jc = (self._ik_world(place, self._arm_seed(), check_collision=True)
              or self._ik_world(place, None, check_collision=True))
        if jc is None:
            log.warning("Place-descent correction IK failed; keeping precomputed pose")
            return
        place_seg.end_qpos = {"arm": np.asarray(jc)}
        q_place = jc

        # keep the retreat straight up from the corrected place pose
        retreat_seg = getattr(self, "_retreat_seg", None)
        if retreat_seg is not None:
            postplace = place.copy()
            postplace[2, 3] += 0.06
            jp = (self._ik_world(postplace, list(q_place), check_collision=True)
                  or self._ik_world(postplace, None, check_collision=True))
            if jp is not None:
                retreat_seg.start_qpos = None  # filled from live qpos at segment start
                retreat_seg.end_qpos = {"arm": np.asarray(jp)}

    def _compute_trajectory(self):
        """Long transfer moves are planned by cuRobo motion_gen as whole
        collision-free trajectories (cup registered as obstacle); the short
        structured moves (approach descent, vertical lift, place descent,
        retreat) stay straight-line by design."""
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()  # validated world-frame EE poses

        pc = self.policy_config
        start_q = robot_view.get_move_group("arm").joint_pos.tolist()

        def jseg(name, end_qpos, dur, start_qpos=None):
            return JointMoveSegment(name=name, start_qpos=start_qpos,
                                    end_qpos=end_qpos, duration_s=dur)

        # live start (home for the 1st cube, post-place retreat after) ->
        # pregrasp: planned around the cup
        pregrasp_seg, q_pregrasp = self._plan_transfer_seg(
            "pregrasp", start_q, target_poses["pregrasp"])
        qs: dict[str, dict] = {"pregrasp": {"arm": np.asarray(q_pregrasp)}}

        # remaining straight-line waypoints: seed-chained collision-checked IK,
        # falling back to home seed then cuRobo's 64-seed solve
        seed = list(q_pregrasp)
        for name in ["grasp", "lift", "preplace", "place", "postplace"]:
            jc = (self._ik_world(target_poses[name], seed, check_collision=True)
                  or self._ik_world(target_poses[name], start_q, check_collision=True)
                  or self._ik_world(target_poses[name], None, check_collision=True))
            if jc is None:
                raise ValueError(f"cuRobo IK failed for {name} pose")
            qs[name] = {"arm": np.asarray(jc)}
            seed = list(jc)

        # lift -> preplace carry: planned around the cup (the hollow cup model
        # keeps the 6-cm-over-rim preplace goal feasible — no disable fallback)
        carry_seg, q_preplace = self._plan_transfer_seg(
            "preplace", qs["lift"]["arm"].tolist(), target_poses["preplace"])
        qs["preplace"] = {"arm": np.asarray(q_preplace)}
        # re-chain the place descent from the planned carry endpoint
        jc = (self._ik_world(target_poses["place"], q_preplace, check_collision=True)
              or self._ik_world(target_poses["place"], None, check_collision=True))
        if jc is None:
            raise ValueError("cuRobo IK failed for place pose")
        qs["place"] = {"arm": np.asarray(jc)}

        self._retreat_seg = jseg("retreat", qs["postplace"], 1.0, start_qpos=qs["place"])

        return [
            GripperAction(robot_view, True, 0.0),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[
                    pregrasp_seg,
                    jseg("grasp", qs["grasp"], 1.5, start_qpos=qs["pregrasp"]),
                ],
            ),
            GripperAction(robot_view, False, pc.gripper_close_duration),
            _PlaceCorrectingSequence(
                robot_view, pc.move_settle_time,
                is_holding_object=True,
                gripper_empty_threshold=pc.gripper_empty_threshold,
                move_segments=[
                    jseg("lift", qs["lift"], 1.5, start_qpos=qs["grasp"]),
                    carry_seg,
                    jseg("place", qs["place"], 1.5, start_qpos=qs["preplace"]),
                ],
                correct_fn=self._correct_place_descent,
            ),
            GripperAction(robot_view, True, pc.gripper_open_duration),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[self._retreat_seg],
            ),
        ]

    def _go_home_primitives(self):
        """Final parking move — run once after the last cube, not per cube.

        Per-cube scripts end at the post-place retreat; the next cube's
        pregrasp transfer plans from the live (retreat) joints, so chaining
        needs no detour through home.
        """
        robot_view = self.robot_view
        pc = self.policy_config
        return [
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[JointMoveSegment(
                    name="go_home", start_qpos=None,
                    end_qpos=self.config.robot_config.init_qpos, duration_s=3.0)],
            ),
            NoopAction(robot_view, 2.0),
        ]


@register_config("PiperXCubesInCupDataGenConfig")
class PiperXCubesInCupDataGenConfig(PickAndPlaceDataGenConfig):
    """PiPER-X picks a cube and places it into the in-scene cup on the shelf."""

    scene_dataset: str = "user"
    num_workers: int = 1
    seed: int | None = 0
    filter_for_successful_trajectories: bool = False
    use_passive_viewer: bool = False
    # one pick-place cycle is ~145-175 steps (cubes chain retreat->pregrasp,
    # home only after full success); budget 4 cubes + retry slack
    task_horizon: int | None = 1400

    # Fixed-base tabletop: no 0.7 m pedestal (arm base at z=0, matching board1 top).
    robot_config: PiperXRobotConfig = PiperXRobotConfig(base_size=None)
    camera_config: PiperXCameraSystem = PiperXCameraSystem()

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PiperXCubesInCupTaskSampler,
        dataset_name="user",
        scene_xml_paths=[str(_SCENE_XML)],
        house_inds=None,  # -> pipeline uses range(len(scene_xml_paths)) = [0]
        samples_per_house=4,
        house_variant="base",
        pickup_types=[],  # [] => match any typed object; cube filtered in by grasps
        grasp_libraries=[GRASP_LIBRARY],
        filter_for_grasps=True,  # excludes the grasp-less cup from pickup candidates
        check_robot_placement_visibility=False,  # we don't drive the robot around
        randomize_textures=False,
    )
    policy_config: PickAndPlacePlannerPolicyConfig = PickAndPlacePlannerPolicyConfig(
        policy_cls=PiperXCuroboIKPickAndPlacePlannerPolicy,
        policy_factory=PiperXCuroboIKPickAndPlacePlannerPolicy,
        grasp_libraries=[GRASP_LIBRARY],
        # The tall cup on the elevated shelf puts the rim at ~0.27 m; the default
        # place_z_offset=0.07 pushes lift/preplace to ~0.355 m (beyond PiPER's
        # ~0.30 m IK ceiling at the grasp orientation). Land them at ~0.30 m.
        place_z_offset=0.015,
        # PiPER finger servo needs ~1.5 s to travel from open (0.05) and clamp a
        # 3 cm cube; the default 0.5 s left the fingers still closing at lift.
        gripper_close_duration=1.5,
        # "closed" now targets a 2 cm opening (gentle grip), so an EMPTY
        # gripper rests at ~0.020 m while the held 2.9 cm cube reads ~0.029 m.
        # Empty detection fires below range_min (0.002) + threshold = 0.024 —
        # between the two, with ~4 mm margin each way.
        gripper_empty_threshold=0.022,
    )

    output_dir: Path = Path("experiment_output") / "datagen" / "piper_x_cubes_in_cup_v1"

    @property
    def tag(self) -> str:
        return "piper_x_cubes_in_cup_datagen"
