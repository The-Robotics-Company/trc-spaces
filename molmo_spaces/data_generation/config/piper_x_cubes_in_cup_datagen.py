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
    NoopAction,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.robots.piper_x_config import PiperXRobotConfig
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.pose import pose_mat_to_7d

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

    # Shelf sampling regions in the robot base frame, constrained to the PiPER
    # reach sweet spot: board2 top spans x in [0.27, 0.62], y in [-0.35, 0.35] at
    # z=0.13; cuRobo IK at the 45° grasp orientation solves reliably for
    # x in ~[0.32, 0.40], |y| <= ~0.13, z up to ~0.30 m.
    REGION_X = (0.33, 0.40)
    REGION_Y = (-0.12, 0.12)
    CUBE_Z = 0.144  # cube center; seated ~1 mm into shelf so a contact registers
    CUP_Z = 0.129  # cup body origin; base seated ~1 mm into shelf
    MIN_CUBE_CUP = 0.13  # min XY center-center distance
    _MAX_PLACE_TRIES = 200

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

    def _sample_shelf_xy(self) -> np.ndarray:
        return np.array([np.random.uniform(*self.REGION_X),
                         np.random.uniform(*self.REGION_Y)])

    @staticmethod
    def _pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
        return pose

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Keep the fixed-base arm at the origin; sample cube + cup on the shelf."""
        task_cfg = self.config.task_config
        robot_view = env.current_robot.robot_view

        # Fixed-base tabletop arm (base_size=None => arm base at z=0). Not moved;
        # just record the base pose.
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        cup_xy = self._sample_shelf_xy()
        cup = create_mlspaces_body(env.current_data, self.place_receptacle_name)
        cup.pose = self._pose(cup_xy[0], cup_xy[1], self.CUP_Z,
                              yaw=np.random.uniform(-np.pi, np.pi))

        cube_xy = self._sample_shelf_xy()
        for _ in range(self._MAX_PLACE_TRIES):
            if np.linalg.norm(cube_xy - cup_xy) >= self.MIN_CUBE_CUP:
                break
            cube_xy = self._sample_shelf_xy()
        cube = create_mlspaces_body(env.current_data, task_cfg.pickup_obj_name)
        cube.pose = self._pose(cube_xy[0], cube_xy[1], self.CUBE_Z,
                               yaw=np.random.uniform(-np.pi, np.pi))

        mujoco.mj_fwdPosition(env.current_model, env.current_data)

        # The policy reads the cup pose live at plan time; these are bookkeeping.
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(cube.pose).tolist()
        goal_pose = pose_mat_to_7d(cube.pose)
        goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = goal_pose.tolist()
        task_cfg.place_receptacle_start_pose = pose_mat_to_7d(cup.pose).tolist()


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
                )
            )
        return cls._planner

    def get_action(self, info):
        action = super().get_action(info)
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

    def _ik_world(self, pose_world: np.ndarray, seed: list | None) -> list | None:
        """Solve IK for a single 4x4 world-frame EE pose; returns arm joints or None."""
        base = self.task.env.current_robot.robot_view.base.pose
        pose_base = np.linalg.inv(base) @ pose_world
        goal7 = pose_mat_to_7d(pose_base)  # [x, y, z, qw, qx, qy, qz]
        joint_config, _ = self._get_planner().ik_solve(
            goal7.tolist(), seed_config=seed, disable_collision=True
        )
        return joint_config

    def _ik_world_robust(self, pose_world: np.ndarray) -> list | None:
        """IK seeded from the current arm config, falling back to cuRobo's
        64-random-seed solve (seed_config=None). ik_solve uses ONLY the given seed
        when one is passed, so a single seed spuriously fails reachable poses; the
        seed=None solve is what actually determines feasibility."""
        return (self._ik_world(pose_world, self._arm_seed())
                or self._ik_world(pose_world, None))

    def check_feasible_ik(self, pose):
        single = pose.ndim == 2
        poses = [pose] if single else list(pose)
        mask = np.array([self._ik_world_robust(p) is not None for p in poses], dtype=bool)
        return bool(mask[0]) if single else mask

    # The cube sits ~0.03 m along the tool +z (approach) beyond the TCP/grasp_site
    # (measured in the assembled scene); used to place the CUBE over the cup.
    _GRIP_ALONG_APPROACH = 0.03

    def _get_grasp_poses(self, grasp_pose_world, pickup_obj, place_receptacle,
                         robot_view, task_config):
        """pregrasp / grasp / lift. Unlike the base (which drives lift all the way
        to receptacle_top + clearance, ~0.32 m for the tall cup and past PiPER's
        reach), lift is just a modest straight-up raise to clear the shelf while
        keeping the grasp orientation; the level place poses handle the cup height.
        """
        pregrasp = grasp_pose_world.copy()
        pregrasp[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp[:3, 2]
        if not self.check_feasible_ik(pregrasp):
            raise ValueError("IK failed for pregrasp pose")
        if not self.check_feasible_ik(grasp_pose_world):
            raise ValueError("IK failed for grasp pose")
        lift = grasp_pose_world.copy()
        lift[2, 3] += 0.08  # raise the cube ~8 cm straight up, clear of the shelf
        if not self.check_feasible_ik(lift):
            raise ValueError("IK failed for lift pose")
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

        preplace = tcp_for_cube(cup_xy[0], cup_xy[1], rim_z + 0.06)
        if not self.check_feasible_ik(preplace):
            raise ValueError("IK failed for preplace pose")
        place = tcp_for_cube(cup_xy[0], cup_xy[1], rim_z + 0.015)
        if not self.check_feasible_ik(place):
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

    def _compute_trajectory(self):
        """Solve cuRobo IK per waypoint (seed-chained) and drive in joint space."""
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()  # validated world-frame EE poses

        order = ["pregrasp", "grasp", "lift", "preplace", "place", "postplace"]
        seed = robot_view.get_move_group("arm").joint_pos.tolist()
        home_seed = list(seed)
        qs: dict[str, dict] = {}
        for name in order:
            # Seed-chain for joint-space continuity, but fall back to broader
            # seeding when a single seed lands in a bad IK basin (ik_solve uses
            # only the given seed; seed=None triggers cuRobo's 64-random-seed
            # solve = robust, matching the base feasibility check).
            jc = self._ik_world(target_poses[name], seed)
            if jc is None:  # retry from the folded-home seed
                jc = self._ik_world(target_poses[name], home_seed)
            if jc is None:  # last resort: cuRobo multi-seed solve (no continuity)
                jc = self._ik_world(target_poses[name], None)
            if jc is None:
                raise ValueError(f"cuRobo IK failed for {name} pose")
            qs[name] = {"arm": np.asarray(jc)}
            seed = list(jc)  # chain for joint-space continuity

        pc = self.policy_config

        def jseg(name, end_qpos, dur, start_qpos=None):
            return JointMoveSegment(name=name, start_qpos=start_qpos,
                                    end_qpos=end_qpos, duration_s=dur)

        return [
            GripperAction(robot_view, True, 0.0),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[
                    jseg("pregrasp", qs["pregrasp"], 2.5),
                    jseg("grasp", qs["grasp"], 1.5, start_qpos=qs["pregrasp"]),
                ],
            ),
            GripperAction(robot_view, False, pc.gripper_close_duration),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                is_holding_object=True,
                gripper_empty_threshold=pc.gripper_empty_threshold,
                move_segments=[
                    jseg("lift", qs["lift"], 1.5, start_qpos=qs["grasp"]),
                    jseg("preplace", qs["preplace"], 2.0, start_qpos=qs["lift"]),
                    jseg("place", qs["place"], 1.5, start_qpos=qs["preplace"]),
                ],
            ),
            GripperAction(robot_view, True, pc.gripper_open_duration),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[jseg("retreat", qs["postplace"], 1.0, start_qpos=qs["place"])],
            ),
            JointMoveSequence(
                robot_view, pc.move_settle_time,
                move_segments=[jseg("go_home", self.config.robot_config.init_qpos, 3.0)],
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
    # level-place trajectory is a few s longer than the default 400-step budget
    task_horizon: int | None = 600

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
    )

    output_dir: Path = Path("experiment_output") / "datagen" / "piper_x_cubes_in_cup_v1"

    @property
    def tag(self) -> str:
        return "piper_x_cubes_in_cup_datagen"
