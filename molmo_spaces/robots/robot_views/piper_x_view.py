"""
Robot view + move groups for the PiPER-X robot.

PiPER-X is a 6-DOF arm (AgileX PiPER) with a coupled parallel gripper, so its
structure mirrors the i2rt YAM (see robot_views/i2rt_yam_view.py). The two
gripper fingers (gripper_joint1/gripper_joint2) are driven by a single actuator
and coupled by an equality constraint (gripper_joint2 = -gripper_joint1) in the
MJCF (assets/piper_x/piper_x.xml).
"""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class PiperXBaseGroup(MocapRobotBaseGroup):
    """Mocap base group. The 'base' body is created at runtime by
    PiperXRobot.add_robot_to_scene; the arm root ('base_link') is attached under it."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class PiperXArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """6-DOF arm group (joint1..joint6)."""

    def __init__(
        self,
        mj_data: MjData,
        base_group: PiperXBaseGroup,
        namespace: str = "",
        grasp_site_name: str = "grasp_site",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}joint{i + 1}").id for i in range(6)]
        act_ids = [model.actuator(f"{namespace}joint{i + 1}").id for i in range(6)]
        self._arm_root_id = model.body(f"{namespace}base_link").id
        self._ee_site_id = model.site(f"{namespace}{grasp_site_name}").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class PiperXGripperGroup(MJCFFrameMixin, GripperGroup):
    """Coupled parallel gripper group.

    The two finger joints move in opposite directions (gripper_joint2 =
    -gripper_joint1 via an equality constraint) and are driven by a single
    'gripper' actuator (ctrlrange 0.0..0.05).
    """

    def __init__(self, mj_data: MjData, base_group: PiperXBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}gripper_joint1").id,
            model.joint(f"{namespace}gripper_joint2").id,
        ]
        act_ids = [model.actuator(f"{namespace}gripper").id]
        root_body_id = model.body(f"{namespace}gripper_base").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        """gripper actuator ctrlrange is 0.0 (closed) .. 0.05 (open)."""
        self.ctrl = [0.05 if open else 0.0]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        """(min, max) opening. Fingers travel 0..0.05 each in opposite directions;
        measured via find_gripper_finger_range: 0.002 (closed) .. 0.098 (open)."""
        return 0.002, 0.098

    @property
    def inter_finger_dist(self) -> float:
        """Total opening = sum of |finger positions| (fingers are coupled, opposite signs)."""
        return float(np.abs(self.joint_pos[0]) + np.abs(self.joint_pos[1]))

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class PiperXRobotView(RobotView):
    """Robot view for the PiPER-X 6-DOF arm with coupled parallel gripper."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = PiperXBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": PiperXArmGroup(mj_data, base, namespace=namespace),
            "gripper": PiperXGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}piper_x"

    @property
    def base(self) -> PiperXBaseGroup:
        return self._move_groups["base"]
