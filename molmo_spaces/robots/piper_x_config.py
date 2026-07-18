"""Robot config for PiPER-X (AgileX PiPER 6-DOF arm + coupled parallel gripper)."""

from pathlib import Path
from typing import Any, Callable

from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.piper_x import PiperXRobot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.piper_x_view import PiperXRobotView

# assets/piper_x lives at the repo root: molmo_spaces/robots/piper_x_config.py -> parents[2]
_PIPER_X_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "piper_x"


class PiperXRobotConfig(BaseRobotConfig):
    """Configuration for the PiPER-X 6-DOF robot."""

    robot_cls: type[PiperXRobot] | None = PiperXRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = PiperXRobot
    robot_view_factory: RobotViewFactory | None = PiperXRobotView
    robot_namespace: str = "robot_0/"
    name: str = "piper_x"
    robot_xml_path: Path = Path("piper_x.xml")
    robot_dir: Path | None = _PIPER_X_ASSET_DIR
    # Base platform [width, depth, height] - raises the arm to counter height so
    # the ~0.6m-reach arm can access typical tabletop/counter objects.
    base_size: list[float] | None = [0.2, 0.2, 0.7]
    # Home config: arm folded up, offset joints 4/5 away from the wrist singularity.
    # gripper = [gripper_joint1, gripper_joint2] (coupled, opposite signs)
    init_qpos: dict[str, list[float]] = {
        "base": [],
        "arm": [0.0, 1.2, -1.2, 0.0, 0.6, 0.0],
        "gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]
