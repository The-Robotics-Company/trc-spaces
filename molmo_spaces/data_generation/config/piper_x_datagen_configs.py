"""Data generation configs for the PiPER-X robot.

Registers a pick datagen config so it is discoverable by
`python -m molmo_spaces.data_generation.main PiperXPickDataGenConfig`.

Camera system references the wrist/exo cameras baked into the PiPER-X MJCF
(assets/piper_x/piper_x.xml). Modelled on examples/add_robot/xarm7_datagen.py.
"""

from pathlib import Path

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    AllCameraTypes,
    CameraSystemConfig,
    MjcfCameraConfig,
)
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.robots.piper_x_config import PiperXRobotConfig
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


class PiperXCameraSystem(CameraSystemConfig):
    # Nonstandard resolution avoids the EGL square-image issue (molmospaces #84).
    img_resolution: tuple[int, int] = (624, 352)

    cameras: list[AllCameraTypes] = [
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/",
        ),
        MjcfCameraConfig(
            name="exo_camera",
            mjcf_name="exo_camera",
            robot_namespace="robot_0/",
        ),
    ]


@register_config("PiperXPickDataGenConfig")
class PiperXPickDataGenConfig(PickBaseConfig):
    robot_config: PiperXRobotConfig = PiperXRobotConfig()
    camera_config: PiperXCameraSystem = PiperXCameraSystem()
    # Path (not str): main.py builds output_dir / name / timestamp with `/`.
    output_dir: Path = Path("experiment_output") / "datagen" / "piper_x_pick_v1"
    num_workers: int = 1
    # TODO(M2): flip back to True once PiPER-X grasp libraries exist. Until then
    # the shared 'droid' (Franka) grasps don't fit the PiPER gripper, so pick
    # success is ~0; saving all trajectories lets us validate the record path.
    filter_for_successful_trajectories: bool = False
    # Reuse the ithor house already fetched during M0 (avoids extra downloads).
    scene_dataset: str = "ithor"
    data_split: str = "train"
    seed: int | None = 0
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        dataset_name="ithor",
        house_inds=[1],
        samples_per_house=6,
        # PiPER-X has ~0.5-0.6m reach; constrain base-object radius accordingly.
        base_pose_sampling_radius_range=(0.15, 0.45),
        # Place the base (0.7m pedestal) so its top sits ~0.17m below the object,
        # putting the object inside the arm's reach envelope.
        robot_object_z_offset=-0.87,
        robot_object_z_offset_random_min=-0.1,
        robot_object_z_offset_random_max=0.1,
    )

    @property
    def tag(self) -> str:
        return "piper_x_pick_datagen"
