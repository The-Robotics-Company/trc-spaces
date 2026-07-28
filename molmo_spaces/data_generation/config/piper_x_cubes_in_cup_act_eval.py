"""Closed-loop sim eval of a LeRobot ACT checkpoint on the cubes-in-cup task.

Mirrors the piper-x-policy eval-watcher pattern: run the LEARNED policy in the
exact datagen environment (same scene, task sampler DR, cameras, and multi-cube
success judge as ``PiperXCubesInCupDataGenConfig``) so per-checkpoint success
rates and rollout videos are apples-to-apples with the training data.

The policy consumes/produces the LeRobot dataset contract written by
``scripts/data/format_conversion/piper_x_to_lerobot.py``:

  obs    observation.images.{wrist,exo}  180x320 RGB
         observation.state (16)          [tcp eef_9d (9), gripper/GRIPPER_MAX (1), arm qpos (6)]
  action (16)                            [ee eef_9d (9), gripper cmd/GRIPPER_MAX (1), arm joint cmd (6)]

Only ``action[9:16]`` is executable: arm joint targets + gripper actuator value.

Parameterized via env vars (each eval is a fresh process; the config registry
offers no CLI overrides):

  ACT_EVAL_CKPT      path to a LeRobot ``pretrained_model`` dir (required)
  ACT_EVAL_EPISODES  episodes to roll out (default 16)
  ACT_EVAL_WORKERS   parallel sim workers (default 4)
  ACT_EVAL_SEED      episode sampling seed (default 20260728; differs from the
                     datagen seed 0 so eval layouts aren't training layouts)
  ACT_EVAL_OUT       output root (default experiment_output/act_eval)

Run:  ACT_EVAL_CKPT=... python -m molmo_spaces.data_generation.main PiperXCubesInCupACTEvalConfig
Driver with W&B logging: scripts/eval/act_checkpoint_eval.py
"""

import logging
import os
from pathlib import Path

import numpy as np

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceTaskSamplerConfig
from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
    PiperXCubesInCupDataGenConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory

log = logging.getLogger(__name__)

# Keep in sync with scripts/data/format_conversion/piper_x_to_lerobot.py:
# both the obs gripper qpos and the gripper action are actuator-units / GRIPPER_MAX.
GRIPPER_MAX = 0.05
IMG_HW = (180, 320)


def _pose7_wxyz_to_eef9d(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] -> [x,y,z, R[:,0], R[:,1]] (same as the converter)."""
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(pose7[:3], dtype=np.float32)
    rotmat = Rotation.from_quat(pose7[3:], scalar_first=True).as_matrix()
    return np.concatenate([xyz, rotmat[:, 0], rotmat[:, 1]]).astype(np.float32)


def _resize_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = hw
    if frame.shape[:2] == (h, w):
        return frame
    return np.array(Image.fromarray(frame).resize((w, h), resample=Image.BICUBIC))


class LeRobotACTPolicy(InferencePolicy):
    """Runs a LeRobot ACT checkpoint against the molmo_spaces obs/action dicts."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config, task)
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.model = None  # lazy: instantiate post-fork, once per worker process
        self.device = None
        self.preprocessor = None  # normalizer pipeline, loaded with the checkpoint
        self.postprocessor = None  # unnormalizer pipeline for predicted actions

    def reset(self):
        if self.model is not None:
            self.model.reset()  # clears the internal action-chunk queue

    def prepare_model(self):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        if not os.path.isdir(self.checkpoint_path):
            raise FileNotFoundError(f"ACT checkpoint dir not found: {self.checkpoint_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ACTPolicy.from_pretrained(self.checkpoint_path)
        self.model.to(self.device).eval()

        # lerobot >=0.4 keeps normalization OUT of the policy, in pre/post-processor
        # pipelines saved next to the weights; training runs batch = preprocessor(batch)
        # before policy.forward. Skipping them feeds the net unnormalized inputs and
        # reads its outputs as raw joint targets: measured 0.37 rad vs 0.0094 rad mean
        # joint error on ckpt 100000, i.e. 79x worse and every rollout a silent failure.
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.checkpoint_path,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": self.device}},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.checkpoint_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        log.info(f"Loaded ACT checkpoint {self.checkpoint_path} (+processors) on {self.device}")

    def obs_to_model_input(self, obs):
        import torch

        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        try:
            wrist = obs["wrist_camera"]
            exo = obs["exo_camera"]
            qpos = obs["qpos"]
            tcp_pose = obs["tcp_pose"]
        except KeyError as e:
            raise KeyError(f"obs missing {e}; available keys: {sorted(obs.keys())}") from e

        eef9d = _pose7_wxyz_to_eef9d(np.asarray(tcp_pose, dtype=np.float32))
        grip = np.asarray([qpos["gripper"][0] / GRIPPER_MAX], dtype=np.float32)
        arm = np.asarray(qpos["arm"], dtype=np.float32)
        state = np.concatenate([eef9d, grip, arm])

        def img(frame):
            t = torch.from_numpy(_resize_rgb(frame, IMG_HW).copy()).float() / 255.0
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        def vec(x):
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)

        # sub-key features included too: policies trained with the dataset's full
        # feature set normalize every configured input, and missing keys raise
        return {
            "observation.images.wrist": img(wrist),
            "observation.images.exo": img(exo),
            "observation.state": vec(state),
            "observation.state.eef_9d": vec(eef9d),
            "observation.state.gripper_position": vec(grip),
            "observation.state.joint_position": vec(arm),
            "task": [self.task.get_task_description()],
        }

    def inference_model(self, model_input):
        import torch

        if self.model is None:
            self.prepare_model()
            model_input = {  # re-place tensors: device was unknown before load
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in model_input.items()
            }
        with torch.inference_mode():
            # normalize in, unnormalize out — the pipelines are part of the checkpoint
            action = self.model.select_action(self.preprocessor(dict(model_input)))
            action = self.postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()

    def model_output_to_action(self, model_output):
        # action[0:9] is the ee-pose head (not executable here); drive the arm
        # with the joint head and the gripper with its actuator command.
        arm = np.asarray(model_output[10:16], dtype=np.float64)
        grip = float(np.clip(model_output[9], 0.0, 1.0)) * GRIPPER_MAX
        return {"arm": arm, "gripper": np.array([grip])}

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "lerobot_act"
        info["policy_checkpoint"] = str(self.checkpoint_path)
        return info


class LeRobotACTPolicyConfig(BasePolicyConfig):
    policy_cls: type = LeRobotACTPolicy
    policy_factory: PolicyFactory = LeRobotACTPolicy
    policy_type: str = "learned"
    checkpoint_path: str = ""


@register_config("PiperXCubesInCupACTEvalConfig")
class PiperXCubesInCupACTEvalConfig(PiperXCubesInCupDataGenConfig):
    """Datagen config with the cuRobo planner swapped for a trained ACT policy."""

    num_workers: int = int(os.environ.get("ACT_EVAL_WORKERS", "4"))
    seed: int | None = int(os.environ.get("ACT_EVAL_SEED", "20260728"))
    filter_for_successful_trajectories: bool = False  # record failures: eval, not data

    policy_config: LeRobotACTPolicyConfig = LeRobotACTPolicyConfig(
        checkpoint_path=os.environ.get("ACT_EVAL_CKPT", ""),
    )

    task_sampler_config: PickAndPlaceTaskSamplerConfig = (
        PiperXCubesInCupDataGenConfig.model_fields["task_sampler_config"].default.model_copy(
            update={"samples_per_house": int(os.environ.get("ACT_EVAL_EPISODES", "16"))}
        )
    )

    output_dir: Path = Path(os.environ.get("ACT_EVAL_OUT", "experiment_output/act_eval"))

    @property
    def tag(self) -> str:
        return "piper_x_cubes_in_cup_act_eval"
