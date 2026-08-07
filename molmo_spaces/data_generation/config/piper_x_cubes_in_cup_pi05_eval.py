"""Closed-loop sim eval of a LeRobot pi05 checkpoint on the cubes-in-cup task.

Same harness as ``piper_x_cubes_in_cup_act_eval.py`` (datagen env, planner
swapped for the learned policy, ACT_EVAL_* env vars, same success judge) with
the policy half swapped for pi05:

  * PI05Policy instead of ACTPolicy (same pre/post-processor pipeline contract;
    the pi05 preprocessor additionally tokenizes the "task" string).
  * The language conditioning must match training: the dataset task string, not
    the sampler's description. Override with PI05_EVAL_TASK.
  * No delta-action anchor: this baseline trains on absolute actions.

Run:  ACT_EVAL_CKPT=... python -m molmo_spaces.data_generation.main PiperXCubesInCupPI05EvalConfig
Driver with W&B logging: piper-x-policy/policies/act/eval/act_checkpoint_eval.py
  --config-name PiperXCubesInCupPI05EvalConfig --wandb-project piper-x-pi05
"""

import logging
import os

from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_act_eval import (
    LeRobotACTPolicy,
    LeRobotACTPolicyConfig,
    PiperXCubesInCupACTEvalConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.policy.base_policy import PolicyFactory

log = logging.getLogger(__name__)

# Must equal the string in the training dataset's meta/tasks.parquet — pi05 is
# language-conditioned and was finetuned against exactly this instruction.
DEFAULT_TASK = "Pick up all the cubes and put them into the cup"


class LeRobotPI05Policy(LeRobotACTPolicy):
    """Runs a LeRobot pi05 checkpoint against the molmo_spaces obs/action dicts."""

    def prepare_model(self):
        import torch
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        if not os.path.isdir(self.checkpoint_path):
            raise FileNotFoundError(f"pi05 checkpoint dir not found: {self.checkpoint_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PI05Policy.from_pretrained(self.checkpoint_path)
        self.model.to(self.device).eval()

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
        self.task_str = os.environ.get("PI05_EVAL_TASK", DEFAULT_TASK)
        log.info(
            f"Loaded pi05 checkpoint {self.checkpoint_path} (+processors) on "
            f"{self.device}; task={self.task_str!r}"
        )

    def obs_to_model_input(self, obs):
        model_input = super().obs_to_model_input(obs)
        # language conditioning must be the training instruction, not the
        # sampler's task description
        model_input["task"] = [getattr(self, "task_str", DEFAULT_TASK)]
        return model_input

    def inference_model(self, model_input):
        import torch

        if self.model is None:
            self.prepare_model()
            model_input = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in model_input.items()
            }
            model_input["task"] = [self.task_str]
        with torch.inference_mode():
            action = self.model.select_action(self.preprocessor(dict(model_input)))
            action = self.postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "lerobot_pi05"
        return info


class LeRobotPI05PolicyConfig(LeRobotACTPolicyConfig):
    policy_cls: type = LeRobotPI05Policy
    policy_factory: PolicyFactory = LeRobotPI05Policy


@register_config("PiperXCubesInCupPI05EvalConfig")
class PiperXCubesInCupPI05EvalConfig(PiperXCubesInCupACTEvalConfig):
    """ACT eval config with the policy swapped for a trained pi05 checkpoint."""

    policy_config: LeRobotPI05PolicyConfig = LeRobotPI05PolicyConfig(
        checkpoint_path=os.environ.get("ACT_EVAL_CKPT", ""),
    )

    @property
    def tag(self) -> str:
        return "piper_x_cubes_in_cup_pi05_eval"
