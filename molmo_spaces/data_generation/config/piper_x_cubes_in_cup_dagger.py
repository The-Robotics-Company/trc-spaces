"""Switch-point DAgger collector for the cubes-in-cup task.

Run-3 post-mortem (piper-x-policy/docs/act_run3_why_10x_data_failed.md): the
expert data is so proprio-predictable that BC never learns to use vision, and
contains zero recovery behavior. This collector generates exactly the missing
(student-visited state -> expert corrective action) pairs:

  1. The trained ACT student drives the episode for a random prefix
     (uniform switch step in [switch_min, switch_max]), reaching the
     slightly-off-distribution states BC actually visits at eval time.
  2. At the switch step, the cuRobo expert replans FROM THAT STATE (the same
     re-entry path ``_advance_to_next_cube`` uses between cubes) and finishes
     the episode. Its actions are executed and recorded as usual.

Only the expert suffix is training data; the switch step is journaled into the
per-episode sampling sidecar (``sampling.dagger.switch_step``) so the lerobot
converter can slice off the student prefix. Everything else — scene, DR,
cameras, success judge, h5+mp4 saving — is inherited from
``PiperXCubesInCupDataGenConfig``, so DAgger episodes are format-identical to
the base v2 dataset.

Known caveat (accepted): if the student is HOLDING a cube at the switch step,
the expert's script starts with gripper-open and drops it; the episode then
recovers by re-picking (cube lands on the shelf) or fails the success judge
and is filtered at conversion. No phase re-entry is attempted.

Env vars (fresh process per run; registry has no CLI overrides):
  DAGGER_CKPT        LeRobot ``pretrained_model`` dir of the student (required)
  DAGGER_EPISODES    episodes to attempt (default 100)
  DAGGER_WORKERS     parallel sim workers (default 16)
  DAGGER_SEED        sampling seed (default 7; differs from datagen 0 and
                     eval 20260728 so layouts overlap neither)
  DAGGER_SWITCH_MIN / DAGGER_SWITCH_MAX
                     switch-step range (default 20..600; ~433 = mean episode)
  DAGGER_OUT         output root (default experiment_output/datagen/
                     piper_x_cubes_in_cup_dagger_v1)

Run:
  DAGGER_CKPT=... python -m molmo_spaces.data_generation.main PiperXCubesInCupDAggerConfig
"""

import logging
import os
from pathlib import Path

import numpy as np

from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceTaskSamplerConfig
from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_act_eval import (
    LeRobotACTPolicy,
)
from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
    GRASP_LIBRARY,
    PiperXCubesInCupDataGenConfig,
    PiperXCubesInCupTaskSampler,
    PiperXCuroboIKPickAndPlacePlannerPolicy,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.policy.base_policy import PolicyFactory

log = logging.getLogger(__name__)

# The policy has no handle on the per-worker task sampler, but the switch step
# must land in the sampler's per-episode record to reach the episode sidecar
# JSON. Both objects live in the same worker process: the sampler registers
# itself here, the policy journals through it. Best-effort by design.
_CURRENT_SAMPLER = None


class PiperXCubesInCupDaggerTaskSampler(PiperXCubesInCupTaskSampler):
    """Stock sampler that additionally exposes itself for policy-side journaling."""

    def sample_task(self, *args, **kwargs):
        global _CURRENT_SAMPLER
        _CURRENT_SAMPLER = self
        return super().sample_task(*args, **kwargs)


class PiperXCubesInCupDaggerPolicy(PiperXCuroboIKPickAndPlacePlannerPolicy):
    """ACT student for a random prefix, cuRobo expert from the switch step on."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config, task)
        self._student = LeRobotACTPolicy(exp_config, task)
        self._switch_step: int = 0
        self._switched = False

    def reset(self, reset_retries: bool = True):
        if self.task is not None and self.task.episode_step_count == 0:
            self._switched = False
            pc = self.config.policy_config
            # global np stream: seeded per work item by seed_task_sampling, so
            # draws vary per episode AND are reproducible (policy instances are
            # recreated per episode — instance-local RNG would repeat one value)
            self._switch_step = int(np.random.randint(pc.switch_min, pc.switch_max + 1))
            self._student.task = self.task
            self._student.reset()
            if _CURRENT_SAMPLER is not None:
                _CURRENT_SAMPLER.record_sampling(
                    "dagger",
                    switch_step=self._switch_step,
                    student_ckpt=str(pc.checkpoint_path),
                )
        # expert reset: at t=0 this also initializes _skip_cubes/_episode_cubes
        # and plans a (discarded) initial script — cheap relative to a rollout,
        # and keeps the parent's episode-start bookkeeping on its normal path
        super().reset(reset_retries)

    def get_action(self, info):
        if not self._switched and self.task.episode_step_count >= self._switch_step:
            self._switched = True
            log.info(
                f"DAgger switch at step {self.task.episode_step_count}: "
                "expert replanning from student-visited state"
            )
            # replan from the live sim state; targets the nearest shelf cube.
            # False => nothing pickable remains (student knocked cubes off or
            # finished) — fall through, the expert script then ends the episode.
            self._advance_to_next_cube()
        if self._switched:
            # the student can leave cubes where the expert cannot plan (shoved
            # past the reach envelope, wedged against the cup): planning raises
            # ValueError through retry-reset. Blacklist that cube and move on;
            # end the episode when nothing pickable remains (failure -> filtered).
            for _ in range(len(PiperXCubesInCupTaskSampler.CUBE_NAMES) + 1):
                try:
                    return super().get_action(info)
                except ValueError as e:
                    log.warning(f"Expert planning failed mid-episode ({e}); skipping cube")
                    self._skip_cubes.add(self.config.task_config.pickup_obj_name)
                    if not self._advance_to_next_cube():
                        break
            action = self.robot_view.get_noop_ctrl_dict()
            action["done"] = True
            action["success"] = False
            return action
        action = self._student.get_action(info)
        # student never terminates the episode — the expert suffix must run
        action.pop("done", None)
        return action


class DaggerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    checkpoint_path: str = ""
    switch_min: int = int(os.environ.get("DAGGER_SWITCH_MIN", "20"))
    switch_max: int = int(os.environ.get("DAGGER_SWITCH_MAX", "600"))


@register_config("PiperXCubesInCupDAggerConfig")
class PiperXCubesInCupDAggerConfig(PiperXCubesInCupDataGenConfig):
    """Datagen config with the DAgger student-prefix/expert-suffix policy."""

    num_workers: int = int(os.environ.get("DAGGER_WORKERS", "16"))
    seed: int | None = int(os.environ.get("DAGGER_SEED", "7"))
    # keep failures out of the dataset at the source; debug 1% still saved
    filter_for_successful_trajectories: bool = True

    policy_config: DaggerPolicyConfig = DaggerPolicyConfig(
        policy_cls=PiperXCubesInCupDaggerPolicy,
        policy_factory=PiperXCubesInCupDaggerPolicy,
        checkpoint_path=os.environ.get("DAGGER_CKPT", ""),
        grasp_libraries=[GRASP_LIBRARY],
        # keep in sync with PiperXCubesInCupDataGenConfig.policy_config
        place_z_offset=0.015,
        gripper_close_duration=1.5,
        gripper_empty_threshold=0.022,
    )

    task_sampler_config: PickAndPlaceTaskSamplerConfig = (
        PiperXCubesInCupDataGenConfig.model_fields["task_sampler_config"].default.model_copy(
            update={
                "task_sampler_class": PiperXCubesInCupDaggerTaskSampler,
                "samples_per_house": int(os.environ.get("DAGGER_EPISODES", "100")),
            }
        )
    )

    output_dir: Path = Path(
        os.environ.get(
            "DAGGER_OUT", "experiment_output/datagen/piper_x_cubes_in_cup_dagger_v1"
        )
    )

    @property
    def tag(self) -> str:
        return "piper_x_cubes_in_cup_dagger"
