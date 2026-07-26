"""Headless/viewer validation of multi-cube cubes-in-cup episodes.

Samples one task (1-4 cubes + cup, randomized), runs the chained planner policy,
and per cube reports the release hover error (cube-vs-cup XY at gripper-open) and
the in-hand slip (max cube drift in the TCP frame between close and open).
At the end, counts how many of the active cubes ended up inside the cup.

VIEW=0 runs headless. Argument = seed.
"""

import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (  # noqa: E402
    PiperXCubesInCupDataGenConfig,
    PiperXCubesInCupTaskSampler,
)

_mod_log = logging.getLogger("molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen")
_mod_log.setLevel(logging.INFO)
_mod_log.addHandler(logging.StreamHandler(sys.stdout))

MAX_STEPS = 20000
CUBE_NAMES = PiperXCubesInCupTaskSampler.CUBE_NAMES


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    np.random.seed(seed)
    show = os.environ.get("VIEW", "1") == "1"

    exp_config = PiperXCubesInCupDataGenConfig()
    # the sampler reseeds from exp_config.seed during setup, which would make
    # every run identical — route the CLI seed through the config
    exp_config.seed = seed
    sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)

    task = policy = viewer = None
    for attempt in range(10):
        try:
            task = sampler.sample_task()
            pad_mu = os.environ.get("PAD_MU")  # override fingertip pad sliding friction
            if pad_mu is not None:
                m = task.env.mj_model
                pads = [g for g in range(m.ngeom)
                        if m.geom_friction[g][0] > 1.5
                        and m.body(m.geom(g).bodyid.item()).name.startswith("robot_")]
                for g in pads:
                    m.geom_friction[g][0] = float(pad_mu)
                print(f"[preview] gripper pad sliding friction -> {pad_mu} ({len(pads)} geoms)")
            policy = exp_config.policy_config.policy_factory(exp_config, task)
            task.register_policy(policy)
            if show:
                import mujoco.viewer

                mj_data = task.env.mj_datas[task.env.current_batch_index]
                viewer = mujoco.viewer.launch_passive(mj_data.model, mj_data)
                viewer.opt.sitegroup[0] = False
                task.viewer = viewer
            observation, _ = task.reset()
            break
        except Exception as e:  # noqa: BLE001
            print(f"[attempt {attempt}] sampling/reset failed: {e}")
            if viewer is not None:
                viewer.close()
            task = policy = viewer = None
    if task is None:
        print("FAIL: could not sample a feasible task")
        return 1
    mj_data = task.env.mj_datas[task.env.current_batch_index]

    om = task.env.object_managers[task.env.current_batch_index]
    cup = om.get_object_by_name(exp_config.task_config.place_receptacle_name)
    cup_xy = lambda: np.asarray(cup.position[:2])  # noqa: E731

    def on_shelf(o):
        return 0.135 < o.position[2] < 0.20

    cubes = {n: om.get_object_by_name(n) for n in CUBE_NAMES}
    active = [n for n, o in cubes.items() if on_shelf(o)]
    print(f"layout: {len(active)} active cube(s) {active}, cup {np.asarray(cup.position).round(3)}")
    for n in active:
        print(f"  {n}: {np.asarray(cubes[n].position).round(3)}")

    gr = policy.robot_view.get_gripper(policy.robot_view.get_gripper_movegroup_ids()[0])

    def in_hand_offset(obj):
        T = gr.leaf_frame_to_world
        return T[:3, :3].T @ (np.asarray(obj.position) - T[:3, 3])

    hold_phases = ("lift", "preplace", "place")
    offset0, max_drift = None, 0.0
    hovers = []
    prev_phase = None
    wall_start, sim_start = time.time(), mj_data.time
    for step in range(MAX_STEPS):
        if viewer is not None and not viewer.is_running():
            break
        action = policy.get_action(observation)
        if action is None or action.get("done"):
            break
        phase = policy.get_phase()
        target = getattr(policy, "_pickup_obj", None)  # live current-cube handle
        if phase != prev_phase:
            print(f"[step {step:5d}] phase -> {phase}")
            if phase == "gripper-close":
                offset0, max_drift = None, 0.0  # new grasp: restart slip tracking
            if (phase == "gripper-open" and prev_phase and prev_phase.startswith("place")
                    and target is not None):
                d = np.asarray(target.position[:2]) - cup_xy()
                hovers.append(float(np.linalg.norm(d)))
                print(f"  RELEASE {target.name if hasattr(target, 'name') else '?'}: "
                      f"hover XY err {hovers[-1]:.4f} m, cube z {target.position[2]:.3f}, "
                      f"in-hand slip {max_drift * 1000:.1f} mm")
            prev_phase = phase
        observation, reward, terminal, truncated, infos = task.step(action)
        if target is not None and phase and phase.startswith(hold_phases):
            if offset0 is None:
                offset0 = in_hand_offset(target)
            else:
                max_drift = max(max_drift, float(np.linalg.norm(in_hand_offset(target) - offset0)))
        if viewer is not None:
            viewer.sync()
            lag = (mj_data.time - sim_start) - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)
        if terminal or truncated:
            print(f"episode ended by task: terminal={terminal} truncated={truncated}")
            break

    in_cup = [n for n in active
              if np.linalg.norm(np.asarray(cubes[n].position[:2]) - cup_xy()) < 0.05
              and cubes[n].position[2] > 0.13]
    print(f"RESULT cubes_in_cup={len(in_cup)}/{len(active)} {in_cup} "
          f"hovers={['%.4f' % h for h in hovers]}")
    while viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
    return 0 if len(in_cup) == len(active) else 1


if __name__ == "__main__":
    sys.exit(main())
