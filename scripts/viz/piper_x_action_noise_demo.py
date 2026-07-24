"""Cubes-in-cup preview with cranked-up action noise, to visualize servo behavior.

Same episode runner as scripts/viz/piper_x_cubes_in_cup_preview.py, plus:
  - NOISE_SCALE   action_scale_factor        (default 0.1  -> try 1.0)
  - NOISE_POS_CAP max_tcp_position_noise [m] (default 0.02 -> try 0.05)
  - NOISE_ROT_CAP max_tcp_rotation_noise     (default 0.1  -> try 0.3)
  - per-step tracking of |clean commanded target - actual joints|, mapped to
    TCP space via the Jacobian, printed as mean/p99/max every 300 steps.
The point: deviation stays BOUNDED (jitter around the plan), it does not drift,
because targets are absolute waypoints servoed by the position controller.
"""

import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (  # noqa: E402
    PiperXCubesInCupDataGenConfig,
    PiperXCubesInCupTaskSampler,
)

MAX_STEPS = 20000
CUBE_NAMES = PiperXCubesInCupTaskSampler.CUBE_NAMES


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    np.random.seed(seed)
    show = os.environ.get("VIEW", "1") == "1"

    exp_config = PiperXCubesInCupDataGenConfig()
    exp_config.seed = seed

    nc = exp_config.robot_config.action_noise_config
    nc.action_scale_factor = float(os.environ.get("NOISE_SCALE", 1.0))
    nc.max_tcp_position_noise = float(os.environ.get("NOISE_POS_CAP", 0.05))
    nc.max_tcp_rotation_noise = float(os.environ.get("NOISE_ROT_CAP", 0.3))
    print(f"ACTION NOISE: scale={nc.action_scale_factor} "
          f"pos_cap={nc.max_tcp_position_noise * 100:.0f}cm rot_cap={nc.max_tcp_rotation_noise}rad "
          f"(defaults: 0.1 / 2cm / 0.1)")

    sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)

    task = policy = viewer = None
    for attempt in range(10):
        try:
            task = sampler.sample_task()
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
    print(f"layout: {len(active)} cube(s) {active}")

    robot = task.env.robots[0]
    arm_mg_id = robot.get_arm_move_group_ids()[0]
    arm_mg = robot.robot_view.get_move_group(arm_mg_id)

    devs = []  # TCP-space deviation estimate per step [m]
    prev_phase = None
    wall_start, sim_start = time.time(), mj_data.time
    for step in range(MAX_STEPS):
        if viewer is not None and not viewer.is_running():
            break
        action = policy.get_action(observation)
        if action is None or action.get("done"):
            break
        phase = policy.get_phase()
        if phase != prev_phase:
            print(f"[step {step:5d}] phase -> {phase}")
            prev_phase = phase
        observation, reward, terminal, truncated, infos = task.step(action)

        clean = robot.last_unnoised_cmd_joint_pos()
        if clean and arm_mg_id in clean:
            dq = np.asarray(clean[arm_mg_id]) - np.asarray(arm_mg.joint_pos)
            J = robot.robot_view.get_jacobian(arm_mg_id, [arm_mg_id])
            devs.append(float(np.linalg.norm((J @ dq)[:3])))
            if len(devs) % 300 == 0:
                d = np.array(devs)
                print(f"  [dev] n={len(d)} TCP dev from clean target: "
                      f"mean {d.mean() * 1000:.1f} mm · p99 {np.percentile(d, 99) * 1000:.1f} mm · "
                      f"max {d.max() * 1000:.1f} mm  (last 300: mean {d[-300:].mean() * 1000:.1f} mm)")

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
    d = np.array(devs) if devs else np.zeros(1)
    print(f"RESULT cubes_in_cup={len(in_cup)}/{len(active)} {in_cup}")
    print(f"DEVIATION over {len(devs)} steps: mean {d.mean() * 1000:.1f} mm · "
          f"p99 {np.percentile(d, 99) * 1000:.1f} mm · max {d.max() * 1000:.1f} mm — "
          f"last-quarter mean {d[-len(d) // 4:].mean() * 1000:.1f} mm (flat = bounded, no drift)")
    while viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
