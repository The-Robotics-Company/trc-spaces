"""Watch and measure in-hand slip live: cube pose tracked in the gripper frame.

Opens the interactive MuJoCo viewer on a cubes-in-cup rollout and, for every
policy step where the cube is in the hand, records the cube's translation and
its rotation about the grip (pad-normal) axis relative to where it sat at
grasp time. Prints running numbers; final summary separates linear slip from
torsional spin so the slip mode is measured, not guessed.

    conda activate mlspaces
    TORSIONAL=0.05 python scripts/viz/piper_x_grip_slip_probe.py [seed]

TORSIONAL overrides friction[1] on both finger pads (authored 0.05).
IMPRATIO overrides model option impratio (authored 1.0).
CONE=elliptic switches the friction cone (authored pyramidal).
NOSLIP overrides noslip_iterations (authored 0).
VIEW=0 runs headless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "viz"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from record_sampling_demos import NOISE_DEFAULT, _make_task, _set_noise  # noqa: E402


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    torsional = float(os.environ.get("TORSIONAL", "0.05"))
    view = os.environ.get("VIEW", "1") != "0"

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
    )

    cfg = PiperXCubesInCupDataGenConfig()
    _set_noise(cfg, NOISE_DEFAULT)
    _, task = _make_task(cfg, seed)
    policy = cfg.policy_config.policy_factory(cfg, task)
    task.register_policy(policy)
    obs, _ = task.reset()
    env = task.env
    data = env.current_data
    model = env.current_model

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    pads = {}
    for name, tag in (("robot_0/gripper_link1", "f1"), ("robot_0/gripper_link2", "f2")):
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == bid(name) and model.geom_contype[i]:
                pads[i] = tag
                model.geom_friction[i][1] = torsional
    cubes = {}
    for cn in ("cube", "cube2_cube", "cube3_cube", "cube4_cube"):
        b = bid(cn)
        for i in range(model.ngeom):
            if b >= 0 and model.geom_bodyid[i] == b and model.geom_contype[i]:
                cubes[i] = cn
    grip_body = bid("robot_0/gripper_base")
    model.opt.impratio = float(os.environ.get("IMPRATIO", model.opt.impratio))
    if os.environ.get("CONE", "").lower() == "elliptic":
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.noslip_iterations = int(os.environ.get("NOSLIP", model.opt.noslip_iterations))
    print(f"[slip] seed {seed} · pad torsional friction = {torsional} · "
          f"impratio {model.opt.impratio} · "
          f"cone {'elliptic' if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else 'pyramidal'} · "
          f"noslip {model.opt.noslip_iterations}")

    viewer = None
    if view:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = (0.32, 0.0, 0.15)
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 160, -25, 1.0

    def grip_T():
        T = np.eye(4)
        T[:3, :3] = data.xmat[grip_body].reshape(3, 3)
        T[:3, 3] = data.xpos[grip_body]
        return T

    def cube_T(b):
        T = np.eye(4)
        T[:3, :3] = data.xmat[b].reshape(3, 3)
        T[:3, 3] = data.xpos[b]
        return T

    # per held cube: pose in gripper frame at grasp onset
    ref: dict[str, np.ndarray] = {}
    series: dict[str, list] = {}
    try:
        for t in range(20000):
            action = policy.get_action(obs)
            if action is None or action.get("done"):
                break
            obs, _r, term, trunc, _i = task.step(action)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            touching = {}
            for ci in range(data.ncon):
                con = data.contact[ci]
                g1, g2 = con.geom1, con.geom2
                p = g1 if g1 in pads else (g2 if g2 in pads else None)
                c = g1 if g1 in cubes else (g2 if g2 in cubes else None)
                if p is not None and c is not None:
                    touching.setdefault(cubes[c], set()).add(pads[p])
            for cn, fingers in touching.items():
                if len(fingers) < 2:
                    continue
                T = np.linalg.inv(grip_T()) @ cube_T(bid(cn))
                if cn not in ref:
                    ref[cn] = T
                    series[cn] = []
                    print(f"[slip] t={t}: grasped {cn}")
                    continue
                D = np.linalg.inv(ref[cn]) @ T
                disp = D[:3, 3] * 1000.0
                # rotation split: about grip x-axis (pad normal) vs total
                ang_axis = np.degrees(
                    np.arctan2(D[2, 1] - D[1, 2], D[1, 1] + D[2, 2])
                )
                ang_total = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
                series[cn].append((t, *disp, ang_axis, ang_total))
                if len(series[cn]) % 25 == 0:
                    print(f"[slip] t={t} {cn}: slid {np.linalg.norm(disp):.1f} mm · "
                          f"spun {ang_axis:+.1f}° about pad normal ({ang_total:.1f}° total)")
            if term or trunc:
                break
    finally:
        if viewer is not None:
            viewer.close()

    print(f"\n[slip] episode: {t} steps · success {bool(task.judge_success())}")
    for cn, rows in series.items():
        if not rows:
            continue
        a = np.array(rows)
        d = np.linalg.norm(a[:, 1:4], axis=1)
        print(f"[slip] {cn}: held {len(rows)} steps")
        print(f"[slip]   translation in gripper frame: final {d[-1]:.1f} mm · max {d.max():.1f} mm "
              f"(x {a[-1,1]:+.1f}, y {a[-1,2]:+.1f}, z {a[-1,3]:+.1f})")
        print(f"[slip]   spin about pad normal: final {a[-1,4]:+.1f}° · max |{np.abs(a[:,4]).max():.1f}|° "
              f"· total rot final {a[-1,5]:.1f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
