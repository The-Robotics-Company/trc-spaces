"""Offline re-render of a saved PiPER-X cubes-in-cup episode: simulate once, render many.

The post-hoc DR axis from docs/dr_and_task_sampling_map.md. The repo's
MultithreadedDomainRandomizationOfflineOpenGLRenderer is a stub — it expects
state_data.npz / task_metadata.json files nothing in datagen writes, and its
_randomize_lights() touches a model.light_rgba field MuJoCo doesn't have — so
this script implements the capability against the real episode format
(trajectories*.h5) and the real per-episode randomizers:

  1. load traj_<i>: frozen SavedEpisode (exact cube/cup poses, robot init
     qpos, frozen cameras) + the per-step CLEAN commanded actions
  2. rebuild the scene, re-execute the actions once with action noise
     DISABLED, capturing the full sim qpos per policy step (the state tape)
  3. re-render the tape N times, one fresh LightingRandomizer (+ stock
     TextureRandomizer) draw per pass — same trajectory, new visuals

Validation media: the replayed exo_camera video next to the dataset's original
episode mp4, plus the measured per-step TCP gap replay-vs-recorded.

Known limitation (measured): SavedEpisode only freezes the task-relevant
objects (pickup cube + cup). Episodes whose original layout had extra cubes on
the shelf replay with those cubes parked on the floor instead.

    conda activate mlspaces
    MUJOCO_GL=egl python scripts/viz/piper_x_offline_rerender.py \
        [--h5 <trajectories.h5>] [--traj 0] [--draws 3] [--out results/sampling_demos]
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

def _default_h5() -> Path:
    """Newest cubes-in-cup datagen run's house_0 trajectories file."""
    root = REPO / "experiment_output/datagen/piper_x_cubes_in_cup_v1/PiperXCubesInCupDataGenConfig"
    runs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    for run in reversed(runs):
        h5 = run / "house_0" / "trajectories_batch_1_of_1.h5"
        if h5.exists():
            return h5
    raise SystemExit(f"no trajectories h5 found under {root} — pass --h5")


def load_episode(h5_path: Path, traj: int):
    """Per-step clean actions + recorded state + the frozen SavedEpisode."""
    import h5py
    import numpy as np

    def decode_rows(ds):
        return [json.loads(bytes(row).rstrip(b"\x00")) for row in ds]

    with h5py.File(h5_path) as f:
        t = f[f"traj_{traj}"]
        scene = json.loads(bytes(t["obs_scene"][()]))
        saved = pickle.loads(base64.b64decode(scene["frozen_config"]))
        return {
            "actions": decode_rows(t["actions/commanded_action"]),
            "qpos0": decode_rows(t["obs/agent/qpos"][:1])[0],
            "tcp_pose": np.array(t["obs/extra/tcp_pose"]),
            "success": bool(t["success"][-1]),
            "saved": saved,
        }


def restore_texture_defaults(rnd, model) -> None:
    """TextureRandomizer has no restore_defaults(); write authored values back."""
    for name, d in rnd._defaults.items():
        gid = next(g for g, dd in rnd._geom_id_to_defaults.items() if dd is d)
        model.geom_rgba[gid] = d["geom_rgba"]
        if d["mat_id"] >= 0:
            model.geom_matid[gid] = d["mat_id"]
            model.mat_rgba[d["mat_id"]] = d["mat_rgba"]
            model.mat_specular[d["mat_id"]] = d["mat_specular"]
            model.mat_shininess[d["mat_id"]] = d["mat_shininess"]
            if d["texture_id"] >= 0:
                texid = model.mat_texid[d["mat_id"]]
                if getattr(texid, "ndim", 0):  # (mjNTEXROLE,) row: RGB role = 1
                    model.mat_texid[d["mat_id"], 1] = d["texture_id"]
                else:
                    model.mat_texid[d["mat_id"]] = d["texture_id"]


def rebuild_scene(saved, seed: int = 0):
    """Fresh task with DR randomizers available, forced to the saved episode state."""
    import numpy as np

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
        PiperXCubesInCupTaskSampler,
    )
    from molmo_spaces.env.data_views import create_mlspaces_body

    cfg = PiperXCubesInCupDataGenConfig()
    cfg.robot_config.action_noise_config.enabled = False  # deterministic re-execution
    ts = cfg.task_sampler_config
    ts.randomize_lighting = True  # constructs the randomizers + bakes the texture pool
    ts.randomize_textures = True
    ts.randomize_textures_all = True
    ts.randomize_dynamics = False

    np.random.seed(seed)
    cfg.seed = seed
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    task = sampler.sample_task()
    env = task.env
    data = env.current_data

    # sample_task already rolled lighting + textures once — back to authored
    sampler.lighting_randomizer.restore_defaults()
    restore_texture_defaults(sampler.texture_randomizer, env.current_model)

    # force the saved episode state: recorded objects to their exact poses,
    # every other cube to its parking spot (original episodes with >1 shelf
    # cube are only partially frozen — see module docstring)
    poses = saved.task_config.object_poses
    for i, name in enumerate(PiperXCubesInCupTaskSampler.CUBE_NAMES):
        body = create_mlspaces_body(data, name)
        if name in poses:
            body.position = np.array(poses[name][:3])
            body.quat = np.array(poses[name][3:])
        else:
            px, py = PiperXCubesInCupTaskSampler._PARK_XY[i]
            body.position = np.array([px, py, PiperXCubesInCupTaskSampler._PARK_Z])
            body.quat = np.array([1.0, 0.0, 0.0, 0.0])
    cup = create_mlspaces_body(data, "cup")
    cup.position = np.array(poses["cup"][:3])
    cup.quat = np.array(poses["cup"][3:])

    return cfg, sampler, task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=None,
                    help="trajectories h5 (default: newest datagen run)")
    ap.add_argument("--traj", type=int, default=None,
                    help="trajectory index (default: first successful one, else 0)")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "sampling_demos")
    ap.add_argument("--fragment", action="store_true",
                    help="write manifest_offline_rerender.json for the data viewer")
    args = ap.parse_args()
    if args.h5 is None:
        args.h5 = _default_h5()
    if args.traj is None:
        import h5py

        with h5py.File(args.h5) as f:
            oks = [i for i in range(len(f)) if bool(f[f"traj_{i}"]["success"][-1])]
        args.traj = oks[0] if oks else 0
    print(f"[rerender] episode source: {args.h5} traj_{args.traj}")
    args.out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from molmo_spaces.utils.pose import pose_mat_to_7d

    ep = load_episode(args.h5, args.traj)
    cfg, sampler, task = rebuild_scene(ep["saved"])
    env = task.env
    data = env.current_data
    model = env.current_model

    # robot to the recorded step-0 state
    robot = env.robots[0]
    rv = robot.robot_view
    for mg_id in ("arm", "gripper"):
        rv.get_move_group(mg_id).joint_pos = np.array(ep["qpos0"][mg_id])
    for controller in robot.controllers.values():
        controller.reset()
    mujoco.mj_forward(model, data)
    task.reset()

    # ---- stage 1: re-execute the clean actions once, capture the state tape
    gripper_mg = rv.get_gripper_movegroup_ids()[0]
    tape = []
    tcp_gap = []
    exo_path = args.out / f"rerender_traj{args.traj}_replay_exo.mp4"
    writer = imageio.get_writer(exo_path, format="ffmpeg", fps=cfg.fps, quality=6)
    for t, action in enumerate(ep["actions"]):
        task.step(action)
        tape.append(np.array(data.qpos))
        writer.append_data(env.render_rgb_frame("exo_camera"))
        tcp_now = pose_mat_to_7d(rv.get_move_group(gripper_mg).leaf_frame_to_robot)
        tcp_gap.append(float(np.linalg.norm(tcp_now[:3] - ep["tcp_pose"][t][:3])))
    writer.close()
    tape = np.asarray(tape)
    gap = np.asarray(tcp_gap)

    # replayed task outcome, judged the same way datagen does (success[-1])
    replay_success = bool(task.judge_success())
    print(f"[rerender] replayed {len(tape)} steps · TCP gap mean "
          f"{gap.mean() * 1000:.1f} mm / p95 {np.percentile(gap, 95) * 1000:.1f} mm / "
          f"max {gap.max() * 1000:.1f} mm · success replay={replay_success} "
          f"recorded={ep['success']}")

    # keep the original recording next to the replay for the A/B check
    orig = args.h5.parent / f"episode_{args.traj:08d}_exo_camera_batch_1_of_1.mp4"
    orig_name = None
    if orig.exists():
        orig_name = f"rerender_traj{args.traj}_original_exo.mp4"
        shutil.copy(orig, args.out / orig_name)

    # ---- stage 2: re-render the tape N times under fresh visual draws
    def diag_frame(renderer, cam):
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    draw_files = []
    strip = []
    for k in range(args.draws):
        if k == 0:  # draw 0 = authored visuals, the reference pass
            sampler.lighting_randomizer.restore_defaults()
            restore_texture_defaults(sampler.texture_randomizer, model)
        else:
            sampler.lighting_randomizer.randomize(data)
            sampler.texture_randomizer.randomize(data)
        renderer = mujoco.Renderer(model, 352, 624)  # fresh: textures re-upload
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.32, 0.0, 0.15)
        cam.azimuth, cam.elevation, cam.distance = 160, -25, 1.0

        name = f"rerender_traj{args.traj}_draw{k}.mp4"
        w = imageio.get_writer(args.out / name, format="ffmpeg", fps=cfg.fps, quality=6)
        for t in range(len(tape)):
            data.qpos[:] = tape[t]
            mujoco.mj_forward(model, data)
            frame = diag_frame(renderer, cam)
            w.append_data(frame)
            if t == len(tape) // 3:
                strip.append((k, frame))
        w.close()
        renderer.close()
        draw_files.append(name)
        print(f"[rerender] draw {k}: {name}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(strip), figsize=(4.3 * len(strip), 2.9))
    for ax, (k, img) in zip(np.ravel(axes), strip):
        ax.imshow(img)
        ax.set_title("authored" if k == 0 else f"visual draw {k}", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Offline re-render — one simulated trajectory (traj {args.traj}), "
        f"{len(strip)} visual passes, same frame", fontsize=12)
    fig.tight_layout()
    strip_name = f"rerender_traj{args.traj}_strip.png"
    fig.savefig(args.out / strip_name, dpi=110)

    metrics = (
        f"{len(tape)} steps re-executed (noise off) · TCP gap vs recording: "
        f"mean {gap.mean() * 1000:.1f} mm / p95 {np.percentile(gap, 95) * 1000:.1f} mm / "
        f"max {gap.max() * 1000:.1f} mm · success replay={replay_success} "
        f"recorded={ep['success']} · {args.draws} render passes"
    )
    print(f"[rerender] {metrics}")

    if args.fragment:
        media = [{"type": "image", "file": strip_name}]
        if orig_name:
            media.append({"type": "video", "file": orig_name,
                          "label": "original recording (exo cam)"})
        media.append({"type": "video", "file": exo_path.name,
                      "label": "replay from clean actions (exo cam)"})
        media += [
            {"type": "video", "file": f,
             "label": "authored visuals (diag view)" if k == 0 else f"visual draw {k} (diag view)"}
            for k, f in enumerate(draw_files)
        ]
        (args.out / "manifest_offline_rerender.json").write_text(json.dumps({
            "id": "offline_rerender",
            "group": "post-hoc",
            "title": "Offline re-render — simulate once, render many",
            "desc": "A saved episode is rebuilt from its frozen config (exact "
            "cube/cup poses, init qpos), its clean recorded actions are "
            "re-executed once with action noise disabled, and the resulting "
            "state tape is re-rendered under fresh lighting/texture draws — "
            "visual DR without re-simulating. The repo's offline DR renderer "
            "class is a stub; this script is the working path. Caveat: only "
            "task-relevant objects are frozen, so extra shelf cubes from "
            "multi-cube episodes replay parked on the floor.",
            "command": f"MUJOCO_GL=egl python scripts/viz/piper_x_offline_rerender.py "
                       f"--traj {args.traj} --draws {args.draws} --fragment",
            "metrics": metrics,
            "media": media,
        }, indent=2))
        print(f"[rerender] wrote {args.out / 'manifest_offline_rerender.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
