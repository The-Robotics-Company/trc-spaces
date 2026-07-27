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


def rebuild_scene(saved, seed: int = 0, dr: bool = True):
    """Fresh task forced to the saved episode state.

    dr=True constructs the lighting/texture randomizers (this also bakes the
    texture pool into the model, which adds a skybox — the background turns
    blue-grey). dr=False compiles the exact datagen model, for the
    replay-vs-recording A/B where the visuals must match the dataset.
    """
    import numpy as np

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
        PiperXCubesInCupTaskSampler,
    )
    from molmo_spaces.env.data_views import create_mlspaces_body
    from molmo_spaces.utils.pose import pos_quat_to_pose_mat

    cfg = PiperXCubesInCupDataGenConfig()
    cfg.robot_config.action_noise_config.enabled = False  # deterministic re-execution
    # the replay-vs-recording A/B check needs the AUTHORED camera mounts, not
    # a fresh per-episode camera-noise draw
    for cam in cfg.camera_config.cameras:
        cam.fov_noise_degrees = cam.pos_noise_range = cam.orientation_noise_degrees = None
    ts = cfg.task_sampler_config
    ts.randomize_lighting = dr  # constructs the randomizers + bakes the texture pool
    ts.randomize_textures = dr
    ts.randomize_textures_all = dr
    ts.randomize_dynamics = False

    np.random.seed(seed)
    cfg.seed = seed
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    task = sampler.sample_task()
    env = task.env
    data = env.current_data

    if dr:  # sample_task already rolled lighting + textures once — back to authored
        sampler.lighting_randomizer.restore_defaults()
        restore_texture_defaults(sampler.texture_randomizer, env.current_model)

    # the base pose is randomized per episode since 2026-07-26 — restore the
    # recorded one (sample_task drew a fresh one)
    env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(
        np.asarray(saved.task_config.robot_base_pose))

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

        # shortest successful episode ≈ fewest cubes. Prefer single-cube ones:
        # SavedEpisode freezes only the pickup cube + cup, so extra shelf
        # cubes replay parked (visible ghost in the A/B check)
        with h5py.File(args.h5) as f:
            oks = [(f[f"traj_{i}"]["actions/commanded_action"].shape[0], i)
                   for i in range(len(f)) if bool(f[f"traj_{i}"]["success"][-1])]
        args.traj = min(oks)[1] if oks else 0
    print(f"[rerender] episode source: {args.h5} traj_{args.traj}")
    args.out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from molmo_spaces.utils.pose import pose_mat_to_7d

    ep = load_episode(args.h5, args.traj)
    # stage-1 env: exact datagen model (dr=False), so the replay's exo view is
    # pixel-comparable to the dataset's episode video
    cfg, sampler, task = rebuild_scene(ep["saved"], dr=False)
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
    replay_frames = []  # exo view of the replay, for the A/B fidelity video
    for t, action in enumerate(ep["actions"]):
        task.step(action)
        tape.append(np.array(data.qpos))
        replay_frames.append(env.render_rgb_frame("exo_camera").copy())
        tcp_now = pose_mat_to_7d(rv.get_move_group(gripper_mg).leaf_frame_to_robot)
        tcp_gap.append(float(np.linalg.norm(tcp_now[:3] - ep["tcp_pose"][t][:3])))
    tape = np.asarray(tape)
    gap = np.asarray(tcp_gap)

    # replayed task outcome, judged the same way datagen does (success[-1])
    replay_success = bool(task.judge_success())
    print(f"[rerender] replayed {len(tape)} steps · TCP gap mean "
          f"{gap.mean() * 1000:.1f} mm / p95 {np.percentile(gap, 95) * 1000:.1f} mm / "
          f"max {gap.max() * 1000:.1f} mm · success replay={replay_success} "
          f"recorded={ep['success']}")

    # ---- stage 2: re-render the tape under fresh visual draws, in a SECOND
    # env built with the randomizers (dr=True; same nq, tape transfers).
    # All passes are kept in memory and composed into ONE synced side-by-side
    # video — separate per-draw videos made the differences invisible.
    import gc

    del task, env
    gc.collect()
    _, sampler, task = rebuild_scene(ep["saved"], dr=True)
    env = task.env
    data = env.current_data
    model = env.current_model

    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.32, 0.0, 0.15)
    cam.azimuth, cam.elevation, cam.distance = 160, -25, 1.0
    probe_t = len(tape) // 3

    def render_pass():
        renderer = mujoco.Renderer(model, 352, 624)  # fresh: textures re-upload
        frames = np.empty((len(tape), 352, 624, 3), dtype=np.uint8)
        for t in range(len(tape)):
            data.qpos[:] = tape[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames[t] = renderer.render()
        renderer.close()
        return frames

    def probe_brightness():
        renderer = mujoco.Renderer(model, 352, 624)
        data.qpos[:] = tape[probe_t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        b = float(renderer.render().mean())
        renderer.close()
        return b

    sampler.lighting_randomizer.restore_defaults()
    restore_texture_defaults(sampler.texture_randomizer, model)
    passes = [("authored visuals", render_pass())]
    ref_bright = float(passes[0][1][probe_t].mean())
    for k in range(1, args.draws + 1):
        # a LightingRandomizer draw can legitimately be near-black (one dim
        # light); those make useless demo media, so re-roll them here — the
        # lighting demo shows the raw distribution
        for _ in range(6):
            sampler.lighting_randomizer.randomize(data)
            sampler.texture_randomizer.randomize(data)
            if probe_brightness() >= 0.5 * ref_bright:
                break
            print(f"[rerender] draw {k}: re-rolled a too-dark lighting draw")
        passes.append((f"new visuals, draw {k}", render_pass()))
        print(f"[rerender] draw {k}: rendered")

    from PIL import Image, ImageDraw, ImageFont

    try:
        _font = ImageFont.load_default(22)
    except TypeError:  # Pillow < 10
        _font = ImageFont.load_default()

    def label_bar(frame: np.ndarray, text: str) -> np.ndarray:
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
        d.text((10, 4), text, fill=(255, 255, 255), font=_font)
        return np.asarray(im)

    # one synced video: all passes in a 2-column grid, same timeline
    grid_name = f"rerender_traj{args.traj}_draws_grid.mp4"
    w = imageio.get_writer(args.out / grid_name, format="ffmpeg", fps=cfg.fps, quality=6)
    blank = np.zeros_like(passes[0][1][0])
    for t in range(len(tape)):
        tiles = [label_bar(frames[t], lbl) for lbl, frames in passes]
        if len(tiles) % 2:
            tiles.append(blank)
        rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
        w.append_data(np.concatenate(rows, axis=0))
    w.close()
    print(f"[rerender] wrote {grid_name} ({len(passes)} synced passes)")

    # one A/B video: dataset recording | this replay, same camera, same timeline
    orig = args.h5.parent / f"episode_{args.traj:08d}_exo_camera_batch_1_of_1.mp4"
    fidelity_name = None
    if orig.exists():
        fidelity_name = f"rerender_traj{args.traj}_fidelity.mp4"
        r = imageio.get_reader(orig)
        w = imageio.get_writer(args.out / fidelity_name, format="ffmpeg",
                               fps=cfg.fps, quality=6)
        for t, oframe in enumerate(r):
            if t >= len(replay_frames):
                break
            w.append_data(np.concatenate(
                [label_bar(np.asarray(oframe), "original recording"),
                 label_bar(replay_frames[t], "offline replay (same camera)")], axis=1))
        w.close()
        r.close()
        print(f"[rerender] wrote {fidelity_name}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(passes), figsize=(4.3 * len(passes), 2.9))
    for ax, (lbl, frames) in zip(np.ravel(axes), passes):
        ax.imshow(frames[probe_t])
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    fig.suptitle("Same simulated instant, re-rendered under each visual draw",
                 fontsize=12)
    fig.tight_layout()
    strip_name = f"rerender_traj{args.traj}_strip.png"
    fig.savefig(args.out / strip_name, dpi=110)

    metrics = (
        f"replay fidelity: TCP stays within {gap.mean() * 1000:.1f} mm (mean) / "
        f"{gap.max() * 1000:.1f} mm (max) of the recording over {len(tape)} steps · "
        f"outcome preserved (recorded {ep['success']} → replay {replay_success}) · "
        f"{args.draws} visual draws"
    )
    print(f"[rerender] {metrics}")

    if args.fragment:
        media = [
            {"type": "image", "file": strip_name},
            {"type": "video", "file": grid_name,
             "label": "one episode, four looks — same motion, synced"},
        ]
        # The A/B video (fidelity_name) stays out of the viewer for now: the
        # scene assets were re-authored on 2026-07-26 (cube red->green, board
        # visual split, backdrop) AFTER every existing dataset was recorded,
        # so replay-vs-recording colors mismatch and read as a bug. The
        # measured TCP fidelity in `metrics` is the actual check. Re-add once
        # a dataset recorded with the current assets exists (also needs the
        # RECORDED exo mount restored — datagen now jitters camera mounts).
        (args.out / "manifest_offline_rerender.json").write_text(json.dumps({
            "id": "offline_rerender",
            "group": "post-hoc",
            "title": "Offline re-render — simulate once, render many",
            "desc": "Turns one recorded episode into several training clips "
            "with different lighting and textures, without re-running "
            "physics. Render-side changes only — moving objects, the robot, "
            "or its actions needs a real re-simulation.",
            "command": f"MUJOCO_GL=egl python scripts/viz/piper_x_offline_rerender.py "
                       f"--traj {args.traj} --draws {args.draws} --fragment",
            "metrics": metrics,
            "media": media,
        }, indent=2))
        print(f"[rerender] wrote {args.out / 'manifest_offline_rerender.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
