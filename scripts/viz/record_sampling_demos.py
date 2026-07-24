"""Record media for the data viewer's "Sampling demos" section (headless).

Each demo shows one source of episode variation for PiPER-X cubes-in-cup
(see docs/dr_and_task_sampling_map.md):

  task_sampling         grid of exo views across seeds — cube count (1-4),
                        cube/cup placement, robot init pose noise
  action_noise          two full rollouts of the SAME seed, exo camera:
                        datagen-default action noise vs cranked-up (the
                        scripts/viz/piper_x_action_noise_demo.py knobs),
                        with the measured TCP deviation stats
  camera_randomization  grid of rendered views from the randomized exo camera
                        (PiperXCubesInCupRandomCamDataGenConfig sampling path)

Writes results/sampling_demos/{manifest.json, *.png, *.mp4}; the viewer builder
(scripts/viz/build_data_viewer.py) picks that directory up via --demos and adds
the section to the gallery index.

    conda activate mlspaces
    MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py            # all
    MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py action_noise

Each demo runs in its own subprocess (fresh GL context / config state); the
parent merges the per-demo manifest fragments.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "sampling_demos"
ALL_DEMOS = ["task_sampling", "action_noise", "camera_randomization"]

# knobs shared with scripts/viz/piper_x_action_noise_demo.py
NOISE_DEFAULT = (0.1, 0.02, 0.1)  # action_scale_factor, pos cap [m], rot cap [rad]
NOISE_CRANKED = (1.0, 0.05, 0.3)
NOISE_SEED = 21
TASK_SEEDS = list(range(9))
CAM_SEED = 7
CAM_N = 12


def _fragment(name: str) -> Path:
    return OUT / f"manifest_{name}.json"


def _write_fragment(name: str, entry: dict) -> None:
    _fragment(name).write_text(json.dumps(entry, indent=2))
    print(f"[demos] wrote {_fragment(name)}")


# --------------------------------------------------------------------------
# shared episode helpers (imported lazily inside demos: heavy molmo imports)
# --------------------------------------------------------------------------


def _make_task(exp_config, seed: int):
    import numpy as np

    np.random.seed(seed)
    exp_config.seed = seed
    sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
    last = None
    for attempt in range(10):
        try:
            return sampler, sampler.sample_task()
        except Exception as e:  # noqa: BLE001
            print(f"[attempt {attempt}] sampling failed: {e}")
            last = e
    raise RuntimeError(f"could not sample a task (seed {seed}): {last}")


def _set_noise(exp_config, knobs) -> None:
    nc = exp_config.robot_config.action_noise_config
    nc.action_scale_factor, nc.max_tcp_position_noise, nc.max_tcp_rotation_noise = knobs


def _diag_renderer(env, w: int = 624, h: int = 352):
    """(renderer, camera) for a diagnostic free-camera view of the workspace.

    The scene's own exo_camera sits on the base aimed at link3 — from there the
    matte-black boards swallow the frame — so demo media renders from a fixed
    external viewpoint where cubes, cup and arm are all legible instead.
    """
    import mujoco

    mj_data = env.mj_datas[env.current_batch_index]
    renderer = mujoco.Renderer(mj_data.model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.32, 0.0, 0.15)
    cam.azimuth, cam.elevation, cam.distance = 160, -25, 1.0
    return renderer, cam, mj_data


def _cubes_on_shelf(task) -> list[str]:
    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupTaskSampler,
    )

    om = task.env.object_managers[task.env.current_batch_index]
    names = []
    for n in PiperXCubesInCupTaskSampler.CUBE_NAMES:
        o = om.get_object_by_name(n)
        if o is not None and 0.135 < o.position[2] < 0.20:
            names.append(n)
    return names


# --------------------------------------------------------------------------
# demos
# --------------------------------------------------------------------------


def demo_task_sampling() -> None:
    """3x3 grid of freshly sampled initial layouts (exo view), one per seed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
    )

    tiles = []
    for s in TASK_SEEDS:
        cfg = PiperXCubesInCupDataGenConfig()
        _set_noise(cfg, NOISE_DEFAULT)
        _, task = _make_task(cfg, s)
        renderer, cam, mj_data = _diag_renderer(task.env)
        renderer.update_scene(mj_data, camera=cam)
        img = renderer.render()
        renderer.close()
        ncubes = len(_cubes_on_shelf(task))
        tiles.append((s, img, ncubes))
        print(f"[task_sampling] seed {s}: {ncubes} cube(s)")
        del task
        gc.collect()

    fig, axes = plt.subplots(3, 3, figsize=(12, 7.2))
    for ax, (s, img, nc) in zip(axes.flat, tiles):
        ax.imshow(img)
        ax.set_title(f"seed {s} · {nc} cube{'s' if nc != 1 else ''}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Task sampling — initial layouts (diagnostic view)", fontsize=12)
    fig.tight_layout()
    media = "task_sampling_grid.png"
    fig.savefig(OUT / media, dpi=110)

    counts = [nc for _, _, nc in tiles]
    _write_fragment(
        "task_sampling",
        {
            "id": "task_sampling",
            "group": "task sampling",
            "title": "Task sampling — cube count & placement",
            "desc": "Fresh initial layout per seed: cube count (1-4), measured "
            "cube/cup placement on the shelf, graduated robot init-pose "
            "noise. This is where most dataset variation comes from.",
            "command": "python scripts/viz/piper_x_cubes_in_cup_preview.py <seed>",
            "metrics": f"seeds {TASK_SEEDS[0]}-{TASK_SEEDS[-1]}: cube counts {counts}",
            "media": [{"type": "image", "file": media}],
        },
    )


def _rollout_video(knobs, seed: int, video_path: Path) -> str:
    """Run one policy rollout, save exo frames as mp4, return deviation stats."""
    import imageio.v2 as imageio
    import numpy as np

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
    )

    cfg = PiperXCubesInCupDataGenConfig()
    _set_noise(cfg, knobs)
    _, task = _make_task(cfg, seed)
    policy = cfg.policy_config.policy_factory(cfg, task)
    task.register_policy(policy)
    observation, _ = task.reset()

    env = task.env
    robot = env.robots[0]
    arm_mg_id = robot.get_arm_move_group_ids()[0]
    arm_mg = robot.robot_view.get_move_group(arm_mg_id)

    renderer, cam, mj_data = _diag_renderer(env)
    writer = imageio.get_writer(video_path, format="ffmpeg", fps=cfg.fps, quality=6)
    devs = []
    steps = 0
    err = None
    # under cranked noise the policy can legitimately fail mid-episode (e.g. a
    # cube gets knocked off the workspace and the retry pose has no IK) — keep
    # the footage up to that point and report the abort instead of crashing
    try:
        for _ in range(20000):
            action = policy.get_action(observation)
            if action is None or action.get("done"):
                break
            observation, _r, terminal, truncated, _i = task.step(action)
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
            steps += 1
            clean = robot.last_unnoised_cmd_joint_pos()
            if clean and arm_mg_id in clean:
                dq = np.asarray(clean[arm_mg_id]) - np.asarray(arm_mg.joint_pos)
                J = robot.robot_view.get_jacobian(arm_mg_id, [arm_mg_id])
                devs.append(float(np.linalg.norm((J @ dq)[:3])))
            if terminal or truncated:
                break
    except Exception as e:  # noqa: BLE001
        err = str(e).splitlines()[0]
        print(f"[action_noise] episode aborted at step {steps}: {err}")
    writer.close()
    renderer.close()

    d = np.asarray(devs) if devs else np.zeros(1)
    stats = (
        f"scale={knobs[0]} pos_cap={knobs[1] * 100:.0f}cm rot_cap={knobs[2]} · "
        f"{steps} steps · TCP dev mean {d.mean() * 1000:.1f} mm / "
        f"p99 {np.percentile(d, 99) * 1000:.1f} mm / max {d.max() * 1000:.1f} mm"
        + (f" · aborted: {err}" if err else "")
    )
    print(f"[action_noise] {video_path.name}: {stats}")
    del task, policy
    gc.collect()
    return stats


def demo_action_noise() -> None:
    """Same seed twice: datagen-default noise vs cranked-up (demo-script knobs)."""
    stats_def = _rollout_video(NOISE_DEFAULT, NOISE_SEED, OUT / "action_noise_default.mp4")
    stats_crk = _rollout_video(NOISE_CRANKED, NOISE_SEED, OUT / "action_noise_cranked.mp4")
    _write_fragment(
        "action_noise",
        {
            "id": "action_noise",
            "group": "always-on DR",
            "title": "Action noise — default vs cranked",
            "desc": "TCP-space noise proportional to the commanded delta, added "
            "to every control step; the dataset stores the CLEAN commands "
            "(DART-style recovery supervision). Same seed, two noise "
            "levels — deviation stays bounded (jitter, not drift).",
            "command": "NOISE_SCALE=1.0 NOISE_POS_CAP=0.05 NOISE_ROT_CAP=0.3 "
            f"python scripts/viz/piper_x_action_noise_demo.py {NOISE_SEED}",
            "metrics": f"default: {stats_def}  |  cranked: {stats_crk}",
            "media": [
                {"type": "video", "file": "action_noise_default.mp4", "label": "datagen default (scale 0.1, caps 2 cm / 0.1 rad)"},
                {"type": "video", "file": "action_noise_cranked.mp4", "label": "cranked (scale 1.0, caps 5 cm / 0.3 rad)"},
            ],
        },
    )


def demo_camera_randomization() -> None:
    """Sampled exo cameras as 3D frustums with their rendered views.

    Mirrors scripts/viz/piper_x_random_cam_rerun.py: same sampling path
    (sampler.setup_cameras -> registry), same cam-to-world poses from
    Camera.get_pose() (RDF: x=right, y=down, z=forward). Output is a JSON the
    viewer turns into an interactive three.js pane (frustum + image plane per
    sample), plus the .rrd from the rerun script itself for `rerun <file>`.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupRandomCamDataGenConfig,
    )

    cfg = PiperXCubesInCupRandomCamDataGenConfig()
    _set_noise(cfg, NOISE_DEFAULT)
    sampler, task = _make_task(cfg, CAM_SEED)
    env = task.env
    center = sampler.get_workspace_center(env)
    w, h = cfg.camera_config.img_resolution

    def jpeg_uri(img: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    cams = []
    for i in range(CAM_N):
        if i > 0:
            sampler.setup_cameras(env)
        cam = env.camera_manager.registry["exo_camera"]
        cams.append(
            {
                "pose": [round(float(x), 5) for x in cam.get_pose().flatten()],
                "fov": round(float(cam.fov), 2),
                "img": jpeg_uri(env.render_rgb_frame("exo_camera")),
            }
        )
        print(f"[camera_randomization] sample {i}: fov {cams[-1]['fov']}")

    media = "camera_samples.json"
    (OUT / media).write_text(
        json.dumps(
            {"w": w, "h": h, "center": [round(float(x), 4) for x in center], "cams": cams}
        )
    )

    pos = np.array([np.asarray(c["pose"]).reshape(4, 4)[:3, 3] for c in cams])
    fov = np.array([c["fov"] for c in cams])
    dist = np.linalg.norm(pos - np.asarray(center), axis=1)

    # the .rrd artifact comes from the rerun script itself (VIEW=0 saves it)
    links = []
    rrd_name = f"random_cams_seed{CAM_SEED}.rrd"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/viz/piper_x_random_cam_rerun.py"),
         str(CAM_SEED), str(CAM_N)],
        cwd=REPO,
        env={**os.environ, "VIEW": "0"},
    )
    rrd_src = REPO / "scripts" / "viz" / rrd_name
    if r.returncode == 0 and rrd_src.exists():
        rrd_src.rename(OUT / rrd_name)
        links.append({"label": f"3D scene recording — open with: rerun {rrd_name}",
                      "file": rrd_name})
    else:
        print(f"[camera_randomization] WARNING: rerun script failed (exit {r.returncode})")

    _write_fragment(
        "camera_randomization",
        {
            "id": "camera_randomization",
            "group": "camera sampling",
            "title": "Randomized exocentric camera",
            "desc": "Spherical sampling around the workspace center with a "
            "segmentation-based visibility check — the real "
            "RandomizedExocentricCameraConfig path. Each sampled camera is "
            "drawn as a frustum with its actual rendered view on the image "
            "plane; drag to orbit, scroll to zoom.",
            "command": f"python scripts/viz/piper_x_random_cam_rerun.py {CAM_SEED} {CAM_N}",
            "metrics": f"dist-to-center [{dist.min():.2f}, {dist.max():.2f}] m · "
            f"fov [{fov.min():.1f}, {fov.max():.1f}]° · "
            f"x [{pos[:, 0].min():+.2f}, {pos[:, 0].max():+.2f}] "
            f"y [{pos[:, 1].min():+.2f}, {pos[:, 1].max():+.2f}] "
            f"z [{pos[:, 2].min():+.2f}, {pos[:, 2].max():+.2f}] m",
            "media": [{"type": "cams3d", "file": media}],
            "links": links,
        },
    )


# Axes that exist in molmo_spaces but are OFF for PiPER-X cubes-in-cup — shown
# as a muted note card so the section reflects the full DR map, not just what
# is active (see docs/dr_and_task_sampling_map.md).
INACTIVE_NOTE = {
    "id": "inactive_axes",
    "group": "off by default",
    "title": "Inactive DR axes (PiPER-X cubes-in-cup)",
    "desc": "randomize_lighting · randomize_textures (+ BRDF jitter) · "
    "randomize_dynamics (mass/friction/inertia ±20%) · robot speckle "
    "textures (Franka-only, silently ignored for PiPER-X) · door joint DR. "
    "All default off and unset in PiperXCubesInCupDataGenConfig — episodes "
    "vary only through task sampling, camera noise and action noise.",
    "command": "docs/dr_and_task_sampling_map.md",
    "media": [],
}


def merge_manifest() -> None:
    demos = []
    for name in ALL_DEMOS:
        p = _fragment(name)
        if p.exists():
            demos.append(json.loads(p.read_text()))
    demos.append(INACTIVE_NOTE)
    (OUT / "manifest.json").write_text(json.dumps({"demos": demos}, indent=2))
    print(f"[demos] merged manifest: {OUT / 'manifest.json'} ({len(demos)} entries)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demos", nargs="*", default=ALL_DEMOS, help=f"subset of {ALL_DEMOS}")
    ap.add_argument("--inline", action="store_true", help="run in-process (no subprocess)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    targets = args.demos or ALL_DEMOS
    bad = [d for d in targets if d not in ALL_DEMOS]
    if bad:
        sys.exit(f"unknown demo(s): {bad} — choose from {ALL_DEMOS}")

    if args.inline:
        for name in targets:
            {
                "task_sampling": demo_task_sampling,
                "action_noise": demo_action_noise,
                "camera_randomization": demo_camera_randomization,
            }[name]()
    else:
        # one subprocess per demo: fresh GL context + config class state
        for name in targets:
            print(f"\n[demos] === {name} ===")
            r = subprocess.run(
                [sys.executable, __file__, name, "--inline"],
                cwd=REPO,
                env={**os.environ, "VIEW": "0"},
            )
            if r.returncode != 0:
                print(f"[demos] {name} FAILED (exit {r.returncode}) — continuing")
    merge_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
