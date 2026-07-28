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
ALL_DEMOS = [
    "task_sampling",
    "action_noise",
    "camera_randomization",
    "camera_noise",
    "lighting_randomization",
    "texture_randomization",
    "dynamics_randomization",
    "init_pose_noise",
    "base_pose_noise",
    "offline_rerender",
]

# knobs shared with scripts/viz/piper_x_action_noise_demo.py
NOISE_DEFAULT = (0.1, 0.02, 0.1)  # action_scale_factor, pos cap [m], rot cap [rad]
NOISE_CRANKED = (1.0, 0.05, 0.3)
NOISE_SEED = 21
# 3 seeds are enough to show the spread (measured cube counts: 1 / 3 / 4)
TASK_SEEDS = [0, 3, 5]
CAM_SEED = 7
CAM_N = 12
# opt-in DR demos (lighting / texture / dynamics): seed 5 = 4 cubes + cup —
# the busiest layout, so the visual change has the most surfaces to show on
DR_SEED = 5
DR_TILES = 8  # authored baseline + 7 re-rolls
DYN_ROUNDS = 200


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
    """(renderer, camera, mj_data) rendering from the scene's ACTUAL exo_camera.

    Previously this used a fixed free-camera viewpoint. Demos now render from the
    real world-fixed exo_camera (the diag view baked into the scene, carrying its
    per-episode mount/FOV noise) so demo media matches what the dataset records.
    """
    import mujoco

    mj_data = env.mj_datas[env.current_batch_index]
    renderer = mujoco.Renderer(mj_data.model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(mj_data.model, mujoco.mjtObj.mjOBJ_CAMERA, "exo_camera")
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

    fig, axes = plt.subplots(1, len(tiles), figsize=(4.3 * len(tiles), 2.9))
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
            "cube/cup placement on the shelf. This is where most dataset "
            "variation comes from. (No robot init-pose noise: inert for "
            "PiPER-X — see the init_pose_noise demo.)",
            "command": "python scripts/viz/piper_x_cubes_in_cup_preview.py <seed>",
            "metrics": f"seeds {TASK_SEEDS}: cube counts {counts}",
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


def demo_camera_noise() -> None:
    """Per-episode MJCF camera noise (stock config, always on): the wrist cam
    gets ±4° FOV + ±1.5/0.5/1 cm mount + (8,4,4)° orientation jitter, the fixed
    exo ±3° FOV + ±1 cm + ±4° — the upstream DROID-wrist / RBY1-head recipes.
    Complements camera_randomization: that demo re-SAMPLES the exo pose from
    scratch (randcam subversion); this one jitters the authored mounts, and it
    applies to BOTH configs' wrist cam. Layout fixed; cameras re-rolled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    N_ROLLS = 3  # + authored column
    N_SPREAD = 200

    sampler, task = _make_dr_task(DR_SEED)
    env = task.env
    cam_cfgs = task.config.camera_config.cameras  # [wrist, exo]
    names = [c.name for c in cam_cfgs]

    def snapshot():
        out = {}
        for n in names:
            cam = env.camera_manager.registry[n]
            out[n] = (cam.get_pose().copy(), float(cam.fov))
        return out

    def deltas(ref, cur, n):
        (T0, f0), (T, f) = ref[n], cur[n]
        dp = np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 100
        c = (np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2
        da = np.degrees(np.arccos(np.clip(c, -1, 1)))
        return dp, da, f - f0

    # authored: disable the noise fields, re-setup, snapshot the clean mounts
    saved = [(c.fov_noise_degrees, c.pos_noise_range, c.orientation_noise_degrees)
             for c in cam_cfgs]
    for c in cam_cfgs:
        c.fov_noise_degrees = c.pos_noise_range = c.orientation_noise_degrees = None
    sampler.setup_cameras(env)
    ref = snapshot()
    cols = [("authored mounts", {n: env.render_rgb_frame(n).copy() for n in names})]
    for c, (fn, pn, on) in zip(cam_cfgs, saved):
        c.fov_noise_degrees, c.pos_noise_range, c.orientation_noise_degrees = fn, pn, on

    for k in range(1, N_ROLLS + 1):
        sampler.setup_cameras(env)
        cur = snapshot()
        parts = []
        for n in names:
            dp, da, df = deltas(ref, cur, n)
            parts.append(f"{n.split('_')[0]}: {dp:.1f}cm {da:.1f}° fov{df:+.1f}°")
        label = f"re-roll {k} · " + " · ".join(parts)
        cols.append((label, {n: env.render_rgb_frame(n).copy() for n in names}))
        print(f"[camera_noise] {label}")

    fig, axes = plt.subplots(len(names), len(cols),
                             figsize=(4.3 * len(cols), 2.6 * len(names)))
    for j, (label, imgs) in enumerate(cols):
        for i, n in enumerate(names):
            ax = axes[i][j]
            ax.imshow(imgs[n])
            ax.axis("off")
            if i == 0:
                ax.set_title(label, fontsize=7)
            if j == 0:
                ax.text(-0.04, 0.5, n, transform=ax.transAxes, fontsize=9,
                        rotation=90, va="center", ha="right")
    fig.suptitle("Camera mount & FOV noise — authored vs per-episode re-rolls "
                 "(layout fixed)", fontsize=12)
    fig.tight_layout()
    media = "camera_noise_grid.png"
    fig.savefig(OUT / media, dpi=110)

    # measured spread (no rendering)
    spread = {n: [] for n in names}
    for _ in range(N_SPREAD):
        sampler.setup_cameras(env)
        cur = snapshot()
        for n in names:
            spread[n].append(deltas(ref, cur, n))
    stats = []
    for n in names:
        a = np.asarray(spread[n])
        stats.append(f"{n}: pos ≤{a[:, 0].max():.1f} cm · rot ≤{a[:, 1].max():.1f}° "
                     f"· fov {a[:, 2].min():+.1f}..{a[:, 2].max():+.1f}°")
    metrics = f"over {N_SPREAD} rolls — " + "  |  ".join(stats)
    print(f"[camera_noise] {metrics}")

    _write_fragment(
        "camera_noise",
        {
            "id": "camera_noise",
            "group": "camera sampling",
            "title": "Camera mount & FOV noise — wrist + exo",
            "desc": "Per-episode jitter of the AUTHORED camera mounts (stock "
            "config, both cameras; upstream DROID-wrist / RBY1-head recipes): "
            "wrist ±4° FOV, ±1.5/0.5/1 cm, (8,4,4)° orientation; exo ±3° FOV, "
            "±1 cm, ±4°. Complements the randomized exocentric camera above — "
            "that path re-samples the exo pose from scratch (randcam config), "
            "and its wrist cam gets this same jitter.",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py camera_noise",
            "metrics": metrics,
            "media": [{"type": "image", "file": media}],
        },
    )
    del task
    gc.collect()


def _make_dr_task(seed: int, *, lighting=False, textures=False, dynamics=False):
    """Task with the opt-in DR flags set EXPLICITLY (config class attrs are
    shared in-process, so every flag is written every time, True or False).
    Same seed => same episode layout: the flags only add independent
    RandomState streams (task_sampler.init_scene), they don't touch the
    global np.random stream the layout sampling draws from."""
    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
    )

    cfg = PiperXCubesInCupDataGenConfig()
    _set_noise(cfg, NOISE_DEFAULT)
    ts = cfg.task_sampler_config
    ts.randomize_lighting = lighting
    ts.randomize_textures = textures  # also bakes the empty-material pool at compile
    ts.randomize_textures_all = textures  # full randomize(): the custom scene has no THOR categories
    ts.randomize_dynamics = dynamics
    return _make_task(cfg, seed)


def _tile_grid(tiles, suptitle: str, path: Path) -> None:
    """Save a 4-wide grid of (label, img) tiles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 2.9 * rows))
    axes = np.ravel(axes)
    for ax, (label, img) in zip(axes, tiles):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def demo_lighting_randomization() -> None:
    """One fixed episode layout (seed 5), lighting re-rolled per tile — exactly
    the per-episode LightingRandomizer.randomize() path from task_sampler
    randomize_scene(), with measured per-roll light state in the tile labels."""
    import mujoco
    import numpy as np

    sampler, task = _make_dr_task(DR_SEED, lighting=True)
    rnd = sampler.lighting_randomizer
    renderer, cam, mj_data = _diag_renderer(task.env)
    model = rnd.model

    def roll_stats():
        on = [i for i in rnd.light_ids if model.light_active[i] > 0]
        diffuse = float(np.mean([model.light_diffuse[i].mean() for i in on])) if on else 0.0
        tilt = 0.0
        for i in rnd.light_ids:
            d0 = rnd._defaults[i]["dir"] / np.linalg.norm(rnd._defaults[i]["dir"])
            d1 = model.light_dir[i] / (np.linalg.norm(model.light_dir[i]) + 1e-9)
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip(d0 @ d1, -1.0, 1.0)))))
        return len(on), diffuse, tilt

    tiles = []
    stats = []
    for k in range(DR_TILES):
        if k == 0:
            rnd.restore_defaults()  # sample_task already rolled once — show authored first
            mujoco.mj_forward(model, mj_data)
        else:
            rnd.randomize(mj_data)
        renderer.update_scene(mj_data, camera=cam)
        img = renderer.render()
        n_on, diffuse, tilt = roll_stats()
        label = (
            f"authored · {n_on}/{len(rnd.light_ids)} on · diffuse {diffuse:.2f}"
            if k == 0
            else f"roll {k} · {n_on}/{len(rnd.light_ids)} on · diffuse {diffuse:.2f} · tilt {tilt:.0f}\N{DEGREE SIGN}"
        )
        tiles.append((label, img))
        if k > 0:
            stats.append((n_on, diffuse, tilt))
        print(f"[lighting_randomization] {label}")
    renderer.close()

    media = "lighting_randomization_grid.png"
    _tile_grid(tiles, "Lighting randomization — same layout, lighting re-rolled per episode", OUT / media)

    n_on = [s[0] for s in stats]
    diffuse = [s[1] for s in stats]
    tilt = [s[2] for s in stats]
    _write_fragment(
        "lighting_randomization",
        {
            "id": "lighting_randomization",
            "group": "opt-in DR",
            "title": "Lighting randomization (randomize_lighting)",
            "desc": "Per-episode perturbation of every light around its authored "
            "value: position ±0.5 m, direction ±1 rad, "
            "ambient/diffuse/specular ±0.1, random on/off (at least one "
            "light is forced back on). Same seed-5 layout in every tile; only "
            "the lighting is re-rolled. Off by default for PiPER-X "
            "cubes-in-cup — flipped on here.",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py lighting_randomization",
            "metrics": f"{len(rnd.light_ids)} lights · {len(stats)} rolls: "
            f"{min(n_on)}-{max(n_on)} lights on · "
            f"diffuse mean {min(diffuse):.2f}-{max(diffuse):.2f} · "
            f"max direction tilt {max(tilt):.0f}\N{DEGREE SIGN} (cap 57\N{DEGREE SIGN})",
            "media": [{"type": "image", "file": media}],
        },
    )


def demo_texture_randomization() -> None:
    """Authored baseline + stock TextureRandomizer rolls on the seed-5 layout.

    The boards were split into a collision geom + a visual geom (board1_visual,
    board2_visual), so the STOCK randomizer — which only admits named contype==0
    (visual-only) geoms — now reaches the table and shelf, not just the cup.
    Every roll jitters, around authored values, each eligible material's rgba
    (+/-0.15), specular / shininess / reflectance (reflectance is a newly-added
    axis), and swaps a random DTD photo (dr_texture_paths) onto any geom that
    already carries a texture — in this scene floor, backdrop wall, cup body.

    A fresh mujoco.Renderer per tile is required — texture bitmaps upload to the
    GL context at renderer creation only."""
    import gc as _gc

    import numpy as np

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        CUBE_COLOR_PALETTE,
    )
    from molmo_spaces.env.arena.randomization.texture import TextureRandomizer

    _, task0 = _make_dr_task(DR_SEED)  # all flags off -> authored look
    renderer, cam, mj_data = _diag_renderer(task0.env)
    renderer.update_scene(mj_data, camera=cam)
    tiles = [("authored", renderer.render())]
    renderer.close()
    del task0
    _gc.collect()

    sampler, task = _make_dr_task(DR_SEED, textures=True)  # bakes empty-material pool
    env = task.env
    mj_data = env.mj_datas[env.current_batch_index]
    model = mj_data.model

    # fresh STOCK randomizer, geom_names=None -> auto all visual-only geoms
    # (boards, floor, backdrop, cubes, cup). Perturbations enlarged for legibility
    # in the montage; the mechanism is the stock code path. The cube palette forces
    # cubes to green|yellow (categorical) and the DTD photo subset is the
    # texture-swap pool (floor, backdrop, cup body), matching the datagen config.
    dtd = task.config.task_sampler_config.dr_texture_paths
    rnd = TextureRandomizer(
        model=model,
        random_state=np.random.RandomState(DR_SEED + 1),
        geom_names=None,
        scene_metadata=env.current_scene_metadata,
        rgba_perturbation_size=0.15,
        specular_perturbation_size=0.4,
        shininess_perturbation_size=0.4,
        reflectance_perturbation_size=0.4,
        categorical_geom_rgba=CUBE_COLOR_PALETTE,
        texture_paths=dtd,
    )
    eligible = list(rnd.geom_names)
    print(f"[texture_randomization] eligible visual geoms: {eligible}")

    def roll_stats():
        # a geom counts as "re-colored" if EITHER its geom_rgba (no-material geoms)
        # or its material rgba (material geoms, incl. categorical cubes) moved.
        # "photo-swap" = the geom sits on a __TEXTURE_RANDOMIZER_MAT_ pool
        # material (comparing against saved matid misses it: sample_task's own
        # roll already moved floor/backdrop onto pool materials before this
        # randomizer saved defaults).
        import mujoco

        rgba_changed = photo_swapped = 0
        refl_vals = []
        for gid, d in rnd._geom_id_to_defaults.items():
            mid = d.get("mat_id", -1)
            geom_moved = not np.allclose(model.geom_rgba[gid], d["geom_rgba"], atol=1e-6)
            mat_moved = (
                mid >= 0
                and d.get("mat_rgba") is not None
                and not np.allclose(model.mat_rgba[mid], d["mat_rgba"], atol=1e-6)
            )
            if geom_moved or mat_moved:
                rgba_changed += 1
            cur_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[gid]))
            if cur_name and cur_name.startswith("__TEXTURE_RANDOMIZER_MAT_"):
                photo_swapped += 1
            if mid >= 0:
                refl_vals.append(float(model.mat_reflectance[mid]))
        return rgba_changed, photo_swapped, refl_vals

    def snap():
        renderer, cam, _ = _diag_renderer(env)  # fresh renderer: re-upload textures
        renderer.update_scene(mj_data, camera=cam)
        img = renderer.render()
        renderer.close()
        return img

    # successive randomize() calls are independent: each perturbs from the
    # authored defaults saved at init, so no reset between rolls is needed.
    stats = []
    for k in range(1, DR_TILES):
        rnd.randomize(mj_data)
        rgba_changed, mat_swapped, refl_vals = roll_stats()
        stats.append((rgba_changed, mat_swapped))
        rlo, rhi = (min(refl_vals), max(refl_vals)) if refl_vals else (0.0, 0.0)
        tiles.append(
            (f"roll {k} · {rgba_changed} re-colored · {mat_swapped} photo-swap · refl {rlo:.2f}-{rhi:.2f}", snap())
        )
        print(f"[texture_randomization] {tiles[-1][0]}")

    media = "texture_randomization_grid.png"
    _tile_grid(
        tiles,
        "Texture & material randomization — DTD photos on floor, backdrop + cup body, "
        "color/gloss rolls on boards, cubes (green|yellow) + cup",
        OUT / media,
    )

    rgba = [s[0] for s in stats]
    total_swaps = sum(s[1] for s in stats)
    n_dtd = len(dtd or [])
    _write_fragment(
        "texture_randomization",
        {
            "id": "texture_randomization",
            "group": "opt-in DR",
            "title": "Texture & material randomization (randomize_textures_all)",
            "desc": "One roll does all of this at once. Surfaces with a texture "
            f"— floor, backdrop wall and cup body — get a random photo from a "
            f"{n_dtd}-image DTD subset (real messy surfaces, full hue range "
            "incl. warm; dr_texture_paths). Flat-colored surfaces (boards, cup) "
            "get rgba ±0.15 plus specular/shininess/reflectance jitter around "
            "authored values. Cubes are categorical: each independently green "
            "OR yellow per episode, gloss still jittering. Same seed-5 layout "
            "in every tile. Off by default for PiPER-X cubes-in-cup.",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py texture_randomization",
            "metrics": f"{len(eligible)} eligible visual geoms · {len(stats)} rolls: "
            f"{min(rgba)}-{max(rgba)} materials re-colored per roll, "
            f"{total_swaps} photo swaps (pool = {n_dtd} DTD images) · "
            "cubes categorical green|yellow",
            "media": [{"type": "image", "file": media}],
        },
    )


def demo_dynamics_randomization() -> None:
    """What DynamicsRandomizer actually does to the NAMED bodies, in real units:
    authored value (diamond) vs the values sampled across DYN_ROUNDS episode
    re-rolls (bar + dots). Every body with a joint is re-rolled — cubes, cup,
    all robot links (no exclusion), plus the planner's invisible grasp-helper
    bodies. Friction homogenization is the headline: every geom of an object
    gets the OBJECT-AVERAGE authored sliding friction ±20%, which strips the
    gripper pads' extra grip (authored 2.0 vs 1.0 elsewhere) every episode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from molmo_spaces.env.arena.arena_utils import (
        get_all_bodies_with_joints_as_mlspaces_objects,
    )

    sampler, task = _make_dr_task(DR_SEED, dynamics=True)
    rnd = sampler.dynamics_randomizer
    env = task.env
    mj_data = env.mj_datas[env.current_batch_index]
    model = env.mj_model

    objects = get_all_bodies_with_joints_as_mlspaces_objects(model, mj_data)
    n_helpers = sum(o.name.startswith("grasp") for o in objects)
    robot_links = [o.name for o in objects if o.name.startswith("robot_")]
    print(f"[dynamics_randomization] {len(objects)} bodies with joints: "
          f"{len(robot_links)} robot links, {n_helpers} grasp helpers, "
          f"{len(objects) - len(robot_links) - n_helpers} task objects")

    # authored values, saved into rnd._defaults before the very first roll
    auth_mass: dict[int, float] = {}
    auth_fric: dict[int, float] = {}
    for obj in objects:
        d = rnd._defaults[obj.object_id]
        auth_mass.update({b: m for b, m in d["body_masses"].items() if m > 0})
        auth_fric.update({g: float(f[0]) for g, f in d["geom_frictions"].items() if f[0] > 0})

    def _bid(name):
        return model.body(name).id

    mass_rows = [  # (label, body id) — smallest to heaviest, real grams
        ("robot link6", _bid("robot_0/link6")),
        ("cube", _bid("cube")),
        ("gripper finger", _bid("robot_0/gripper_link1")),
        ("cup", _bid("cup")),
        ("robot link2 (heaviest)", _bid("robot_0/link2")),
    ]
    def _gbody(g):
        return model.body(model.geom(g).bodyid.item()).name

    cube_gid = next(g for g in auth_fric if model.geom(g).bodyid.item() == _bid("cube"))
    cup_gid = next(g for g in auth_fric if model.geom(g).bodyid.item() == _bid("cup"))
    pad_gids = [g for g in auth_fric
                if _gbody(g).startswith("robot_0/gripper_link") and model.geom_contype[g] != 0]
    link_gid = next(g for g in auth_fric
                    if _gbody(g).startswith("robot_")
                    and not _gbody(g).startswith("robot_0/gripper_link"))
    fric_rows = [  # (label, geom id)
        ("cube", cube_gid),
        ("cup", cup_gid),
        ("robot link", link_gid),
        ("gripper fingertip pad", pad_gids[0]),
    ]

    mass_s = {label: [] for label, _ in mass_rows}
    fric_s = {label: [] for label, _ in fric_rows}
    mass_r, fric_r, inert_r = [], [], []
    for _ in range(DYN_ROUNDS):
        rnd.randomize_objects(objects)
        for label, b in mass_rows:
            mass_s[label].append(float(model.body_mass[b]) * 1000.0)  # grams
        for label, g in fric_rows:
            fric_s[label].append(float(model.geom_friction[g][0]))
        for obj in objects:  # pooled ratios for the metrics line
            d = rnd._defaults[obj.object_id]
            for b, m0 in d["body_masses"].items():
                if m0 > 0:
                    mass_r.append(float(model.body_mass[b]) / m0)
            for g, f0 in d["geom_frictions"].items():
                if f0[0] > 0:
                    fric_r.append(float(model.geom_friction[g][0]) / float(f0[0]))
            for b, i0 in d["body_inertias"].items():
                if np.all(np.asarray(i0) > 0):
                    inert_r.append(float(model.body_inertia[b][0]) / float(i0[0]))
    mass_r, fric_r, inert_r = map(np.asarray, (mass_r, fric_r, inert_r))

    # chart: per-body ranges in real units — authored diamond vs sampled bar
    ink, ink2, hue, surface, grid = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb", "#e4e3df"

    def _fmt(v):
        return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")

    def _range_panel(ax, rows, samples, authored, unit):
        for y, (label, key) in enumerate(rows):
            s = np.asarray(samples[label])
            ax.scatter(s, np.full_like(s, y), s=8, color=hue, alpha=0.12, linewidths=0)
            ax.plot([s.min(), s.max()], [y, y], color=hue, lw=5,
                    solid_capstyle="round", zorder=3)
            ax.plot([authored[key]], [y], marker="D", ms=7, color=ink, zorder=4)
            ax.annotate(f"[{_fmt(s.min())}, {_fmt(s.max())}]", (s.max(), y),
                        xytext=(8, -3), textcoords="offset points",
                        fontsize=8, color=ink2)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=ink)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(unit, fontsize=9, color=ink2)
        ax.set_facecolor(surface)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(ink2)

    fig, (axm, axf) = plt.subplots(1, 2, figsize=(12.9, 4.1), facecolor=surface)
    _range_panel(axm, mass_rows, mass_s, {b: auth_mass[b] * 1000.0 for _, b in mass_rows},
                 "mass [g] — one ±20% factor per body, log scale")
    axm.set_xscale("log")
    axm.set_title("mass", fontsize=11, color=ink)
    _range_panel(axf, fric_rows, fric_s, {g: auth_fric[g] for _, g in fric_rows},
                 "sliding friction μ — object-average authored value ±20%")
    axf.set_xlim(0, 2.4)
    axf.set_title("sliding friction", fontsize=11, color=ink)
    axf.set_ylim(-1.45, len(fric_rows) - 0.4)  # room for the pad note below the rows
    axf.text(
        0.0, -1.3,
        "every episode re-rolls ALL robot geoms to the robot-wide average ±20%.\n"
        "The fingertip pads used to be authored extra-grippy (μ=2.0) and lost\n"
        "~half their grip here — re-authored to μ=1.0 on 2026-07-26 (grip test: no slip difference)",
        fontsize=8.5, color=ink2, va="bottom",
    )
    handles = [
        plt.Line2D([], [], color=hue, lw=5, solid_capstyle="round",
                   label=f"sampled over {DYN_ROUNDS} episodes (min–max + dots)"),
        plt.Line2D([], [], color=ink, marker="D", ms=7, lw=0, label="authored value"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2, frameon=False,
               fontsize=9, labelcolor=ink2)
    fig.suptitle("Dynamics randomization — what each episode actually re-rolls",
                 fontsize=12, color=ink)
    fig.text(0.5, 0.895,
             f"inertia: same ±20%, per body and axis (measured [{inert_r.min():.3f}, "
             f"{inert_r.max():.3f}]) · also re-rolled: {n_helpers} invisible "
             "grasp-planner helper bodies · robot links included, no exclusion",
             ha="center", fontsize=9, color=ink2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    media = "dynamics_randomization_ranges.png"
    fig.savefig(OUT / media, dpi=110, facecolor=surface)

    pads = np.asarray(fric_s["gripper fingertip pad"])
    _write_fragment(
        "dynamics_randomization",
        {
            "id": "dynamics_randomization",
            "group": "opt-in DR",
            "title": "Dynamics randomization (randomize_dynamics)",
            "desc": "Each episode re-rolls physics before the robot moves, for "
            "every body with a joint — cubes, cup, all 8 robot links, and "
            f"{n_helpers} invisible grasp-planner helper bodies. Mass: one "
            "±20% factor per body. Inertia: ±20% per body per axis. Friction: "
            "every geom of an object is set to the object's AVERAGE authored "
            "sliding friction ±20% — this used to quietly strip the fingertip "
            "pads of their extra grip (authored μ=2.0 vs 1.0 elsewhere), so "
            "the pads were re-authored to μ=1.0 on 2026-07-26 after a grip "
            "test showed no slip difference (2/2 cubes in cup either way). The "
            "chart shows authored value vs the range actually sampled. Off by "
            "default for PiPER-X cubes-in-cup — flipped on here.",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py dynamics_randomization",
            "metrics": f"{DYN_ROUNDS} rolls · 4 cubes + cup + 8 robot links + "
            f"{n_helpers} helpers · pooled ratio ranges: mass "
            f"[{mass_r.min():.3f}, {mass_r.max():.3f}] · inertia "
            f"[{inert_r.min():.3f}, {inert_r.max():.3f}] · friction "
            f"[{fric_r.min():.3f}, {fric_r.max():.3f}] (gripper pads: "
            f"authored {auth_fric[pad_gids[0]]:.1f} → sampled "
            f"[{pads.min():.2f}, {pads.max():.2f}])",
            "media": [{"type": "image", "file": media}],
        },
    )


def demo_init_pose_noise() -> None:
    """Robot init-pose noise: the map doc lists it [ON], but PiPER-X ships with
    init_qpos_noise_range=None so it never fires here (MEASURED: every recorded
    episode starts at exactly the home qpos). This demo derives a graduated
    6-joint range with the documented Franka recipe — dq = w*0.1/||J_p @ w||,
    w = [1..6], so distal joints move more and the TCP stays within ~10 cm —
    then shows draws around home plus the measured TCP spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np

    from molmo_spaces.utils.pose import pose_mat_to_7d

    cfg_sampler, task = _make_dr_task(DR_SEED)
    env = task.env
    robot = env.robots[0]
    rv = robot.robot_view
    arm_mg_id = robot.get_arm_move_group_ids()[0]
    arm_mg = rv.get_move_group(arm_mg_id)
    grip_mg = rv.get_gripper_movegroup_ids()[0]
    model, data = env.current_model, env.current_data

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
    )

    stock_range = PiperXCubesInCupDataGenConfig().robot_config.init_qpos_noise_range
    home = np.array(task.config.robot_config.init_qpos["arm"])

    def tcp_pos() -> np.ndarray:
        return pose_mat_to_7d(rv.get_move_group(grip_mg).leaf_frame_to_robot)[:3]

    # derive the graduated range at the home pose (same recipe as the Franka
    # comment in robot_configs.py, budget 10 cm TCP displacement)
    arm_mg.joint_pos = home
    mujoco.mj_forward(model, data)
    tcp_home = tcp_pos()
    w = np.arange(1, len(home) + 1, dtype=float)
    J_p = np.asarray(rv.get_jacobian(arm_mg_id, [arm_mg_id]))[:3]
    dq = w * 0.1 / np.linalg.norm(J_p @ w)

    rng = np.random.RandomState(DR_SEED)
    renderer, cam, mj_data = _diag_renderer(env)
    tiles = []
    for k in range(DR_TILES):
        q = home if k == 0 else home + rng.uniform(-dq, dq)
        arm_mg.joint_pos = q
        mujoco.mj_forward(model, mj_data)
        disp = np.linalg.norm(tcp_pos() - tcp_home) * 100
        renderer.update_scene(mj_data, camera=cam)
        label = "home (as every recorded episode starts)" if k == 0 else \
            f"draw {k} · TCP moved {disp:.1f} cm"
        tiles.append((label, renderer.render()))
        print(f"[init_pose_noise] {label}")
    renderer.close()

    # measured TCP spread over many draws
    N_DRAWS = 500
    disps = []
    for _ in range(N_DRAWS):
        arm_mg.joint_pos = home + rng.uniform(-dq, dq)
        mujoco.mj_forward(model, data)
        disps.append(np.linalg.norm(tcp_pos() - tcp_home) * 100)
    disps = np.asarray(disps)
    arm_mg.joint_pos = home
    mujoco.mj_forward(model, data)

    grid_media = "init_pose_noise_grid.png"
    _tile_grid(
        tiles,
        "Robot init-pose noise — inert for PiPER-X (range derived here with the Franka recipe)",
        OUT / grid_media,
    )

    ink, ink2, hue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor=surface)
    ax.set_facecolor(surface)
    ax.hist(disps, bins=40, color=hue, edgecolor=surface, linewidth=0.4)
    ax.axvline(10, color=ink2, linestyle="--", linewidth=1)
    ax.text(10, 1.02, "10 cm design budget", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8, color=ink2)
    ax.set_xlabel("TCP displacement from home [cm]", fontsize=9, color=ink2)
    ax.set_ylabel("draws", fontsize=9, color=ink2)
    ax.set_title(f"TCP displacement over {N_DRAWS} draws — "
                 f"p95 {np.percentile(disps, 95):.1f} cm, max {disps.max():.1f} cm",
                 fontsize=10, color=ink)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    fig.tight_layout()
    hist_media = "init_pose_noise_spread.png"
    fig.savefig(OUT / hist_media, dpi=110, facecolor=surface)

    _write_fragment(
        "init_pose_noise",
        {
            "id": "init_pose_noise",
            "group": "always-on DR",
            "title": "Robot init-pose noise — inert for PiPER-X",
            "desc": "The sampler perturbs the robot's start pose per episode "
            "(graduated per joint so the TCP stays within ~10 cm), but only "
            "if the robot config defines init_qpos_noise_range — and "
            f"PiperXRobotConfig ships None (measured: stock={stock_range!r}), "
            "so every PiPER-X episode starts at exactly the home pose. Tiles "
            "show what the mechanism would add, using a 6-joint range derived "
            "with the documented Franka recipe (dq = w·0.1/‖J_p·w‖).",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py init_pose_noise",
            "metrics": f"stock range: None (inert) · derived range "
            f"[{', '.join(f'{x:.3f}' for x in dq)}] rad · TCP displacement "
            f"over {N_DRAWS} draws: mean {disps.mean():.1f} / p95 "
            f"{np.percentile(disps, 95):.1f} / max {disps.max():.1f} cm "
            f"(budget 10 cm)",
            "media": [
                {"type": "image", "file": grid_media},
                {"type": "image", "file": hist_media},
            ],
        },
    )


def demo_base_pose_noise() -> None:
    """Robot base-pose randomization (task sampler, always on): the mocap base
    is teleported per episode — uniform ±3 cm on each of x/y/z, plus ±15° yaw
    applied with probability BASE_YAW_PROB (else the base stays square). The
    tile grid holds the DR_SEED layout fixed and re-rolls only the base; the
    rollout runs the full policy on a fresh episode whose base the sampler
    noised itself, proving IK / cuRobo plans / obstacle poses all follow the
    moved base (world targets are re-expressed in the live base frame)."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (
        PiperXCubesInCupDataGenConfig,
        PiperXCubesInCupTaskSampler,
    )
    from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
    from molmo_spaces.utils.pose import pose_mat_to_7d

    pos_lim = PiperXCubesInCupTaskSampler.BASE_POS_NOISE
    yaw_lim = PiperXCubesInCupTaskSampler.BASE_YAW_NOISE
    yaw_p = PiperXCubesInCupTaskSampler.BASE_YAW_PROB

    def _label(T: np.ndarray) -> str:
        d = T[:3, 3] * 100
        yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        return f"dx {d[0]:+.1f} dy {d[1]:+.1f} dz {d[2]:+.1f} cm · yaw {yaw:+.1f}°"

    # --- tile grid: layout fixed, base re-rolled ------------------------------
    _, task = _make_dr_task(DR_SEED)
    env = task.env
    rv = env.robots[0].robot_view
    sampler_draw = rv.base.pose.copy()  # the sampler's own draw for this episode

    # NOT DR_SEED: the sampler's base draw is the first pull from the global
    # np.random stream seeded with DR_SEED, so the same seed here would make
    # re-roll 1 duplicate the sampler's draw (tile 1)
    rng = np.random.RandomState(DR_SEED + 1)
    renderer, cam, mj_data = _diag_renderer(env)
    tiles = []
    for k in range(DR_TILES):
        if k == 0:
            T, label = np.eye(4), "nominal mount (world origin)"
        elif k == 1:
            T, label = sampler_draw, f"sampler's draw · {_label(sampler_draw)}"
        else:
            T = np.eye(4)
            T[:3, 3] = rng.uniform(-pos_lim, pos_lim, 3)
            if rng.random_sample() < yaw_p:  # same gate as the sampler
                T[:3, :3] = R.from_euler("z", rng.uniform(-yaw_lim, yaw_lim)).as_matrix()
            label = f"re-roll {k - 1} · {_label(T)}"
        rv.base.pose = T
        mujoco.mj_forward(env.current_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        tiles.append((label, renderer.render()))
        print(f"[base_pose_noise] {label}")
    renderer.close()
    grid_media = "base_pose_noise_grid.png"
    _tile_grid(
        tiles,
        f"Robot base-pose noise — ±{pos_lim * 100:.0f} cm xyz, "
        f"±{np.degrees(yaw_lim):.0f}° yaw with p={yaw_p} "
        "(layout fixed, base re-rolled)",
        OUT / grid_media,
    )
    del task
    gc.collect()

    # --- full-episode proof on the sampler's own draw -------------------------
    cfg = PiperXCubesInCupDataGenConfig()
    _set_noise(cfg, NOISE_DEFAULT)
    _, task = _make_task(cfg, NOISE_SEED)
    policy = cfg.policy_config.policy_factory(cfg, task)
    task.register_policy(policy)
    observation, _ = task.reset()

    env = task.env
    rv = env.robots[0].robot_view
    base_T = rv.base.pose.copy()
    recorded = np.asarray(task.config.task_config.robot_base_pose)
    base_kept = bool(np.allclose(pose_mat_to_7d(base_T), recorded, atol=1e-6))
    active = _cubes_on_shelf(task)
    print(f"[base_pose_noise] rollout base: {_label(base_T)} "
          f"(survived task.reset: {base_kept}) · {len(active)} cube(s)")

    renderer, cam, mj_data = _diag_renderer(env)
    video_media = "base_pose_noise_rollout.mp4"
    writer = imageio.get_writer(OUT / video_media, format="ffmpeg",
                                fps=cfg.fps, quality=6)
    steps, err = 0, None
    try:
        for _ in range(20000):
            action = policy.get_action(observation)
            if action is None or action.get("done"):
                break
            observation, _r, terminal, truncated, _i = task.step(action)
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
            steps += 1
            if terminal or truncated:
                break
    except Exception as e:  # noqa: BLE001
        err = str(e).splitlines()[0]
        print(f"[base_pose_noise] episode aborted at step {steps}: {err}")
    writer.close()
    renderer.close()

    # outcome: active cubes whose center ended inside the cup cylinder
    data = env.current_data
    om = env.object_managers[env.current_batch_index]
    cup = om.get_object_by_name("cup")
    center, size = body_aabb(data.model, data, cup.object_id)
    rim_z, r_in = center[2] + size[2] / 2, min(size[0], size[1]) / 2
    in_cup = 0
    for name in active:
        p = om.get_object_by_name(name).position
        if np.linalg.norm(p[:2] - center[:2]) < r_in and p[2] < rim_z:
            in_cup += 1
    outcome = f"{in_cup}/{len(active)} cubes in cup, {steps} steps" + \
        (f" · aborted: {err}" if err else "")
    print(f"[base_pose_noise] rollout outcome: {outcome}")

    _write_fragment(
        "base_pose_noise",
        {
            "id": "base_pose_noise",
            "group": "task sampling",
            "title": "Robot base-pose noise — ±3 cm xyz, ±15° yaw (p=0.5)",
            "desc": "The sampler teleports the arm's mocap base every episode: "
            "uniform ±3 cm on each translation axis, plus ±15° yaw applied "
            "with probability 0.5 (else the base stays square). Planning is "
            "unaffected because every IK/motion-gen call re-expresses world "
            "targets (and cuRobo obstacles) in the live base frame; the exo "
            "camera is scene-fixed so the diag view does not move with the "
            "base. Grid: same layout, base re-rolled. Video: full policy "
            "episode on a sampler-noised base.",
            "command": "MUJOCO_GL=egl python scripts/viz/record_sampling_demos.py base_pose_noise",
            "metrics": f"sampler draw (rollout, seed {NOISE_SEED}): "
            f"{_label(base_T)} · base survives task.reset: {base_kept} · "
            f"outcome: {outcome}",
            "media": [
                {"type": "image", "file": grid_media},
                {"type": "video", "file": video_media,
                 "label": f"full episode, base at {_label(base_T)}"},
            ],
        },
    )
    del task, policy
    gc.collect()


def demo_offline_rerender() -> None:
    """Delegates to scripts/viz/piper_x_offline_rerender.py (the working post-hoc
    re-render path — the repo's offline DR renderer class is a stub). Replays
    the successful test-dataset episode and re-renders it under 3 visual draws;
    the script writes the manifest fragment itself (--fragment)."""
    for stale in OUT.glob("rerender_*"):  # traj index may differ between runs
        stale.unlink()
    # pinned: this run's shortest successful episode is single-cube, so the
    # frozen config covers the full layout and the A/B check is clean (the
    # newest-run default may only have multi-cube successes — ghost cubes)
    h5 = (REPO / "experiment_output/datagen/piper_x_cubes_in_cup_v1"
          / "PiperXCubesInCupDataGenConfig/20260726_212737/house_0"
          / "trajectories_batch_1_of_1.h5")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/viz/piper_x_offline_rerender.py"),
         "--h5", str(h5), "--draws", "3", "--fragment", "--out", str(OUT)],
        cwd=REPO,
        env={**os.environ, "VIEW": "0"},
    )
    if r.returncode != 0:
        raise RuntimeError(f"piper_x_offline_rerender.py failed (exit {r.returncode})")


def merge_manifest() -> None:
    demos = []
    for name in ALL_DEMOS:
        p = _fragment(name)
        if p.exists():
            demos.append(json.loads(p.read_text()))
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
                "camera_noise": demo_camera_noise,
                "lighting_randomization": demo_lighting_randomization,
                "texture_randomization": demo_texture_randomization,
                "dynamics_randomization": demo_dynamics_randomization,
                "init_pose_noise": demo_init_pose_noise,
                "base_pose_noise": demo_base_pose_noise,
                "offline_rerender": demo_offline_rerender,
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
