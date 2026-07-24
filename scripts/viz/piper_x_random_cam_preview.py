"""Visualize the front-sector randomized exo camera (cubes-in-cup subversion).

Samples one episode with ``PiperXCubesInCupRandomCamDataGenConfig``, then re-runs
the REAL camera-setup path (spherical sampling + segmentation visibility check)
N times to collect exo-camera poses. Prints measured position stats, saves a
montage of actual rendered views from a few sampled cameras, and opens the
passive viewer with every sampled camera drawn as a sphere + forward arrow
(cyan) and the workspace center as a red sphere.

VIEW=0 skips the viewer. argv[1] = seed (default 0), argv[2] = #samples (default 30).
"""

import logging
import os
import sys
import time

import mujoco
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (  # noqa: E402
    PiperXCubesInCupRandomCamDataGenConfig,
)

BOARD2_BACK_X = 0.62  # back edge of the upper board (board2 spans x 0.27..0.62)


def _z_aligned_mat(z: np.ndarray) -> np.ndarray:
    """Rotation matrix whose +z axis is the given direction (for arrow geoms)."""
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _add_marker(scn, gtype, size, pos, mat, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64),
                        np.asarray(pos, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    show = os.environ.get("VIEW", "1") == "1"
    np.random.seed(seed)

    exp_config = PiperXCubesInCupRandomCamDataGenConfig()
    exp_config.seed = seed
    sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)

    task = None
    for attempt in range(10):
        try:
            task = sampler.sample_task()
            break
        except Exception as e:  # noqa: BLE001
            print(f"[attempt {attempt}] sampling failed: {e}")
    if task is None:
        print("FAIL: could not sample a task")
        return 1

    env = task.env
    center = sampler.get_workspace_center(env)
    print(f"workspace center: {center.round(3)}")

    # Re-run the real camera setup path N times, capturing the exo pose each time.
    cam = env.camera_manager.registry["exo_camera"]
    samples = [(cam.pos.copy(), cam.forward.copy(), cam.up.copy(), cam.fov)]
    for _ in range(n_samples - 1):
        sampler.setup_cameras(env)
        cam = env.camera_manager.registry["exo_camera"]
        samples.append((cam.pos.copy(), cam.forward.copy(), cam.up.copy(), cam.fov))

    pos = np.array([s[0] for s in samples])
    fov = np.array([s[3] for s in samples])
    dist = np.linalg.norm(pos - center, axis=1)
    beyond = int(np.sum(pos[:, 0] > BOARD2_BACK_X))
    print(f"sampled {len(samples)} exo poses:")
    print(f"  x [{pos[:, 0].min():+.3f}, {pos[:, 0].max():+.3f}]  "
          f"y [{pos[:, 1].min():+.3f}, {pos[:, 1].max():+.3f}]  "
          f"z [{pos[:, 2].min():+.3f}, {pos[:, 2].max():+.3f}]")
    print(f"  dist-to-center [{dist.min():.3f}, {dist.max():.3f}] m, "
          f"fov [{fov.min():.1f}, {fov.max():.1f}] deg")
    print(f"  beyond board2 back edge (x > {BOARD2_BACK_X}): "
          f"{beyond}/{len(samples)}")

    # Montage of actual views from 6 sampled cameras (render BEFORE the viewer
    # opens so the offscreen GL context doesn't fight the viewer's).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        picks = np.linspace(0, len(samples) - 1, 6).astype(int)
        fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
        for ax, i in zip(axes.flat, picks):
            p, f, u, v = samples[i]
            cam = env.camera_manager.registry["exo_camera"]
            cam.pos, cam.forward, cam.up, cam.fov = p, f, u, v
            ax.imshow(env.render_rgb_frame("exo_camera"))
            ax.set_title(f"#{i}: pos ({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) fov {v:.0f}",
                         fontsize=8)
            ax.axis("off")
        fig.suptitle(f"exo_camera samples, seed {seed}", fontsize=11)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"random_cam_views_seed{seed}.png")
        fig.savefig(out, dpi=110)
        print(f"view montage: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"montage skipped: {e}")

    if not show:
        return 0

    import mujoco.viewer

    mj_data = env.mj_datas[env.current_batch_index]
    with mujoco.viewer.launch_passive(mj_data.model, mj_data) as viewer:
        scn = viewer.user_scn
        scn.ngeom = 0
        eye = np.eye(3)
        _add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                    center, eye, [1.0, 0.1, 0.1, 0.9])
        for p, f, _u, _v in samples:
            _add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                        p, eye, [0.1, 0.8, 0.9, 0.9])
            _add_marker(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.004, 0.004, 0.16],
                        p, _z_aligned_mat(f), [0.1, 0.8, 0.9, 0.7])
        print("viewer open: cyan spheres = sampled cameras, arrows = view "
              "direction, red = workspace center. Close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
