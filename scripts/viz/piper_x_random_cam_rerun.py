"""Rerun visualization of the randomized exo camera (cubes-in-cup subversion).

Samples one episode with ``PiperXCubesInCupRandomCamDataGenConfig``, re-runs the
real camera-setup path N times, and logs to rerun:

  * the full MuJoCo scene (meshes/boxes) via MujocoRerunLogger,
  * each sampled camera as a Pinhole frustum posed in 3D with its actual
    rendered image displayed on the image plane ("2D on 3D"),
  * the fixed workspace aim point (red).

VIEW=0 writes a .rrd next to this script instead of spawning the viewer.
argv[1] = seed (default 0), argv[2] = #samples (default 12).
"""

import logging
import os
import sys

import numpy as np
import rerun as rr

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

from molmo_spaces.data_generation.config.piper_x_cubes_in_cup_datagen import (  # noqa: E402
    PiperXCubesInCupRandomCamDataGenConfig,
)
from molmo_spaces.viz.rerun_logger import MujocoRerunLogger  # noqa: E402


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 12
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
    w, h = exp_config.camera_config.img_resolution

    # Sample N poses through the real setup path; render each view immediately.
    samples = []
    for i in range(n_samples):
        if i > 0:
            sampler.setup_cameras(env)
        cam = env.camera_manager.registry["exo_camera"]
        samples.append((cam.get_pose().copy(), cam.fov,
                        env.render_rgb_frame("exo_camera").copy()))
    print(f"sampled {len(samples)} exo poses (seed {seed})")

    mj_data = env.mj_datas[env.current_batch_index]
    logger = MujocoRerunLogger(mj_data.model, app_id="piper_x_random_cams")
    logger.init(spawn=show)
    if not show:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"random_cams_seed{seed}.rrd")
        logger.save(out)
        print(f"saving to {out}")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    logger.log_static_scene()
    logger.log_step(mj_data, 0)  # place all bodies at their current pose

    rr.log("world/workspace_center",
           rr.Points3D([center], colors=[[255, 40, 40]], radii=[0.02],
                       labels=["aim"]), static=True)

    for i, (pose, fov, img) in enumerate(samples):
        path = f"world/cams/cam_{i:02d}"
        # Camera.get_pose() is cam-to-world with x=right, y=down, z=forward (RDF)
        rr.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
               static=True)
        fy = (h / 2.0) / np.tan(np.radians(fov) / 2.0)
        rr.log(path, rr.Pinhole(resolution=[w, h], focal_length=[fy, fy],
                                camera_xyz=rr.ViewCoordinates.RDF,
                                image_plane_distance=0.25), static=True)
        rr.log(f"{path}/image", rr.Image(img), static=True)

    print("rerun log complete: frustums under world/cams/, images on the "
          "image planes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
