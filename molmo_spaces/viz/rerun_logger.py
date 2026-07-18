"""Log a MuJoCo model + rollout to rerun.

A scene-agnostic logger: given a compiled ``MjModel`` it logs the static
geometry once (meshes / boxes / spheres / capsules / planes) under a transform
tree, then per control step logs each body's world transform, optional camera
images, and optional scalar timeseries. Works for both live rollouts (call
``log_step`` as you step the sim) and replay (drive a rebuilt ``MjData`` from a
saved trajectory).

Serving for public-IP viewing is handled by the CLI (scripts/viz/rerun_mujoco.py):
write a ``.rrd`` then ``rerun --serve-web recording.rrd --web-viewer-port 8009``,
which hosts the viewer + recording over a single HTTP port.
"""

from __future__ import annotations

import numpy as np
import mujoco

try:
    import rerun as rr
except ImportError as e:  # pragma: no cover
    raise ImportError("rerun is required for molmo_spaces.viz; `pip install rerun-sdk`") from e


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


class MujocoRerunLogger:
    """Logs a MuJoCo scene + rollout to rerun.

    Args:
        model: compiled MjModel.
        app_id: rerun application id.
        root: entity-path root for the 3D scene.
        cameras: list of MJCF camera names to render as image streams (None = none).
        camera_res: (height, width) for rendered camera images.
        timeline: name of the sequence timeline.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        app_id: str = "molmo_spaces",
        root: str = "world",
        cameras: list[str] | None = None,
        camera_res: tuple[int, int] = (240, 320),
        timeline: str = "step",
    ) -> None:
        self.model = model
        self.app_id = app_id
        self.root = root
        self.timeline = timeline
        self.cameras = cameras or []
        self._renderer = None
        if self.cameras:
            h, w = camera_res
            self._renderer = mujoco.Renderer(model, height=h, width=w)
        # cache body names once
        self._body_names = [model.body(i).name or f"body_{i}" for i in range(model.nbody)]

    # ---- setup -----------------------------------------------------------
    def init(self, spawn: bool = False) -> None:
        rr.init(self.app_id, spawn=spawn)

    def save(self, path: str) -> None:
        rr.save(path)

    def connect(self, url: str | None = None) -> None:
        rr.connect_grpc(url) if url else rr.connect_grpc()

    # ---- static geometry -------------------------------------------------
    def log_static_scene(self) -> None:
        """Log all geoms once, each under ``root/scene/<body>/<geom>`` with its
        constant geom-local transform, so per-step body transforms place them."""
        m = self.model
        for gi in range(m.ngeom):
            body_id = int(m.geom_bodyid[gi])
            body = self._body_names[body_id]
            gname = m.geom(gi).name or f"geom_{gi}"
            path = f"{self.root}/scene/{body}/{gname}"
            rgba = m.geom_rgba[gi]
            color = (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)
            # constant geom-local transform (geom frame within its body)
            rr.log(path, rr.Transform3D(
                translation=m.geom_pos[gi],
                quaternion=_wxyz_to_xyzw(m.geom_quat[gi]),
            ), static=True)
            self._log_geom_shape(path + "/shape", gi, color)

    def _log_geom_shape(self, path: str, gi: int, color: np.ndarray) -> None:
        m = self.model
        gtype = m.geom_type[gi]
        size = m.geom_size[gi]
        G = mujoco.mjtGeom
        if gtype == G.mjGEOM_MESH:
            mid = int(m.geom_dataid[gi])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            verts = m.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float32)
            faces = m.mesh_face[fa:fa + fn].reshape(-1, 3).astype(np.uint32)
            rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                   albedo_factor=color), static=True)
        elif gtype == G.mjGEOM_BOX:
            rr.log(path, rr.Boxes3D(half_sizes=[size[:3]], colors=[color]), static=True)
        elif gtype == G.mjGEOM_SPHERE:
            rr.log(path, rr.Ellipsoids3D(half_sizes=[[size[0]] * 3], colors=[color]), static=True)
        elif gtype in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
            # mujoco size = (radius, half-length); rerun capsule spans +z by length
            r, hl = float(size[0]), float(size[1])
            rr.log(path, rr.Capsules3D(radii=[r], lengths=[2 * hl],
                                       translations=[[0, 0, -hl]], colors=[color]), static=True)
        elif gtype == G.mjGEOM_PLANE:
            # render a large thin slab for the ground
            sx = size[0] if size[0] > 0 else 5.0
            sy = size[1] if size[1] > 0 else 5.0
            rr.log(path, rr.Boxes3D(half_sizes=[[sx, sy, 0.001]], colors=[color]), static=True)
        # else: unsupported geom type -> skip silently

    # ---- per-step --------------------------------------------------------
    def log_step(
        self,
        data: mujoco.MjData,
        seq: int,
        sim_time: float | None = None,
        scalars: dict[str, float] | None = None,
    ) -> None:
        rr.set_time(self.timeline, sequence=seq)
        if sim_time is not None:
            rr.set_time("sim_time", duration=float(sim_time))

        # body world transforms
        for bi in range(1, self.model.nbody):
            body = self._body_names[bi]
            rr.log(f"{self.root}/scene/{body}", rr.Transform3D(
                translation=data.xpos[bi],
                quaternion=_wxyz_to_xyzw(data.xquat[bi]),
            ))

        # camera image streams (+ 3D frustum)
        for cam in self.cameras:
            self._log_camera(data, cam)

        # scalar timeseries
        if scalars:
            for name, val in scalars.items():
                rr.log(name, rr.Scalars(float(val)))

    def _log_camera(self, data: mujoco.MjData, cam: str) -> None:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cam_id < 0:
            return
        self._renderer.update_scene(data, camera=cam)
        img = self._renderer.render()
        rr.log(f"{self.root}/cameras/{self.camera_path(cam)}", rr.Image(img))

    @staticmethod
    def camera_path(cam: str) -> str:
        """Sanitize a (possibly namespaced) camera name into a single path component."""
        return cam.replace("/", "_")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
