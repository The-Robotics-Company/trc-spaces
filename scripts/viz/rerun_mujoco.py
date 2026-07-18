"""Log a MuJoCo rollout to rerun and (optionally) serve it for public-IP viewing.

Uses molmo_spaces.viz.MujocoRerunLogger. By default it drives a PiPER-X showcase
rollout (scripted joint trajectory + gripper), logs the 3D scene, wrist/exo
camera images, and joint/gripper timeseries, writes a .rrd, and serves the rerun
web viewer over a single HTTP port so it's reachable at http://<public-ip>:<port>.

    # write + serve on the (already-open) port 8009:
    python scripts/viz/rerun_mujoco.py --serve --port 8009
    # just write a recording to open locally:
    python scripts/viz/rerun_mujoco.py --out /tmp/piper.rrd

The logger is scene-agnostic; point it at any (model, data) to visualize other
rollouts (datagen episodes, replays) the same way.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco

import rerun as rr
import rerun.blueprint as rrb

from molmo_spaces.robots.piper_x import PiperXRobot
from molmo_spaces.robots.piper_x_config import PiperXRobotConfig
from molmo_spaces.viz import MujocoRerunLogger


def make_blueprint(cam_paths: list[str]) -> rrb.Blueprint:
    """Big 3D scene on the left; camera feeds + scalar plots stacked on the right."""
    cam_views = [rrb.Spatial2DView(origin=p, name=p.split("/")[-1]) for p in cam_paths]
    right = rrb.Vertical(
        *cam_views,
        rrb.TimeSeriesView(origin="/", contents=["/joints/**", "/gripper/**"], name="joints"),
        row_shares=[3] * len(cam_views) + [2],
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world/scene", name="Scene"),
            right,
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )

# Single-port web page: an http page may load the viewer JS from an https CDN
# (only https->http is blocked, not the reverse) and fetch the .rrd same-origin,
# so the whole thing serves over one already-open HTTP port.
_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#0e0f12}}#v{{width:100vw;height:100vh}}</style>
</head><body><div id="v"></div>
<script type="module">
import {{ WebViewer }} from "https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@{ver}/+esm";
// the viewer needs an absolute URL; build it same-origin from the page location
const rrd = new URL("recording.rrd", window.location.href).href;
const viewer = new WebViewer();
await viewer.start(rrd, document.getElementById("v"));
</script></body></html>
"""


def serve_single_port(rrd_path: str, port: int, bind: str, title: str = "molmo_spaces rerun") -> int:
    """Serve the recording + a CDN-backed viewer page over a single HTTP port."""
    import rerun as rr

    servedir = Path(tempfile.mkdtemp(prefix="rerun_serve_"))
    shutil.copy(rrd_path, servedir / "recording.rrd")
    (servedir / "index.html").write_text(_INDEX_HTML.format(ver=rr.__version__, title=title))
    print(f"[rerun] serving {rrd_path} + viewer (v{rr.__version__}) on {bind}:{port}")
    print(f"[rerun] open  http://<public-ip>:{port}/")
    return subprocess.call(
        [sys.executable, "-m", "http.server", str(port),
         "--bind", bind, "--directory", str(servedir)]
    )


def build_piper_scene():
    cfg = PiperXRobotConfig()
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0.5, 0.5, 3], dir=[-0.2, -0.2, -1])
    spec.worldbody.add_light(pos=[0.3, -0.3, 0.9], dir=[-0.3, 0.3, -0.3])  # fill for the gripper
    # camera headlight + ambient so close-up views (esp. the wrist cam looking up at
    # the shadowed gripper underside) are lit, not black.
    spec.visual.headlight.ambient = [0.6, 0.6, 0.6]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.active = 1
    # small ground pad (a huge plane dominates the 3D auto-framing -> robot looks tiny)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.02], rgba=[0.55, 0.57, 0.6, 1.0]
    )
    PiperXRobot.add_robot_to_scene(cfg, spec, prefix=cfg.robot_namespace,
                                   pos=[0, 0, 0], quat=[1, 0, 0, 0])
    PiperXRobot.apply_control_overrides(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    return cfg, model, data


# scripted showcase waypoints: (arm 6-vec, gripper target)
WAYPOINTS = [
    (np.array([0.0, 1.2, -1.2, 0.0, 0.6, 0.0]), 0.05),
    (np.array([0.8, 1.0, -1.0, 0.0, 0.7, 0.0]), 0.05),
    (np.array([0.8, 1.4, -1.6, 0.3, 0.9, 0.5]), 0.0),
    (np.array([0.0, 1.4, -1.6, 0.0, 0.9, 0.0]), 0.0),
    (np.array([-0.8, 1.0, -1.0, -0.3, 0.7, -0.5]), 0.05),
    (np.array([0.0, 1.2, -1.2, 0.0, 0.6, 0.0]), 0.05),
]


def run_showcase(cfg, model, data, logger, seg_steps=45, sim_substeps=4):
    ns = cfg.robot_namespace
    arm_act = [model.actuator(f"{ns}joint{i+1}").id for i in range(6)]
    grip_act = model.actuator(f"{ns}gripper").id
    arm_qadr = [model.joint(f"{ns}joint{i+1}").qposadr[0] for i in range(6)]
    grip_qadr = model.joint(f"{ns}gripper_joint1").qposadr[0]

    q0, g0 = WAYPOINTS[0]
    for a, q in zip(arm_qadr, q0):
        data.qpos[a] = q
    data.ctrl[arm_act] = q0
    data.ctrl[grip_act] = g0
    mujoco.mj_forward(model, data)

    seq = 0
    for s in range(len(WAYPOINTS) - 1):
        qa, ga = WAYPOINTS[s]
        qb, gb = WAYPOINTS[s + 1]
        for k in range(seg_steps):
            t = (k + 1) / seg_steps
            data.ctrl[arm_act] = (1 - t) * qa + t * qb
            data.ctrl[grip_act] = (1 - t) * ga + t * gb
            for _ in range(sim_substeps):
                mujoco.mj_step(model, data)
            scalars = {f"joints/joint{i+1}": data.qpos[arm_qadr[i]] for i in range(6)}
            scalars["gripper/opening"] = abs(data.qpos[grip_qadr]) * 2
            logger.log_step(data, seq=seq, sim_time=data.time, scalars=scalars)
            seq += 1
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/piper_x.rrd", help="Output .rrd path.")
    ap.add_argument("--serve", action="store_true",
                    help="Serve the recording over HTTP for public-IP viewing.")
    ap.add_argument("--port", type=int, default=8009, help="Web viewer HTTP port.")
    ap.add_argument("--bind", default="0.0.0.0", help="Bind address.")
    ap.add_argument("--no-cameras", action="store_true", help="Skip camera image streams.")
    args = ap.parse_args()

    cfg, model, data = build_piper_scene()
    ns = cfg.robot_namespace
    cams = [] if args.no_cameras else [f"{ns}wrist_camera", f"{ns}exo_camera"]

    logger = MujocoRerunLogger(model, app_id="piper_x", cameras=cams)
    logger.init(spawn=False)
    logger.save(args.out)
    if cams:
        cam_paths = [f"world/cameras/{MujocoRerunLogger.camera_path(c)}" for c in cams]
        rr.send_blueprint(make_blueprint(cam_paths))
    logger.log_static_scene()
    n = run_showcase(cfg, model, data, logger)
    logger.close()
    print(f"[rerun] logged {n} steps -> {args.out}")

    if args.serve:
        sys.exit(serve_single_port(args.out, args.port, args.bind, title="PiPER-X — rerun"))


if __name__ == "__main__":
    main()
