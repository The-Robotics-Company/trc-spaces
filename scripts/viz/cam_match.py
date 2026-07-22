"""Live real-vs-sim wrist camera matching tool.

Shows three panels side by side — real camera 1, real camera 2, and the MuJoCo
wrist camera — and lets you nudge the sim camera's pose/FOV with the keyboard
until it matches the real wrist cam. Press ``p`` to print the tuned camera as
an MJCF line ready to paste into piper_x.xml.

    conda activate mlspaces
    python scripts/viz/cam_match.py --cam1 0 --cam2 2

Keys (lowercase = -, uppercase = +):
    q/Q w/W e/E   camera position x y z   (2 mm)
    r/R t/T y/Y   camera euler    x y z   (0.5 deg)
    u/U           fovy                    (0.5 deg)
    g/G           gripper close/open      (5 mm)
    [ ]           cycle sim arm pose through showcase waypoints
    0             reset camera to the MJCF values
    p             print the MJCF <camera> line to the terminal
    ESC           quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from rerun_mujoco import build_piper_scene, WAYPOINTS  # noqa: E402

PANEL_W, PANEL_H = 640, 480
POS_STEP = 0.002   # m
ANG_STEP = 0.5     # deg
FOV_STEP = 0.5     # deg
GRIP_STEP = 0.005  # m per keypress; gripper_joint1 range is 0 (closed) .. 0.05 (open)


class SimCam:
    """Wraps the wrist camera's model fields; edits show up on the next render."""

    def __init__(self, model, data, cam_name):
        self.model, self.data = model, data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise SystemExit(f"camera {cam_name!r} not in model")
        self._init_pos = model.cam_pos[self.cam_id].copy()
        self._init_quat = model.cam_quat[self.cam_id].copy()
        self._init_fovy = float(model.cam_fovy[self.cam_id])
        self.reset()

    def reset(self):
        self.pos = self._init_pos.copy()
        w, x, y, z = self._init_quat
        self.euler = Rotation.from_quat([x, y, z, w]).as_euler("XYZ", degrees=True)
        self.fovy = self._init_fovy
        self.apply()

    def apply(self):
        x, y, z, w = Rotation.from_euler("XYZ", self.euler, degrees=True).as_quat()
        self.model.cam_pos[self.cam_id] = self.pos
        self.model.cam_quat[self.cam_id] = [w, x, y, z]
        self.model.cam_fovy[self.cam_id] = self.fovy
        mujoco.mj_forward(self.model, self.data)

    def mjcf_line(self):
        p = " ".join(f"{v:.5f}" for v in self.pos)
        w, x, y, z = self.model.cam_quat[self.cam_id]
        q = f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"
        return f'<camera name="wrist_camera" pos="{p}" quat="{q}" fovy="{self.fovy:.1f}"/>'


def set_arm_pose(cfg, model, data, waypoint, grip=None):
    ns = cfg.robot_namespace
    q, wp_grip = waypoint
    if grip is None:
        grip = wp_grip
    for i, qi in enumerate(q):
        data.qpos[model.joint(f"{ns}joint{i+1}").qposadr[0]] = qi
    data.qpos[model.joint(f"{ns}gripper_joint1").qposadr[0]] = grip
    data.qpos[model.joint(f"{ns}gripper_joint2").qposadr[0]] = -grip
    mujoco.mj_forward(model, data)
    return grip


def label(img, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam1", type=int, default=0, help="V4L index of real camera 1")
    ap.add_argument("--cam2", type=int, default=2, help="V4L index of real camera 2")
    args = ap.parse_args()

    caps = []
    for idx in (args.cam1, args.cam2):
        cap = cv2.VideoCapture(idx)
        # MJPG: two uncompressed YUYV streams exceed a shared USB2 bus and
        # cause intermittent read failures (flicker); compressed streams fit.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PANEL_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"warning: could not open /dev/video{idx}, panel will show NO SIGNAL")
        caps.append(cap)
    last_good = [np.zeros((PANEL_H, PANEL_W, 3), np.uint8) for _ in caps]
    stale = [0, 0]

    cfg, model, data = build_piper_scene(light_bg=True)
    ns = cfg.robot_namespace
    cam = SimCam(model, data, f"{ns}wrist_camera")
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    wp = 0
    grip = set_arm_pose(cfg, model, data, WAYPOINTS[wp])

    # +step on uppercase, -step on lowercase; (target array, index, step)
    bindings = {}
    for key, (arr_name, i, step) in {
        "q": ("pos", 0, POS_STEP), "w": ("pos", 1, POS_STEP), "e": ("pos", 2, POS_STEP),
        "r": ("euler", 0, ANG_STEP), "t": ("euler", 1, ANG_STEP), "y": ("euler", 2, ANG_STEP),
    }.items():
        bindings[ord(key)] = (arr_name, i, -step)
        bindings[ord(key.upper())] = (arr_name, i, +step)

    print(__doc__)
    print("initial:", cam.mjcf_line())
    win = "real1 | real2 | sim wrist cam"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while True:
        panels = []
        for n, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                last_good[n] = cv2.resize(frame, (PANEL_W, PANEL_H))
                stale[n] = 0
            else:
                stale[n] += 1  # keep last good frame instead of flashing black
            tag = f"real cam {n+1}" + (" [NO SIGNAL]" if stale[n] > 30 else "")
            panels.append(label(last_good[n].copy(), tag))

        renderer.update_scene(data, camera=f"{ns}wrist_camera")
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        sim = cv2.rotate(sim, cv2.ROTATE_180)  # DC1 is mounted upside down
        panels.append(label(sim, f"sim wrist cam 180 (pose {wp})"))

        canvas = np.hstack(panels)
        bar = np.zeros((56, canvas.shape[1], 3), np.uint8)
        e = cam.euler
        status = (f"pos {cam.pos[0]:+.4f} {cam.pos[1]:+.4f} {cam.pos[2]:+.4f}   "
                  f"euler {e[0]:+.1f} {e[1]:+.1f} {e[2]:+.1f}   fovy {cam.fovy:.1f}   "
                  f"grip {grip*1000:.0f}mm")
        cv2.putText(bar, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(bar, "q/w/e pos  r/t/y euler  u fovy  g gripper  (upper=+)  [ ] pose  0 reset  p print  ESC quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.imshow(win, np.vstack([canvas, bar]))

        k = cv2.waitKey(15) & 0xFF
        if k == 27:
            break
        elif k in bindings:
            arr_name, i, step = bindings[k]
            getattr(cam, arr_name)[i] += step
            cam.apply()
        elif k in (ord("u"), ord("U")):
            cam.fovy += FOV_STEP if k == ord("U") else -FOV_STEP
            cam.apply()
        elif k in (ord("g"), ord("G")):
            grip = float(np.clip(grip + (GRIP_STEP if k == ord("G") else -GRIP_STEP), 0.0, 0.05))
            set_arm_pose(cfg, model, data, WAYPOINTS[wp], grip=grip)
        elif k == ord("0"):
            cam.reset()
        elif k == ord("p"):
            print(cam.mjcf_line())
        elif k in (ord("["), ord("]")):
            wp = (wp + (1 if k == ord("]") else -1)) % len(WAYPOINTS)
            grip = set_arm_pose(cfg, model, data, WAYPOINTS[wp])

    print("final:  ", cam.mjcf_line())
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
