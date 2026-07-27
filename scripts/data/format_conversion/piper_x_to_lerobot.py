"""Convert PiPER-X cubes-in-cup MlSpaces h5 data to LeRobot format (for ACT / lerobot policies).

PiPER-X port of ``mlspaces_to_lerobot.py`` (which is Franka-DROID only). Differences,
all verified against a real datagen h5 (PiperXCubesInCupDataGenConfig):

  * 6-DOF arm (not 7); state/action layout is eef_9d(9) + gripper(1) + joint(6) = 16.
  * gripper is ONE actuator (coupled fingers): both the obs qpos gripper[0] and the
    1-D ``actions/joint_pos`` gripper command live in actuator units (ctrlrange 0..0.05),
    so BOTH normalize by ``GRIPPER_MAX`` — no separate 0..255 action scale.
  * cameras are ``wrist_camera`` + ``exo_camera``; frames come from sibling mp4s
    (``episode_{i:08d}_{cam}_batch_*.mp4``), since this pipeline writes no
    ``obs/sensor_data`` group and no ``valid_trajectory_index.json``. Trajectories are
    iterated straight from the h5 (``traj_0``..``traj_N``) and paired to the mp4 whose
    episode index matches the traj order within the same batch file.

Frame/action convention is inherited unchanged from the Franka script and the
molmospaces data_format.md: drop the dummy first action + the done-sentinel tail,
``effective = traj_len - 2``, and pair ``action[i+1]`` with ``state[i]``.

Usage:
    python scripts/data/format_conversion/piper_x_to_lerobot.py <datagen_run_dir> \
        --repo-id local/piper_x_cubes
"""
import argparse
import json
import re
from pathlib import Path

import decord
import h5py
import numpy as np
from decord import VideoReader
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

decord.bridge.set_bridge("native")

# PiPER-X gripper actuator ctrlrange is 0.0 .. 0.05 (full open). Both the observed
# finger qpos[0] and the commanded 1-D gripper action are in these units.
GRIPPER_MAX = 0.05
IMG_HW = (180, 320)
ARM_DOF = 6
# eef_9d (9) + gripper (1) + joint_position (6)
STATE_DIM = 9 + 1 + ARM_DOF

# (lerobot short name, MlSpaces camera name / mp4 infix)
CAMERAS = [
    ("wrist", "wrist_camera"),
    ("exo", "exo_camera"),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path, help="Datagen run dir (searched recursively for trajectories_batch_*.h5)")
    ap.add_argument("--repo-id", default="local/piper_x_cubes")
    ap.add_argument("--root", type=Path, default=None, help="Output root (defaults to $HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for quick tests")
    ap.add_argument("--image-writer-processes", type=int, default=2)
    ap.add_argument("--image-writer-threads", type=int, default=8)
    return ap.parse_args()


def decode_json_bytes(h5_row) -> dict:
    raw = h5_row.tobytes() if isinstance(h5_row, np.ndarray) else h5_row
    s = raw.decode("utf-8").rstrip("\x00")
    return json.loads(s) if s else {}


def pose_quat_wxyz_to_eef9d(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] (scalar-first) -> eef_9d = [x,y,z, R[:,0], R[:,1]]."""
    xyz = pose7[..., :3]
    rotmat = Rotation.from_quat(pose7[..., 3:], scalar_first=True).as_matrix()
    rot6d = np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


_BATCH_RE = re.compile(r"trajectories_(batch_\d+_of_\d+)\.h5$")


def _video_path(h5_path: Path, traj_order: int, ml_cam: str) -> Path:
    """mp4 in the same dir whose episode index matches this traj's order within the batch.

    The h5 is ``trajectories_batch_B_of_N.h5`` and frames live in sibling files
    ``episode_{idx:08d}_{cam}_batch_B_of_N.mp4``; traj_j (h5 key order) -> episode idx j.
    Falls back to the j-th sorted mp4 for that camera if the exact name is absent.
    """
    m = _BATCH_RE.search(h5_path.name)
    if m:
        cand = h5_path.parent / f"episode_{traj_order:08d}_{ml_cam}_{m.group(1)}.mp4"
        if cand.exists():
            return cand
    mp4s = sorted(h5_path.parent.glob(f"episode_*_{ml_cam}_*.mp4"))
    if traj_order < len(mp4s):
        return mp4s[traj_order]
    raise FileNotFoundError(f"No mp4 for {ml_cam} traj#{traj_order} next to {h5_path}")


def load_video_frames(h5_path: Path, traj_order: int, ml_cam: str, count: int) -> np.ndarray:
    vr = VideoReader(str(_video_path(h5_path, traj_order, ml_cam)))
    if count > len(vr):
        raise RuntimeError(f"Requested {count} frames but video has {len(vr)}")
    return vr.get_batch(list(range(count))).asnumpy()  # (count, H, W, 3) RGB uint8


def resize_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if frame.shape[:2] == (h, w):
        return frame
    return np.array(Image.fromarray(frame).resize((w, h), resample=Image.BICUBIC))


def build_features() -> dict:
    feats = {}
    for short, _ in CAMERAS:
        feats[f"observation.images.{short}"] = {
            "dtype": "video",
            "shape": (*IMG_HW, 3),
            "names": ["height", "width", "channel"],
        }
    feats["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]}
    feats["action"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["action"]}
    for prefix in ("observation.state", "action"):
        feats[f"{prefix}.eef_9d"] = {"dtype": "float32", "shape": (9,), "names": ["eef_9d"]}
        feats[f"{prefix}.gripper_position"] = {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]}
        feats[f"{prefix}.joint_position"] = {"dtype": "float32", "shape": (ARM_DOF,), "names": ["joint_position"]}
    return feats


def iter_trajectories(data_dir: Path):
    """Yield (h5_path, traj_key, traj_order, traj_len) straight from the h5 files."""
    h5s = sorted(data_dir.rglob("trajectories_batch_*.h5"))
    if not h5s:
        raise RuntimeError(f"No trajectories_batch_*.h5 under {data_dir}")
    for h5_path in h5s:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
            for order, tk in enumerate(traj_keys):
                yield h5_path, tk, order, f[tk]["obs/agent/qpos"].shape[0]


def read_policy_fps(h5_path: Path, traj_key: str) -> int:
    with h5py.File(h5_path, "r") as f:
        scene = decode_json_bytes(f[traj_key]["obs_scene"][()])
    return round(1000.0 / scene["policy_dt_ms"])


def convert_episode(dataset: LeRobotDataset, h5_path: Path, traj_key: str,
                    traj_order: int, traj_len: int) -> int:
    """Return number of frames written, or 0 if skipped."""
    # Drop dummy first action, done sentinel last action, and last 2 states
    # (per https://github.com/allenai/molmospaces/blob/main/docs/data_format.md)
    effective = traj_len - 2
    if effective < 1:
        return 0

    with h5py.File(h5_path, "r") as f:
        tg = f[traj_key]
        task: str = decode_json_bytes(tg["obs_scene"][()])["task_description"]

        qpos = [decode_json_bytes(tg["obs/agent/qpos"][i]) for i in range(effective)]
        obs_eef9d = pose_quat_wxyz_to_eef9d(tg["obs/extra/tcp_pose"][:effective])

        # action[i] pairs with state[i] after dropping the padded first action
        act_joint = [decode_json_bytes(tg["actions/joint_pos"][i + 1]) for i in range(effective)]
        act_ee_rows = np.stack(
            [np.asarray(decode_json_bytes(tg["actions/ee_pose"][i + 1])["arm"], dtype=np.float32)
             for i in range(effective)],
            axis=0,
        )
        act_eef9d = pose_quat_wxyz_to_eef9d(act_ee_rows)

        cam_frames = {
            short: load_video_frames(h5_path, traj_order, ml_cam, effective)
            for short, ml_cam in CAMERAS
        }

    for j in range(effective):
        joint_pos = np.asarray(qpos[j]["arm"], dtype=np.float32)
        # obs gripper: coupled finger qpos[0], actuator units -> [0,1]
        gripper_pos = np.asarray([qpos[j]["gripper"][0] / GRIPPER_MAX], dtype=np.float32)
        act_joint_pos = np.asarray(act_joint[j]["arm"], dtype=np.float32)
        # action gripper: single commanded actuator value, same units -> [0,1]
        act_gripper = np.asarray([act_joint[j]["gripper"][0] / GRIPPER_MAX], dtype=np.float32)

        frame = {
            "observation.state": np.concatenate([obs_eef9d[j], gripper_pos, joint_pos]),
            "observation.state.eef_9d": obs_eef9d[j],
            "observation.state.gripper_position": gripper_pos,
            "observation.state.joint_position": joint_pos,
            "action": np.concatenate([act_eef9d[j], act_gripper, act_joint_pos]),
            "action.eef_9d": act_eef9d[j],
            "action.gripper_position": act_gripper,
            "action.joint_position": act_joint_pos,
        }
        for short, frames in cam_frames.items():
            frame[f"observation.images.{short}"] = resize_rgb(frames[j], IMG_HW)
        frame["task"] = task  # lerobot >=0.4: task is a frame field, not an add_frame kwarg
        dataset.add_frame(frame)

    dataset.save_episode()
    return effective


def write_modality_json(root: Path):
    split_16 = {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": STATE_DIM},
    }
    modality = {
        "state": split_16,
        "action": split_16,
        "video": {short: {"original_key": f"observation.images.{short}"} for short, _ in CAMERAS},
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    with (root / "meta" / "modality.json").open("w") as f:
        json.dump(modality, f, indent=2)


def main():
    args = parse_args()

    trajs = list(iter_trajectories(args.data_dir))
    if args.max_episodes:
        trajs = trajs[: args.max_episodes]
    if not trajs:
        raise RuntimeError(f"No trajectories found in {args.data_dir}")

    fps = read_policy_fps(trajs[0][0], trajs[0][1])
    print(f"Converting {len(trajs)} episodes from {args.data_dir} (fps={fps})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=build_features(),
        robot_type="piper_x",
        root=args.root,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    skipped = 0
    for h5_path, traj_key, traj_order, traj_len in tqdm(trajs, desc="episodes"):
        try:
            n = convert_episode(dataset, h5_path, traj_key, traj_order, traj_len)
            if n == 0:
                skipped += 1
            else:
                total_frames += n
        except Exception as e:
            print(f"\nFAILED {h5_path}::{traj_key}: {e}")
            skipped += 1

    write_modality_json(dataset.root)

    print(f"\nDone. {total_frames} frames across {len(trajs) - skipped} episodes ({skipped} skipped).")
    print(f"Dataset at: {dataset.root}")


if __name__ == "__main__":
    main()
