"""Convert a LeRobot v3 video dataset to image (PNG) storage for fast training.

Random access into the chunked training mp4s costs ~99 ms/sample (pyav seek +
GOP decode of 2 cameras), which caps an ACT run at ~110 samples/s on 8 vCPUs
(GPU ~30% idle). PNG frames read back in a few ms, making the dataloader
GPU-bound instead of decode-bound.

Decoding here is sequential per episode (one seek per episode per camera via
the dataset's own chunked-video lookup), so the one-off conversion is fast;
PNG encoding is parallelized with lerobot's async image writer.

Usage:
    python scripts/data/format_conversion/lerobot_video_to_image.py \
        --src-root experiment_output/lerobot/piper_x_cubes_in_cup_v1 \
        --dst-root experiment_output/lerobot/piper_x_cubes_in_cup_v1_images \
        --repo-id local/piper_x_cubes_in_cup_v1_images

Expect ~40-100 KB/frame on disk (this dataset: ~500k PNGs, tens of GB) —
write to instance-local disk, not EFS.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

AUTO_KEYS = {"index", "episode_index", "task_index", "frame_index", "timestamp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--src-repo-id", default="local/piper_x_cubes_in_cup_v1")
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/piper_x_cubes_in_cup_v1_images")
    ap.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes (smoke test)")
    ap.add_argument("--image-writer-processes", type=int, default=4)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    src = LeRobotDataset(args.src_repo_id, root=args.src_root, video_backend="pyav")
    video_keys = list(src.meta.video_keys)
    print(f"src: {src.meta.total_episodes} episodes, {src.meta.total_frames} frames, video keys: {video_keys}")

    if args.dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.dst_root)

    dst_features = {}
    for key, ft in src.meta.features.items():
        if key in AUTO_KEYS:
            continue
        ft = dict(ft)
        if ft["dtype"] == "video":
            ft["dtype"] = "image"
            ft.pop("info", None)
        dst_features[key] = ft

    dst = LeRobotDataset.create(
        args.repo_id,
        fps=src.fps,
        features=dst_features,
        root=args.dst_root,
        robot_type=src.meta.robot_type,
        use_videos=False,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    n_eps = src.meta.total_episodes if args.max_episodes is None else min(args.max_episodes, src.meta.total_episodes)
    t_start = time.perf_counter()
    for ep_idx in tqdm(range(n_eps), desc="episodes"):
        ep = src.meta.episodes[ep_idx]
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        rows = src.hf_dataset[lo:hi]  # column dict; no video decode here
        assert all(int(e) == ep_idx for e in rows["episode_index"][:1]), "episode/index misalignment"
        timestamps = [float(t) for t in rows["timestamp"]]

        # one sequential decode per camera for the whole episode
        frames = src._query_videos({k: timestamps for k in video_keys}, ep_idx)

        task = ep["tasks"][0] if len(ep["tasks"]) else ""
        for j in range(hi - lo):
            frame = {}
            for key in dst_features:
                if key in video_keys:
                    img = frames[key][j]
                    img = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    frame[key] = img
                else:
                    v = rows[key][j]
                    frame[key] = np.atleast_1d(np.asarray(v, dtype=np.float32)) if not isinstance(v, str) else v
            frame["task"] = task  # lerobot >=0.4: task is a frame field, not an add_frame kwarg
            dst.add_frame(frame)
        dst.save_episode()

    if hasattr(dst, "finalize"):
        dst.finalize()
    dst.stop_image_writer()
    dt = time.perf_counter() - t_start
    n_frames = dst.meta.total_frames
    print(f"done: {n_eps} episodes, {n_frames} frames in {dt / 60:.1f} min ({n_frames / dt:.0f} frames/s)")
    print(f"dst: {args.dst_root}  repo-id: {args.repo_id}")


if __name__ == "__main__":
    main()
