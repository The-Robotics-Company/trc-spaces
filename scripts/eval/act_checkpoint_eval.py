"""Evaluate one LeRobot ACT checkpoint in sim and log rate + rollout video to W&B.

Runs ``PiperXCubesInCupACTEvalConfig`` (the datagen env with the planner swapped
for the checkpoint), reads per-episode success from the recorded h5s, stitches
the first --video-episodes rollouts (wrist|exo side by side, green/red border by
success) into one mp4, and logs to a W&B run:

  eval/success_rate, eval/successes, eval/episodes, eval/rollout (video)

logged at step=--step (the training step, parsed from a lerobot checkpoint path
``.../checkpoints/<step>/pretrained_model`` when --step is omitted). A fixed
--wandb-run-id + resume=allow makes successive checkpoint evals append to one
curve — one eval run per training run, next to it in the same project (the
piper-x-policy eval_watcher.sh pattern).

Example:
  python scripts/eval/act_checkpoint_eval.py \
      --ckpt outputs/train/act_cubes/checkpoints/020000/pretrained_model \
      --episodes 16 --wandb-project piper-x-act

Continuous use: scripts/eval/act_eval_watcher.sh
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "PiperXCubesInCupACTEvalConfig"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="LeRobot pretrained_model dir")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out-root", type=Path, default=Path("experiment_output/act_eval"))
    ap.add_argument("--step", type=int, default=None,
                    help="Training step for the W&B x-axis (default: from ckpt path)")
    ap.add_argument("--video-episodes", type=int, default=6)
    ap.add_argument("--wandb-project", default="piper-x-act")
    ap.add_argument("--wandb-run-id", default=None,
                    help="Fixed eval-run id; successive evals append (default eval-<train run dir>)")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


def infer_step(ckpt: Path) -> int | None:
    m = re.search(r"checkpoints/0*(\d+)", str(ckpt))
    return int(m.group(1)) if m else None


def infer_run_name(ckpt: Path) -> str:
    # .../<train_run_dir>/checkpoints/<step>/pretrained_model -> <train_run_dir>
    parts = ckpt.resolve().parts
    if "checkpoints" in parts:
        name = parts[parts.index("checkpoints") - 1]
    else:
        name = ckpt.resolve().parent.name
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def run_rollouts(args) -> Path:
    """Run the eval config in a fresh process; return this eval's run dir."""
    out_root = args.out_root / f"step_{args.step if args.step is not None else 'x'}_{time.strftime('%Y%m%d_%H%M%S')}"
    env = os.environ.copy()
    env.update(
        ACT_EVAL_CKPT=str(args.ckpt.resolve()),
        ACT_EVAL_EPISODES=str(args.episodes),
        ACT_EVAL_WORKERS=str(args.workers),
        ACT_EVAL_SEED=str(args.seed),
        ACT_EVAL_OUT=str(out_root),
    )
    cmd = [sys.executable, "-m", "molmo_spaces.data_generation.main", CONFIG_NAME]
    print(f"[eval] rollouts: {' '.join(cmd)}  (episodes={args.episodes} workers={args.workers})")
    res = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"rollout run failed (exit {res.returncode}); see output above")
    if not any(out_root.rglob("trajectories_batch_*.h5")):
        raise RuntimeError(f"no trajectory h5s under {out_root}")
    return out_root


def episode_results(run_dir: Path) -> list[dict]:
    """[{h5, traj, order, success, steps}] in deterministic order."""
    out = []
    for h5_path in sorted(run_dir.rglob("trajectories_batch_*.h5")):
        with h5py.File(h5_path, "r") as f:
            for order, tk in enumerate(sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))):
                succ = np.asarray(f[tk]["success"])
                effective = max(len(succ) - 2, 1)  # drop done-sentinel tail like the converter
                out.append({
                    "h5": h5_path, "traj": tk, "order": order,
                    "success": bool(succ[effective - 1]), "steps": int(effective),
                })
    return out


def _episode_mp4(h5_path: Path, order: int, cam: str) -> Path | None:
    m = re.search(r"trajectories_(batch_\d+_of_\d+)\.h5$", h5_path.name)
    if m:
        p = h5_path.parent / f"episode_{order:08d}_{cam}_{m.group(1)}.mp4"
        if p.exists():
            return p
    mp4s = sorted(h5_path.parent.glob(f"episode_*_{cam}_*.mp4"))
    return mp4s[order] if order < len(mp4s) else None


def build_rollout_video(episodes: list[dict], out_path: Path, max_episodes: int, fps: int = 15):
    """Stitch wrist|exo per episode, 4px green/red border by success, concat in time."""
    writer = None
    for ep in episodes[:max_episodes]:
        caps = []
        for cam in ("wrist_camera", "exo_camera"):
            p = _episode_mp4(ep["h5"], ep["order"], cam)
            if p is None:
                break
            caps.append(cv2.VideoCapture(str(p)))
        if len(caps) != 2:
            for c in caps:
                c.release()
            continue
        color = (40, 200, 40) if ep["success"] else (40, 40, 220)  # BGR
        while True:
            frames = []
            for c in caps:
                ok, fr = c.read()
                if not ok:
                    break
                frames.append(fr)
            if len(frames) != 2:
                break
            tile = np.concatenate(frames, axis=1)
            tile = cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
            if writer is None:
                h, w = tile.shape[:2]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(tile)
        for c in caps:
            c.release()
    if writer is not None:
        writer.release()
        return out_path
    return None


def main():
    args = parse_args()
    if args.step is None:
        args.step = infer_step(args.ckpt)
    run_dir = run_rollouts(args)

    episodes = episode_results(run_dir)
    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    rate = successes / n if n else 0.0
    print(f"[eval] step={args.step}: {successes}/{n} success ({100 * rate:.1f}%)")

    video = build_rollout_video(episodes, run_dir / "eval_rollout.mp4", args.video_episodes)
    if video:
        print(f"[eval] rollout video: {video}")

    summary = {"step": args.step, "episodes": n, "successes": successes,
               "success_rate": rate, "run_dir": str(run_dir), "ckpt": str(args.ckpt)}
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_wandb:
        import wandb

        run_id = args.wandb_run_id or f"eval-{infer_run_name(args.ckpt)}"
        run = wandb.init(project=args.wandb_project, id=run_id, name=run_id,
                         resume="allow", job_type="sim-eval")
        payload = {"eval/success_rate": rate, "eval/successes": successes, "eval/episodes": n}
        if video:
            payload["eval/rollout"] = wandb.Video(str(video), format="mp4",
                                                  caption=f"step {args.step}: {successes}/{n}")
        run.log(payload, step=args.step)
        run.finish()
        print(f"[eval] logged to W&B project={args.wandb_project} run={run_id} step={args.step}")

    # machine-readable one-liner for the watcher
    print(f"EVAL_RESULT step={args.step} rate={rate:.4f} successes={successes} episodes={n}")


if __name__ == "__main__":
    main()
