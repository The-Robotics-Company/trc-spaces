#!/usr/bin/env bash
# Build a DAgger training dataset: <BASE> + DAgger expert suffixes -> <DST>.
# Four stages, mirroring gen_cubes_6k_pipeline.sh:
#   1. dagger h5 -> lerobot video dataset (successful-only, student prefix sliced
#      off via the sidecar switch_step)
#   2. video -> image format (video decode in training is the 440ms/sample trap)
#   3. aggregate BASE + dagger images -> DST
#   4. verify DST, then print the exact cleanup that reclaims the inputs
#
# Round 1 (2026-08-03) built v3 = v2_images + 2499 DAgger eps = 8197 eps/3.42M
# frames. For round 2 pass BASE=..._v3_images DST=..._v4_images TAG=dagger2.
#
# Sizing, learned the hard way: stage 3 writes a FULL COPY (v3 = 273G) and an
# ENOSPC mid-merge leaves a partial dataset that has to be deleted and redone.
# The preflight below refuses to start without room. Each DAgger episode is
# ~75MB in image format, so DST ≈ BASE + 0.075GB × episodes.
#
# Usage: build_dagger_v3.sh <dagger_datagen_run_dir>
# Env: BASE, DST, TAG, LR (dataset root), NEED_GB (preflight override)
set -euo pipefail
export PATH=$HOME/miniconda3/envs/lerobot/bin:$PATH

TRC=$HOME/trc-spaces
RUN=${1:?usage: build_dagger_v3.sh <dagger_datagen_run_dir>}
LR=${LR:-$TRC/experiment_output/lerobot}
BASE=${BASE:-piper_x_cubes_in_cup_v2_images}
DST=${DST:-piper_x_cubes_in_cup_v3_images}
TAG=${TAG:-dagger1}

# ---- preflight: stage 3 needs room for a full copy of BASE + the new episodes
BASE_GB=$(du -sBG "$LR/$BASE" 2>/dev/null | cut -f1 | tr -d 'G' || echo 0)
N_EPS=$(ls "$RUN"/house_*/trajectories_batch_*.h5 2>/dev/null | wc -l)
NEED_GB=${NEED_GB:-$(( BASE_GB + (N_EPS * 4 / 50) + 80 ))}
AVAIL_GB=$(df -B1G --output=avail "$LR" | tail -1 | tr -d ' ')
echo "=== preflight: BASE $BASE=${BASE_GB}G, $N_EPS dagger work items"
echo "=== $LR has ${AVAIL_GB}G free, need ~${NEED_GB}G for stages 1-3"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "ERROR: not enough space. Free some first — the usual candidates:" >&2
  du -shx "$LR"/* 2>/dev/null | sort -h | tail -5 >&2
  echo "  (a dataset superseded by an earlier merge is always safe to delete)" >&2
  exit 1
fi

echo "=== [$(date -u +%T)] stage 1: h5 -> lerobot video (dagger-sliced)"
python "$TRC/scripts/data/format_conversion/piper_x_to_lerobot.py" "$RUN" \
  --repo-id "local/piper_x_cubes_in_cup_$TAG" \
  --root "$LR/piper_x_cubes_in_cup_$TAG" \
  --successful-only --dagger-slice

echo "=== [$(date -u +%T)] stage 2: video -> image format"
python "$TRC/scripts/data/format_conversion/lerobot_video_to_image.py" \
  --src-root "$LR/piper_x_cubes_in_cup_$TAG" \
  --src-repo-id "local/piper_x_cubes_in_cup_$TAG" \
  --dst-root "$LR/piper_x_cubes_in_cup_${TAG}_images" \
  --repo-id "local/piper_x_cubes_in_cup_${TAG}_images"

echo "=== [$(date -u +%T)] stage 3: aggregate $BASE + $TAG -> $DST"
BASE="$BASE" DST="$DST" TAG="$TAG" LR="$LR" python - <<'EOF'
import os
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
lr, base, dst, tag = Path(os.environ["LR"]), os.environ["BASE"], os.environ["DST"], os.environ["TAG"]
src = f"piper_x_cubes_in_cup_{tag}_images"
aggregate_datasets(
    repo_ids=[f"local/{base}", f"local/{src}"],
    aggr_repo_id=f"local/{dst}",
    roots=[lr / base, lr / src],
    aggr_root=lr / dst,
)
EOF

echo "=== [$(date -u +%T)] stage 4: verify $DST"
BASE="$BASE" DST="$DST" TAG="$TAG" LR="$LR" python - <<'EOF'
import json, os, sys
from pathlib import Path
lr, base, dst, tag = Path(os.environ["LR"]), os.environ["BASE"], os.environ["DST"], os.environ["TAG"]
def info(name):
    return json.load(open(lr / name / "meta/info.json"))
d, b, s = info(dst), info(base), info(f"piper_x_cubes_in_cup_{tag}_images")
ok = d["total_episodes"] == b["total_episodes"] + s["total_episodes"]
print(f"{dst}: {d['total_episodes']} eps / {d['total_frames']} frames "
      f"(= {b['total_episodes']} base + {s['total_episodes']} {tag})")
if not ok:
    sys.exit(f"FAIL: episode counts do not add up — do NOT delete the inputs")
print("episode counts add up.")
EOF

cat <<CLEANUP

=== [$(date -u +%T)] $DST ready: $LR/$DST
Its frames now include every frame of the inputs, so once a training run has
loaded it you can reclaim them (verified above, but eyeball a sample first):
  rm -rf $LR/$BASE $LR/piper_x_cubes_in_cup_${TAG}_images
Keep $LR/piper_x_cubes_in_cup_$TAG (video format, ~10x smaller) as the
regenerable original if you want an audit trail.
CLEANUP
