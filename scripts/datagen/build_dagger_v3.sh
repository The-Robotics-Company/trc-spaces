#!/usr/bin/env bash
# Build the v3 training dataset = v2_images (5698 teacher eps) + DAgger round-1
# expert suffixes. Three stages, mirroring gen_cubes_6k_pipeline.sh:
#   1. dagger h5 -> lerobot video dataset (successful-only, student prefix sliced
#      off via the sidecar switch_step)
#   2. video -> image format (video decode in training is the 440ms/sample trap)
#   3. aggregate v2_images + dagger images -> piper_x_cubes_in_cup_v3_images
# Usage: build_dagger_v3.sh <dagger_datagen_run_dir>
set -euo pipefail
export PATH=$HOME/miniconda3/envs/lerobot/bin:$PATH

TRC=$HOME/trc-spaces
RUN=${1:?usage: build_dagger_v3.sh <dagger_datagen_run_dir>}
LR=$TRC/experiment_output/lerobot

echo "=== [$(date -u +%T)] stage 1: h5 -> lerobot video (dagger-sliced)"
python $TRC/scripts/data/format_conversion/piper_x_to_lerobot.py "$RUN" \
  --repo-id local/piper_x_cubes_in_cup_dagger1 \
  --root "$LR/piper_x_cubes_in_cup_dagger1" \
  --successful-only --dagger-slice

echo "=== [$(date -u +%T)] stage 2: video -> image format"
python $TRC/scripts/data/format_conversion/lerobot_video_to_image.py \
  --src-root "$LR/piper_x_cubes_in_cup_dagger1" \
  --src-repo-id local/piper_x_cubes_in_cup_dagger1 \
  --dst-root "$LR/piper_x_cubes_in_cup_dagger1_images" \
  --repo-id local/piper_x_cubes_in_cup_dagger1_images

echo "=== [$(date -u +%T)] stage 3: aggregate v2 + dagger -> v3"
python - <<'EOF'
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
lr = Path.home() / "trc-spaces/experiment_output/lerobot"
aggregate_datasets(
    repo_ids=["local/piper_x_cubes_in_cup_v2_images",
              "local/piper_x_cubes_in_cup_dagger1_images"],
    aggr_repo_id="local/piper_x_cubes_in_cup_v3_images",
    roots=[lr / "piper_x_cubes_in_cup_v2_images",
           lr / "piper_x_cubes_in_cup_dagger1_images"],
    aggr_root=lr / "piper_x_cubes_in_cup_v3_images",
)
EOF
echo "=== [$(date -u +%T)] v3 done: $LR/piper_x_cubes_in_cup_v3_images"
