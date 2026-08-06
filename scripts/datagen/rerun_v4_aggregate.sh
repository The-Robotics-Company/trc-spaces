#!/usr/bin/env bash
# Rerun ONLY stages 3+4 of build_dagger_v3.sh for the v4 build, after the
# 2026-08-06 ENOSPC: aggregate_datasets() routes every source parquet through
# HF_DATASETS_CACHE, and the NVMe cache disk (549G) cannot hold the v3 training
# cache (273G, mmap'd by a live training run — undeletable) plus the aggregate
# scratch (~300G). So the scratch goes on root instead, sized-checked, and is
# deleted on success. Inputs (stages 1-2) are already built and verified.
set -euo pipefail
export PATH=$HOME/miniconda3/envs/lerobot/bin:$PATH

TRC=$HOME/trc-spaces
LR=$TRC/experiment_output/lerobot
BASE=piper_x_cubes_in_cup_v3_images
SRC=piper_x_cubes_in_cup_dagger2_images
DST=piper_x_cubes_in_cup_v4_images
SCRATCH=$HOME/hf_aggregate_scratch

# deliberate override of the never-cache-on-root rule: sized, and removed below
export HF_DATASETS_CACHE=$SCRATCH

source "$TRC/../piper-x-policy/scripts/env/dataset_cache.sh" 2>/dev/null || true
# inputs must verify before touching anything
python - <<'EOF'
import json, os, sys
from pathlib import Path
lr = Path(os.path.expanduser("~/trc-spaces/experiment_output/lerobot"))
b = json.load(open(lr / "piper_x_cubes_in_cup_v3_images/meta/info.json"))
s = json.load(open(lr / "piper_x_cubes_in_cup_dagger2_images/meta/info.json"))
assert b["total_episodes"] == 8197 and s["total_episodes"] == 1523, (b, s)
print(f"inputs ok: {b['total_episodes']} base + {s['total_episodes']} dagger2")
EOF

require_free_gb "$HOME" 700 "v4 aggregate (copy ~310G + arrow scratch ~310G)"

echo "=== [$(date -u +%T)] removing partial $DST"
rm -rf "$LR/$DST"
mkdir -p "$SCRATCH"

echo "=== [$(date -u +%T)] stage 3: aggregate $BASE + dagger2 -> $DST (scratch=$SCRATCH)"
LR="$LR" python - <<'EOF'
import os
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
lr = Path(os.environ["LR"])
aggregate_datasets(
    repo_ids=["local/piper_x_cubes_in_cup_v3_images", "local/piper_x_cubes_in_cup_dagger2_images"],
    aggr_repo_id="local/piper_x_cubes_in_cup_v4_images",
    roots=[lr / "piper_x_cubes_in_cup_v3_images", lr / "piper_x_cubes_in_cup_dagger2_images"],
    aggr_root=lr / "piper_x_cubes_in_cup_v4_images",
)
EOF

echo "=== [$(date -u +%T)] stage 4: verify $DST"
LR="$LR" python - <<'EOF'
import json, os, sys
from pathlib import Path
lr = Path(os.environ["LR"])
def info(name): return json.load(open(lr / name / "meta/info.json"))
d, b, s = (info(n) for n in ("piper_x_cubes_in_cup_v4_images",
                             "piper_x_cubes_in_cup_v3_images",
                             "piper_x_cubes_in_cup_dagger2_images"))
ok = d["total_episodes"] == b["total_episodes"] + s["total_episodes"]
print(f"v4: {d['total_episodes']} eps / {d['total_frames']} frames "
      f"(= {b['total_episodes']} + {s['total_episodes']})")
if not ok:
    sys.exit("FAIL: episode counts do not add up — do NOT delete the inputs")
print("episode counts add up.")
EOF

echo "=== [$(date -u +%T)] dropping aggregate scratch"
rm -rf "$SCRATCH"
echo "V4_AGGREGATE_OK"
