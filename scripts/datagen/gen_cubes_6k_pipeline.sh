#!/usr/bin/env bash
# Cubes-in-cup run-3 data: 6000 attempts (~5.7k successes) -> lerobot -> image
# dataset, all local. Expected ~18.5 h datagen (benchmarked 5.4-5.6 eps/min,
# 16 workers, g6e.4xlarge) + conversions.
#
#   tmux new -d -s datagen6k 'bash scripts/datagen/gen_cubes_6k_pipeline.sh'
#
# no `conda activate`: its hook trips over set -u (NVCC_PREPEND_FLAGS unbound)
set -euo pipefail
export PATH=/home/ubuntu/miniconda3/envs/lerobot/bin:$PATH
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /home/ubuntu/trc-spaces

echo "=== [1/3] datagen: 6000 samples, 16 workers ($(date)) ==="
python -m molmo_spaces.data_generation.main PiperXCubesInCupDataGenConfig
RUN=$(ls -td experiment_output/datagen/piper_x_cubes_in_cup_v1/PiperXCubesInCupDataGenConfig/*/ | head -1)
echo "=== datagen run dir: $RUN"

echo "=== [2/3] h5 -> lerobot (successful episodes only) ($(date)) ==="
python scripts/data/format_conversion/piper_x_to_lerobot.py "$RUN" \
  --repo-id local/piper_x_cubes_in_cup_v2 \
  --root experiment_output/lerobot/piper_x_cubes_in_cup_v2 \
  --successful-only \
  --image-writer-processes 6 --image-writer-threads 4

echo "=== [3/3] video -> image (training-ready) ($(date)) ==="
python scripts/data/format_conversion/lerobot_video_to_image.py \
  --src-root experiment_output/lerobot/piper_x_cubes_in_cup_v2 \
  --src-repo-id local/piper_x_cubes_in_cup_v2 \
  --dst-root experiment_output/lerobot/piper_x_cubes_in_cup_v2_images \
  --repo-id local/piper_x_cubes_in_cup_v2_images \
  --image-writer-processes 6 --image-writer-threads 4 \
  --overwrite

echo "=== PIPELINE_DONE ($(date)) ==="
