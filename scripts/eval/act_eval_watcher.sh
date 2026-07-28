#!/usr/bin/env bash
# Watch a LeRobot training run's checkpoints dir; sim-eval every NEW checkpoint
# and append success rate + rollout video to the run's companion W&B eval run.
# The trc-spaces port of piper-x-policy's scripts/eval_watcher.sh.
#
#   bash scripts/eval/act_eval_watcher.sh <train_run_dir> [episodes]
#
# <train_run_dir> is the lerobot output dir containing checkpoints/<step>/pretrained_model.
# Each finished checkpoint is evaluated once (ledger: <train_run_dir>/eval_done.txt);
# results also append to <train_run_dir>/eval_log.csv. Ctrl-C to stop.
#
# Env: EVAL_ARGS       extra flags for act_checkpoint_eval.py (e.g. --workers 8)
#      WANDB_PROJECT   W&B project (default piper-x-act)
#      EVAL_RUN_ID     override the W&B eval run id. Needed when RE-evaluating a
#                      training run (e.g. after an eval-bridge fix): W&B drops
#                      points whose step is below the run's max, so replaying
#                      10k..100k into the existing run would silently lose them.
set -u
TRAIN_DIR=${1:?usage: act_eval_watcher.sh <train_run_dir> [episodes]}
EPISODES=${2:-16}
CKPTS="$TRAIN_DIR/checkpoints"
DONE="$TRAIN_DIR/eval_done.txt"
CSV="$TRAIN_DIR/eval_log.csv"
PROJECT=${WANDB_PROJECT:-piper-x-act}
# one W&B eval run per training run (fixed id + resume=allow appends);
# never share ids across trainings — W&B drops points with step < run max step
RUN_ID=${EVAL_RUN_ID:-"eval-$(basename "$(cd "$TRAIN_DIR" && pwd)" | tr '.:' '--' | tr -cd 'a-zA-Z0-9_-')"}
cd "$(dirname "$0")/../.."

echo "[watcher] train dir: $TRAIN_DIR  episodes/eval: $EPISODES  wandb: $PROJECT/$RUN_ID"
touch "$DONE"
[ -f "$CSV" ] || echo "time,step,success_rate,successes,episodes" > "$CSV"

while true; do
  if [ -d "$CKPTS" ]; then
    for d in $(ls "$CKPTS" 2>/dev/null | sort -n); do
      case "$d" in *[!0-9]*) continue ;; esac  # skip lerobot's 'last' symlink
      PM="$CKPTS/$d/pretrained_model"
      [ -f "$PM/model.safetensors" ] || continue
      grep -qx "$d" "$DONE" && continue
      # wait until the writer is done (size stable for 10 s)
      s1=$(stat -c %s "$PM/model.safetensors"); sleep 10
      s2=$(stat -c %s "$PM/model.safetensors")
      [ "$s1" != "$s2" ] && continue
      echo "[watcher] evaluating checkpoint $d"
      OUT=$(python scripts/eval/act_checkpoint_eval.py --ckpt "$PM" \
              --episodes "$EPISODES" --wandb-project "$PROJECT" \
              --wandb-run-id "$RUN_ID" ${EVAL_ARGS:-} 2>&1 | tee /dev/stderr | grep '^EVAL_RESULT' || true)
      if [ -n "$OUT" ]; then
        step=$(sed -n 's/.*step=\([0-9None]*\).*/\1/p' <<<"$OUT")
        rate=$(sed -n 's/.*rate=\([0-9.]*\).*/\1/p' <<<"$OUT")
        succ=$(sed -n 's/.*successes=\([0-9]*\).*/\1/p' <<<"$OUT")
        eps=$(sed -n 's/.*episodes=\([0-9]*\).*/\1/p' <<<"$OUT")
        echo "$(date +%F_%T),${step},${rate},${succ},${eps}" >> "$CSV"
        echo "$d" >> "$DONE"
        echo "[watcher] ckpt $d: rate=$rate ($succ/$eps)"
      else
        echo "[watcher] ckpt $d: eval FAILED (will retry next poll)"
      fi
    done
  fi
  sleep 60
done
