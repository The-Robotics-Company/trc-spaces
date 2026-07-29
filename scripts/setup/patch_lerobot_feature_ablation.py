"""Patch installed lerobot with two env-gated ablation hooks (idempotent).

Run 2 post-mortem: the policy's arm control barely uses the cameras
(image-sensitivity ratio 0.13 — see ~/act_eval_analysis/vision_ablation.py).
Run-3 ablations attack that proprio shortcut. Neither hook changes behavior
unless its env var is set, so patched lerobot stays safe for normal use.

1. factory.make_policy: TRC_EXCLUDE_FEATURES="k1,k2,..." drops dataset features
   from the derived policy input/output features. Used to de-duplicate the
   doubled 16-dim state/action (packed key + sub-keys both stored) and, with
   observation.state also excluded, to train camera-only. Filtering here (not
   in the dataset) avoids rewriting 16 GB of parquet: extra batch keys flow
   through the dataloader but the policy/normalizer never touch them.

2. ACTPolicy.forward: TRC_STATE_DROPOUT=<p> zeroes the (already normalized,
   so 0 == mean) state vector per-sample with prob p during training only.

Like patch_lerobot_image_dataset_query.py, this is LOST on any lerobot
reinstall/upgrade — re-run it after. Marker: PATCH(trc.
"""
import re
import sys

SP = "/home/ubuntu/miniconda3/envs/lerobot/lib/python3.11/site-packages/lerobot"

PATCHES = [
    {
        "file": f"{SP}/policies/factory.py",
        "marker": "PATCH(trc: TRC_EXCLUDE_FEATURES",
        "anchor": "        features = dataset_to_policy_features(ds_meta.features)\n",
        "insert": """\
        # PATCH(trc: TRC_EXCLUDE_FEATURES): drop named features from the policy config
        import os as _os

        _excl = {s.strip() for s in _os.environ.get("TRC_EXCLUDE_FEATURES", "").split(",") if s.strip()}
        if _excl:
            features = {k: ft for k, ft in features.items() if k not in _excl}
""",
    },
    {
        "file": f"{SP}/policies/act/modeling_act.py",
        "marker": "PATCH(trc: TRC_STATE_DROPOUT",
        "anchor": '        """Run the batch through the model and compute the loss for training or validation."""\n',
        "insert": """\
        # PATCH(trc: TRC_STATE_DROPOUT): zero normalized state per-sample w.p. p, training only
        import os as _os

        _p = float(_os.environ.get("TRC_STATE_DROPOUT", "0") or 0)
        if self.training and _p > 0 and OBS_STATE in batch:
            batch = dict(batch)
            _keep = torch.rand(batch[OBS_STATE].shape[0], 1, device=batch[OBS_STATE].device) >= _p
            batch[OBS_STATE] = batch[OBS_STATE] * _keep.to(batch[OBS_STATE].dtype)
""",
    },
]

for p in PATCHES:
    src = open(p["file"]).read()
    if p["marker"] in src:
        print(f"[skip] already patched: {p['file']}")
        continue
    n = src.count(p["anchor"])
    if n != 1:
        sys.exit(f"[FAIL] anchor found {n}x (expected 1) in {p['file']}")
    src = src.replace(p["anchor"], p["anchor"] + p["insert"])
    open(p["file"], "w").write(src)
    print(f"[ok] patched {p['file']}")

print("done")
