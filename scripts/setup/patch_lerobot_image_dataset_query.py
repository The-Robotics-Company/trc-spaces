#!/usr/bin/env python
"""Patch lerobot's delta_timestamps query for image-format datasets.

lerobot 0.4.4 `LeRobotDataset._query_hf_dataset` falls back to row-first
parquet access when querying delta columns (e.g. the 100-step ACT action
chunk). On image-format datasets each row carries the embedded PNG bytes of
every camera, so reading 100 action vectors decodes ~200 images
(~440 ms/sample, dataloader-bound at GPU ~0%). Video-format datasets are
unaffected (rows hold no image bytes).

The patch caches each non-video delta column once as a single tensor
(action column: 16 MB, ~0.3 s) and slices it (~0.04 ms). Outputs are
bit-identical; measured on tao-ohio 2026-07-28: 442 -> 5.3 ms/sample,
ACT bs64 training 0.7 -> 8.8 steps/s, GPU 0 -> 94%.

Idempotent: safe to re-run. Must be re-applied after any lerobot
reinstall/upgrade (check whether upstream has fixed it first).

Usage: python scripts/setup/patch_lerobot_image_dataset_query.py
"""

import shutil
import sys

import lerobot.datasets.lerobot_dataset as m

MARKER = "PATCH(trc): image-dataset delta column cache"

OLD = """            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result"""

NEW = f"""            # {MARKER}: row-first fallback decodes embedded images on
            # image-format datasets (~440ms/sample). Cache small non-image
            # columns as tensors once and slice directly.
            if not hasattr(self, "_delta_column_cache"):
                self._delta_column_cache = {{}}
            if key not in self._delta_column_cache:
                try:
                    import numpy as _np
                    _col = self.hf_dataset.with_format("numpy", columns=[key])[:][key]
                    self._delta_column_cache[key] = torch.from_numpy(_np.ascontiguousarray(_col))
                except Exception:
                    self._delta_column_cache[key] = None
            _cache = self._delta_column_cache[key]
            if _cache is not None:
                result[key] = _cache[relative_indices]
                continue
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result"""


def main() -> int:
    path = m.__file__
    src = open(path).read()
    if "PATCH(trc" in src:  # also matches the hand-applied 2026-07-28 patch
        print(f"already patched: {path}")
        return 0
    if src.count(OLD) != 1:
        print(
            f"ERROR: expected exactly 1 match in {path}, found {src.count(OLD)}.\n"
            "lerobot version changed — check whether upstream fixed the row-first "
            "fallback before adapting this patch.",
            file=sys.stderr,
        )
        return 1
    shutil.copy(path, path + ".orig_query_backup")
    open(path, "w").write(src.replace(OLD, NEW))
    print(f"patched: {path}\nbackup:  {path}.orig_query_backup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
