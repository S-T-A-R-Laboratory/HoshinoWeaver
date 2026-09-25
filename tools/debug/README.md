# Debug tools

This directory contains developer-facing diagnostic entrypoints. They call
current HoshinoWeaver production code, and sometimes private implementation
APIs, but they are not stable application interfaces.

Run them from the repository root with Python's module form so the project
package and sibling debug modules resolve consistently:

```powershell
python -m tools.debug.debug_norma_ba <image-folder> [options]
python -m tools.debug.debug_norma_ba_sampling <image-folder> [options]
python -m tools.debug.debug_fisheye_two_frame_align --reference <ref> --source <src> --output <image> [options]
python -m tools.debug.debug_manual_fisheye_pairs --pairs <csv> [options]
python -m tools.debug.debug_star_detection <image> [options]
python -m tools.debug.debug_star_shrink <image> [options]
```

`debug_norma_ba_sampling` can prepare reusable local star-detection and full
matched-edge caches before running cap, edge-set, and sampling-seed sweeps:

```powershell
python -m tools.debug.debug_norma_ba_sampling <image-folder> `
  --reference-index 0 --edge-sets "1;1,2;1,2,4" `
  --prepare-cache-only
```

`debug_norma_ba` accepts separate masks for two different purposes:
`--detection-mask` restricts star detection and matching, while
`--evaluation-mask` only restricts image-domain residual measurements.
`--star-reference-baseline` additionally solves every reference-to-frame pair
as an independent two-image alignment for direct stability and timing
comparison with BA. `--uniform-sample-fraction 0.1` retains a deterministic,
evenly spaced 10% subset (including sequence endpoints) for long-sequence
diagnostics.

The cache uses Python pickle and must only be loaded from a trusted local
workspace. File metadata, mask, camera initialization, offsets, and matching
seed participate in cache invalidation.

The scripts may change together with internal APIs. Keep reusable algorithmic
behavior in `hoshicore/` and test it there; the tracked test suite must not
import these tools. Local datasets and generated results remain outside version
control.
