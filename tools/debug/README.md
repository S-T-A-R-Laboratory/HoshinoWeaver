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

## SkySeg v2 sky segmentation

```powershell
python -m tools.debug.debug_sky_segmentation input.jpg --output .test-tmp/skyseg
```

Calls the production `services.segmentation.predict_sky()` API with the bundled
SkySeg v2 weight, a 512/768/1024 global pyramid, and selected-window refinement
on a working image capped at long edge 1536. Windows use unscaled RGB and
normalized zero padding; native-size fusion precedes thresholding.
`requirements.txt` includes ONNX Runtime; no training framework or standalone skyseg
package is used. EXIF orientation and RGB ICC conversion are applied before
inference. Outputs native-size `probability.npy`, `probability.png`,
`sky_mask.png` (255=sky), and `run.json` with actual providers and per-scale times.
The default `--provider auto` uses the service's lazy accelerator selection;
use `--provider CPUExecutionProvider` to explicitly force CPU.
Use `--strategy pyramid` to skip ROI refinement, `--working-long-edge` to change
the working cap, `--scales 512` for a single global scale experiment, `--threshold` to change the binary cutoff,
or `--provider CUDAExecutionProvider` / `DmlExecutionProvider` with a compatible
installed runtime. See [service contract](../../docs/services.md#skyseg-v2-天地分割).

This tool validates segmentation capability and exports results for inspection.
Service sharing, serialization, repeatability and cleanup are maintained by
`tests/component/test_services.py` and `tests/component/test_sky_segmentation.py`.

## Photoshop RAW experiment

```powershell
python tools/debug/photoshop_raw_poc.py "C:/photos/example.CR2" --hide
python tools/debug/photoshop_raw_poc.py --show
```

One standalone script opens RAW through Photoshop/ACR, using the adjacent
same-name XMP and default render settings. TIFF is the temporary pixel exchange;
`render_raw()` returns RGB pixels plus EXIF, XMP and the rendered TIFF's ICC.
`to_exif_data()` adapts the result to the project's `ExifData`.

Windows defaults to the local CC2018 executable. On macOS, pass the installed
`.app` path with `--photoshop`; the AppleScript/AppKit branch is experimental
and has not been run on a Mac. Platform differences stay in this script.

`--output-dir` selects a new output directory; otherwise results go under
`benchmarks/local/`. `--timeout` stops waiting without quitting PS. Hidden runs
attempt to reveal PS on failure. Use disposable RAW/XMP copies for experiments.
Dependencies are numpy, tifffile, psutil and pyexiv2 (already project dependencies).
