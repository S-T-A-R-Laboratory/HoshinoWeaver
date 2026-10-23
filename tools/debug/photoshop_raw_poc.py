"""Local Photoshop/ACR smoke test; retains artifacts and leaves PS running."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hoshicore.component.photoshop import photoshop_windows, render_raw, to_exif_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',
                        nargs='?',
                        type=Path,
                        help='RAW path; adjacent XMP is handled by ACR')
    parser.add_argument('--photoshop',
                        type=Path,
                        default=None,
                        help='Windows .exe or macOS .app path')
    parser.add_argument('--output-dir',
                        type=Path,
                        help='New directory for retained artifacts')
    parser.add_argument('--timeout', type=float, default=180)
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        '--hide',
        action='store_true',
        help='Hide PS UI while rendering (desktop session required)')
    window.add_argument('--show',
                        action='store_true',
                        help='Reveal the running PS UI, then exit')
    args = parser.parse_args()
    try:
        if args.show:
            if args.raw is not None:
                parser.error('Do not supply raw with --show')
            print(
                json.dumps(photoshop_windows(args.photoshop, show=True),
                           ensure_ascii=False,
                           indent=2))
            return
        if args.raw is None:
            parser.error('raw is required for rendering')
        if args.output_dir is None:
            args.output_dir = (
                Path(__file__).resolve().parents[2] / 'benchmarks' / 'local' /
                ('photoshop-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
        pixels, metadata = render_raw(args.raw,
                                      photoshop=args.photoshop,
                                      output_dir=args.output_dir,
                                      timeout=args.timeout,
                                      hide=args.hide)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(
        json.dumps(
            {
                key: value
                for key, value in metadata.items()
                if key not in ('xmp', 'icc_profile', 'exif', 'photoshop_exif',
                               'windows_after')
            },
            ensure_ascii=False,
            indent=2))
    print(f'Pixels returned: {pixels.shape}, {pixels.dtype}; '
          f'EXIF entries: {len(metadata.get("exif", []))}; '
          f'XMP characters: {len(metadata["xmp"])}')


if __name__ == '__main__':
    main()
