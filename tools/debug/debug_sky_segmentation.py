"""Run the production SkySeg service and save native-size sky masks."""
import argparse
import asyncio
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from hoshicore.services import create_service_registry


async def main(args):
    with Image.open(args.image) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        profile = source.info.get('icc_profile')
        if profile and source.mode in ('RGB', 'RGBA'):
            rgb = ImageCms.profileToProfile(
                rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile('sRGB'), outputMode='RGB')
        pixels = np.asarray(rgb).copy()
    settings = {'segmentation_providers': [args.provider]}
    if args.model:
        settings['segmentation_model_path'] = str(args.model)
    async with create_service_registry().scope(settings) as services:
        prediction = await services.segmentation.predict_sky(
            pixels, scales=args.scales, threshold=args.threshold, color_order='RGB')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'probability.npy', prediction.probability)
    for name, image in [('sky_mask.png', prediction.mask * 255),
                        ('probability.png', np.rint(prediction.probability * 255).astype(np.uint8))]:
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError(f'Cannot encode {name}')
        encoded.tofile(args.output / name)
    report = json.dumps(prediction.info, ensure_ascii=False, indent=2)
    (args.output / 'run.json').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--provider', default='CPUExecutionProvider',
                        choices=['CPUExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider'])
    parser.add_argument('--scales', type=int, nargs='+', default=[512, 768, 1024])
    parser.add_argument('--threshold', type=float, default=.5)
    asyncio.run(main(parser.parse_args()))
