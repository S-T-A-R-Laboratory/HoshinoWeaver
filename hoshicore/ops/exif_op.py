import fractions
from asyncio import gather
from typing import Any, Awaitable, Mapping, Optional

from loguru import logger

from ..component.exifdata import CommonExifTags, ExifData, read_exif_data
from ..engine.registry import register_op
from .base import BaseOp, ParallelBaseOp


@register_op()
class ExifReadOp(ParallelBaseOp):
    INPUTS = {"fnames": {"type": "sequence", "description": "File names"}}
    OUTPUTS = {"result": {"type": "sequence", "description": "Exif sequence"}}
    PARALLEL_ARGS_LIST = ["fnames"]
    CONCURRENCY = 4

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]],
                                    configs: dict[str, Any]):
        fname: str = await data['fnames']
        exif = read_exif_data(fname)
        return {"result": exif}


@register_op()
class ExifReduceOp(BaseOp):
    INPUTS = {"exifs": {"type": "sequence", "description": "Exif sequence"}}
    CONFIGS: dict[str, Any] = {
        "merge_method": {
            "type": "str",
            "description": "Merge method"
        }
    }
    OUTPUTS = {"result": {"type": "exif", "description": "Exif"}}

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        merge_method = configs["merge_method"]

        tot_num = self.length
        assert tot_num is not None, "TrailStackerOp requires sequence length information."

        if merge_method == "sum":
            time_cumsum = fractions.Fraction(0)
            base_exif = None
            for i in range(tot_num):
                input_data = self._async_convert_inputs()
                cur_exif: ExifData = await input_data['exifs']
                if base_exif is None and cur_exif is not None:
                    base_exif = cur_exif

                time = cur_exif.get_exif(CommonExifTags.ExposureTime)
                if isinstance(time, str):
                    if "/" in time:
                        num, denom = time.split("/")
                        time = fractions.Fraction(int(num), int(denom))
                    else:
                        time = fractions.Fraction(int(float(time) * 100), 100)
                    time_cumsum += fractions.Fraction(time)
            if base_exif is None:
                await self._broadcast_outputs({"result": None})
                return
            base_exif.set_exif(
                CommonExifTags.ExposureTime,
                "/".join(map(str, time_cumsum.as_integer_ratio())))
            logger.info(
                f"Calculated total exposure time = {time_cumsum:.2f}s.")
            await self._broadcast_outputs({"result": base_exif})
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")
