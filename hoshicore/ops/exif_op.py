import fractions
from asyncio import gather
from typing import Any, Awaitable, Mapping, Optional

from loguru import logger

from ..component.exif import CommonExifTags, ExifData, read_exif_data
from ..component.queue import StreamExhausted
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

        if merge_method == "sum":
            time_cumsum = fractions.Fraction(0)
            base_exif = None
            for i in self._input_range():
                input_data = self._async_convert_inputs()
                try:
                    cur_exif: ExifData = await input_data['exifs']
                except StreamExhausted:
                    break
                if cur_exif is None:
                    continue
                if base_exif is None:
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
            if float(time_cumsum) != 0:
                base_exif.set_exif(
                    CommonExifTags.ExposureTime,
                    "/".join(map(str, time_cumsum.as_integer_ratio())))
            logger.info(
                f"Calculated total exposure time = {float(time_cumsum):.2f}s.")
            await self._broadcast_outputs({"result": base_exif})
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")
