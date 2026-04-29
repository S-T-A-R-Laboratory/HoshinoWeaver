from pathlib import Path

import numpy as np
import pytest

from hoshicore.component.queue import FileCacheQueue

pytestmark = pytest.mark.asyncio


class TestFileCacheQueue:
    async def test_put_get_with_pickle(self, tmp_path):
        queue = FileCacheQueue(maxsize=4, serializer="pickle", temp_path=tmp_path)
        item = {"a": 1, "b": [1, 2, 3]}

        await queue.put(item)
        file_path = queue.queue._queue[0]
        assert Path(file_path).exists()

        got = await queue.get()
        assert got == item
        assert not Path(file_path).exists()

    async def test_put_get_with_json(self, tmp_path):
        queue = FileCacheQueue(maxsize=4, serializer="json", temp_path=tmp_path)
        item = {"name": "hnw", "ok": True, "num": 7}

        await queue.put(item)
        got = await queue.get()
        assert got == item

    async def test_put_get_with_numpy(self, tmp_path):
        queue = FileCacheQueue(maxsize=4, serializer="numpy", temp_path=tmp_path)
        item = np.array([1, 2, 3], dtype=np.int32)

        await queue.put(item)
        got = await queue.get()
        np.testing.assert_array_equal(got, item)

    async def test_unsupported_serializer_on_put(self, tmp_path):
        queue = FileCacheQueue(maxsize=4, serializer="bad", temp_path=tmp_path)
        with pytest.raises(ValueError):
            await queue.put({"x": 1})

    async def test_get_raises_value_error_and_removes_bad_file(self, tmp_path):
        queue = FileCacheQueue(maxsize=4, serializer="json", temp_path=tmp_path)
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not-a-json", encoding="utf-8")

        await queue.queue.put(str(bad_path))
        with pytest.raises(ValueError):
            await queue.get()

        assert not bad_path.exists()

    async def test_clear_removes_queue_files_and_resets_counter(self, tmp_path):
        queue = FileCacheQueue(maxsize=8, serializer="pickle", temp_path=tmp_path)
        await queue.put({"id": 1})
        await queue.put({"id": 2})
        assert queue._queue_counter == 2

        stale_file = tmp_path / f"{queue.prefix}_stale.pkl"
        stale_file.write_bytes(b"dummy")
        assert stale_file.exists()

        queue.clear()
        assert queue.queue.empty()
        assert queue._queue_counter == 0
        assert not stale_file.exists()

        remain = list(tmp_path.glob(f"{queue.prefix}_*"))
        assert remain == []
