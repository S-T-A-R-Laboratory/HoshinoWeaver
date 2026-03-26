import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hoshicore.component.queue import FileCacheQueue


class TestFileCacheQueue(unittest.IsolatedAsyncioTestCase):
    async def test_put_get_with_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=4, tot_num=1, serializer="pickle", temp_path=tmp_dir
            )
            item = {"a": 1, "b": [1, 2, 3]}

            await queue.put(item)
            file_path = queue.queue._queue[0]
            self.assertTrue(Path(file_path).exists())

            got = await queue.get()
            self.assertEqual(got, item)
            self.assertFalse(Path(file_path).exists())

    async def test_put_get_with_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=4, tot_num=1, serializer="json", temp_path=tmp_dir
            )
            item = {"name": "hnw", "ok": True, "num": 7}

            await queue.put(item)
            got = await queue.get()
            self.assertEqual(got, item)

    async def test_put_get_with_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=4, tot_num=1, serializer="numpy", temp_path=tmp_dir
            )
            item = np.array([1, 2, 3], dtype=np.int32)

            await queue.put(item)
            got = await queue.get()
            np.testing.assert_array_equal(got, item)

    async def test_unsupported_serializer_on_put(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=4, tot_num=1, serializer="bad", temp_path=tmp_dir
            )

            with self.assertRaises(ValueError):
                await queue.put({"x": 1})

    async def test_get_raises_value_error_and_removes_bad_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=4, tot_num=1, serializer="json", temp_path=tmp_dir
            )
            bad_path = Path(tmp_dir) / "bad.json"
            bad_path.write_text("not-a-json", encoding="utf-8")

            await queue.queue.put(str(bad_path))
            with self.assertRaises(ValueError):
                await queue.get()

            self.assertFalse(bad_path.exists())

    async def test_clear_removes_queue_files_and_resets_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            queue = FileCacheQueue(
                maxsize=8, tot_num=2, serializer="pickle", temp_path=tmp_dir
            )
            await queue.put({"id": 1})
            await queue.put({"id": 2})
            self.assertEqual(queue._queue_counter, 2)

            stale_file = Path(tmp_dir) / f"{queue.prefix}_stale.pkl"
            stale_file.write_bytes(json.dumps({"z": 1}).encode("utf-8"))
            self.assertTrue(stale_file.exists())

            queue.clear()
            self.assertTrue(queue.queue.empty())
            self.assertEqual(queue._queue_counter, 0)
            self.assertFalse(stale_file.exists())

            remain = list(Path(tmp_dir).glob(f"{queue.prefix}_*"))
            self.assertEqual(remain, [])


if __name__ == "__main__":
    unittest.main()
