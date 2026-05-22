import numpy as np
import pytest

from hoshicore.component.frame_buffer import (
    BaseFrameBuffer,
    DiskFrameBuffer,
    MemoryFrameBuffer,
)


class TestDiskFrameBuffer:
    def test_append_and_getitem(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame1 = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        frame2 = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
        buf.append(frame1, weight=0.5)
        buf.append(frame2)
        assert len(buf) == 2

        got1, w1 = buf[0]
        np.testing.assert_array_equal(got1, frame1)
        assert w1 == pytest.approx(0.5)

        got2, w2 = buf[1]
        np.testing.assert_array_equal(got2, frame2)
        assert w2 is None

        buf.cleanup()

    def test_index_out_of_range(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        with pytest.raises(IndexError):
            buf[0]
        buf.cleanup()

    def test_cleanup_removes_files(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))
        npy_files_before = list(tmp_path.glob("*.npy"))
        assert len(npy_files_before) == 2

        buf.cleanup()
        npy_files_after = list(tmp_path.glob("*.npy"))
        assert len(npy_files_after) == 0

    def test_weight_ndarray(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        weight = np.full((4, 4, 3), 0.8, dtype=np.float32)
        buf.append(frame, weight=weight)

        got_frame, got_weight = buf[0]
        np.testing.assert_array_equal(got_frame, frame)
        np.testing.assert_allclose(got_weight, weight)
        buf.cleanup()


class TestRefCounting:
    def test_single_consumer_cleanup(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))
        buf.cleanup()
        assert list(tmp_path.glob("*.npy")) == []

    def test_two_consumers_first_cleanup_keeps_files(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()  # consumer 1
        buf.acquire()  # consumer 2
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))

        buf.cleanup()  # consumer 1 done
        assert len(list(tmp_path.glob("*.npy"))) == 1  # files still exist

        buf.cleanup()  # consumer 2 done
        assert list(tmp_path.glob("*.npy")) == []  # now cleaned up

    def test_three_consumers(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        buf.acquire()
        buf.acquire()
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))

        buf.cleanup()
        assert len(list(tmp_path.glob("*.npy"))) == 1
        buf.cleanup()
        assert len(list(tmp_path.glob("*.npy"))) == 1
        buf.cleanup()
        assert list(tmp_path.glob("*.npy")) == []

    def test_exception_safety(self, tmp_path):
        """If one consumer hits an exception and calls cleanup, the other still works."""
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()  # consumer A
        buf.acquire()  # consumer B
        frame = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
        buf.append(frame)

        # Consumer A fails and calls cleanup
        buf.cleanup()  # ref 2→1

        # Consumer B can still read
        got, _ = buf[0]
        np.testing.assert_array_equal(got, frame)
        del got  # release mmap before cleanup — Windows holds a file lock until mmap is freed

        # Consumer B finishes
        buf.cleanup()  # ref 1→0 → files deleted
        assert list(tmp_path.glob("*.npy")) == []


class TestMemoryFrameBuffer:
    def test_append_and_getitem(self):
        buf = MemoryFrameBuffer()
        buf.acquire()
        frame1 = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        frame2 = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
        buf.append(frame1, weight=0.5)
        buf.append(frame2)
        assert len(buf) == 2

        got1, w1 = buf[0]
        np.testing.assert_array_equal(got1, frame1)
        assert w1 == pytest.approx(0.5)

        got2, w2 = buf[1]
        np.testing.assert_array_equal(got2, frame2)
        assert w2 is None
        buf.cleanup()

    def test_index_out_of_range(self):
        buf = MemoryFrameBuffer()
        buf.acquire()
        with pytest.raises(IndexError):
            buf[0]
        buf.cleanup()

    def test_cleanup_clears_data(self):
        buf = MemoryFrameBuffer()
        buf.acquire()
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))
        buf.append(np.zeros((2, 2, 1), dtype=np.uint8))
        assert len(buf) == 2
        buf.cleanup()
        assert len(buf) == 0

    def test_weight_ndarray(self):
        buf = MemoryFrameBuffer()
        buf.acquire()
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        weight = np.full((4, 4, 3), 0.8, dtype=np.float32)
        buf.append(frame, weight=weight)

        got_frame, got_weight = buf[0]
        np.testing.assert_array_equal(got_frame, frame)
        np.testing.assert_allclose(got_weight, weight)
        buf.cleanup()


class TestGetRows:
    def test_disk_get_rows_returns_correct_slice(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame = np.arange(60, dtype=np.uint8).reshape(5, 4, 3)
        buf.append(frame)
        chunk, weight = buf.get_rows(0, 1, 3)
        np.testing.assert_array_equal(chunk, frame[1:3])
        assert weight is None
        buf.cleanup()

    def test_disk_get_rows_returns_copy_not_view(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        buf.append(frame)
        chunk, _ = buf.get_rows(0, 0, 2)
        chunk[:] = 99
        # original frame on disk should be unaffected
        reloaded, _ = buf[0]
        assert reloaded[0, 0, 0] == 0
        buf.cleanup()

    def test_disk_get_rows_scalar_weight(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        buf.append(frame, weight=0.7)
        _, weight = buf.get_rows(0, 0, 2)
        assert weight == pytest.approx(0.7)
        buf.cleanup()

    def test_disk_get_rows_ndarray_weight_sliced(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        weight = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        buf.append(frame, weight=weight)
        _, got_weight = buf.get_rows(0, 1, 3)
        np.testing.assert_array_equal(got_weight, weight[1:3])
        buf.cleanup()

    def test_memory_get_rows_returns_correct_slice(self):
        buf = MemoryFrameBuffer()
        buf.acquire()
        frame = np.arange(60, dtype=np.uint8).reshape(5, 4, 3)
        buf.append(frame)
        chunk, weight = buf.get_rows(0, 2, 4)
        np.testing.assert_array_equal(chunk, frame[2:4])
        assert weight is None
        buf.cleanup()

    def test_base_get_rows_index_error(self, tmp_path):
        buf = DiskFrameBuffer(temp_path=tmp_path)
        buf.acquire()
        with pytest.raises(IndexError):
            buf.get_rows(0, 0, 2)
        buf.cleanup()
