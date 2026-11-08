"""Binary mask state and bounded, tile-based stroke history."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Edit:
    tiles: list
    before: int
    after: int

    @property
    def nbytes(self):
        return sum(old.nbytes + new.nbytes for _, _, old, new in self.tiles)


class MaskDocument:
    tile_size = 128

    def __init__(self, mask, path="", history_limit=64 * 1024 * 1024):
        array = np.asarray(mask)
        if array.ndim != 2 or not min(array.shape) or array.dtype != np.bool_:
            raise ValueError("Expected a nonempty two-dimensional bool mask")
        self.mask = array.copy()
        self.path = path
        self.suggested_path = path
        self.history_limit = history_limit
        self.undo_stack = []
        self.redo_stack = []
        self._tiles = None
        self._state = self._saved_state = self._serial = 0
        self._saved_pixels = np.packbits(self.mask, axis=None)

    @property
    def dirty(self):
        return self._state != self._saved_state

    def mark_saved(self, path):
        self.path = path
        self.suggested_path = path
        self._saved_state = self._state
        self._saved_pixels = np.packbits(self.mask, axis=None)

    def discard_changes(self):
        """Restore the saved checkpoint even after old undo tiles are evicted."""
        self.mask[:] = np.unpackbits(self._saved_pixels, count=self.mask.size).reshape(self.mask.shape)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._tiles = None
        self._state = self._saved_state

    def mark_modified(self):
        self._serial += 1
        self._state = self._serial

    def begin_stroke(self):
        if self._tiles is not None:
            raise RuntimeError("A stroke is already active")
        self._tiles = {}

    def _capture(self, x0, y0, x1, y1):
        size = self.tile_size
        for y in range(y0 // size * size, y1, size):
            for x in range(x0 // size * size, x1, size):
                if (x, y) not in self._tiles:
                    self._tiles[x, y] = self.mask[y:y + size, x:x + size].copy()

    def paint_segment(self, start, end, diameter, value):
        """Paint a continuous round stroke in image coordinates; return dirty box."""
        if self._tiles is None:
            raise RuntimeError("Begin a stroke before painting")
        radius = max(.5, diameter / 2)
        ax, ay = start
        bx, by = end
        height, width = self.mask.shape
        x0 = max(0, int(np.floor(min(ax, bx) - radius)))
        y0 = max(0, int(np.floor(min(ay, by) - radius)))
        x1 = min(width, int(np.ceil(max(ax, bx) + radius)))
        y1 = min(height, int(np.ceil(max(ay, by) + radius)))
        if x1 <= x0 or y1 <= y0:
            return None
        self._capture(x0, y0, x1, y1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = 0 if length == 0 else np.clip(
            ((xx + .5 - ax) * dx + (yy + .5 - ay) * dy) / length, 0, 1)
        inside = (xx + .5 - ax - t * dx) ** 2 + (yy + .5 - ay - t * dy) ** 2 <= radius ** 2
        self.mask[y0:y1, x0:x1][inside] = value
        return x0, y0, x1, y1

    def end_stroke(self):
        if self._tiles is None:
            return
        tiles = []
        for (x, y), old in self._tiles.items():
            new = self.mask[y:y + old.shape[0], x:x + old.shape[1]]
            if not np.array_equal(old, new):
                tiles.append((x, y, old, new.copy()))
        self._tiles = None
        if not tiles:
            return
        self._serial += 1
        self.undo_stack.append(Edit(tiles, self._state, self._serial))
        self._state = self._serial
        self.redo_stack.clear()
        total = sum(edit.nbytes for edit in self.undo_stack)
        # Retain at least the newest operation, including a whole-image invert.
        while total > self.history_limit and len(self.undo_stack) > 1:
            total -= self.undo_stack.pop(0).nbytes

    def invert(self):
        self.begin_stroke()
        height, width = self.mask.shape
        self._capture(0, 0, width, height)
        np.logical_not(self.mask, out=self.mask)
        self.end_stroke()

    def undo(self):
        if not self.undo_stack:
            return
        edit = self.undo_stack.pop()
        for x, y, old, _ in edit.tiles:
            self.mask[y:y + old.shape[0], x:x + old.shape[1]] = old
        self._state = edit.before
        self.redo_stack.append(edit)

    def redo(self):
        if not self.redo_stack:
            return
        edit = self.redo_stack.pop()
        for x, y, _, new in edit.tiles:
            self.mask[y:y + new.shape[0], x:x + new.shape[1]] = new
        self._state = edit.after
        self.undo_stack.append(edit)
