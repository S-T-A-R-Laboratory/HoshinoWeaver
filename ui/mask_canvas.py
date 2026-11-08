"""Qt binary mask canvas. Only touched pixels update the display buffer."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QTransform
from PySide6.QtWidgets import QWidget


def image_rgb(image):
    """Copy a Qt image to contiguous RGB888, respecting its row stride."""
    rgb = image.convertToFormat(QImage.Format_RGB888)
    rows = np.frombuffer(rgb.constBits(), np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
    return rows[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).copy()


class MaskCanvas(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.reference = QImage()
        self.document = None
        self.scale = 1.
        self._fit_view = True
        self.offset = QPointF()
        self.mode = "overlay"
        self.tool = "sky"
        self.diameter = 40
        self.screen_brush = False
        self.opacity = .45
        self._drawing = False
        self._panning = False
        self._pan_button = None
        self._cursor = None
        self._display = None
        self._overlay = self._binary = QImage()

    def set_document(self, document, reference=None):
        self.finish_stroke()
        self.document = document
        if reference is not None:
            self.reference = reference
        if document is None:
            self._display = None
            self._overlay = self._binary = QImage()
        else:
            self._display = document.mask.astype(np.uint8)
            height, width = document.mask.shape
            self._overlay = QImage(self._display.data, width, height,
                                   self._display.strides[0], QImage.Format_Indexed8)
            self._overlay.setColorTable([QColor(0, 0, 0, 0).rgba(), QColor(40, 220, 120).rgba()])
            self._binary = QImage(self._display.data, width, height,
                                  self._display.strides[0], QImage.Format_Indexed8)
            self._binary.setColorTable([QColor("black").rgba(), QColor("white").rgba()])
        self.fit()

    def fit(self):
        self._fit_view = True
        if self.document is not None:
            height, width = self.document.mask.shape
            self.scale = min(self.width() / width, self.height() / height)
            self.offset = QPointF((self.width() - width * self.scale) / 2,
                                  (self.height() - height * self.scale) / 2)
        self.update()

    def image_point(self, point):
        return (point - self.offset) / self.scale

    @property
    def effective_diameter(self):
        return self.diameter / self.scale if self.screen_brush else self.diameter

    def sync(self, box=None):
        if self.document is None:
            return
        if box is None:
            self._display[:] = self.document.mask
            self.update()
        else:
            x0, y0, x1, y1 = box
            self._display[y0:y1, x0:x1] = self.document.mask[y0:y1, x0:x1]
            top = QPointF(x0, y0) * self.scale + self.offset
            self.update(QRectF(top.x(), top.y(), (x1 - x0) * self.scale,
                               (y1 - y0) * self.scale).toAlignedRect().adjusted(-2, -2, 2, 2))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 38))
        if self.document is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignCenter, "加载蒙版，或使用当前展示图生成蒙版")
            return
        height, width = self.document.mask.shape
        painter.setTransform(QTransform(self.scale, 0, 0, self.scale,
                                        self.offset.x(), self.offset.y()))
        target = QRectF(0, 0, width, height)
        if self.mode != "mask" and not self.reference.isNull():
            painter.drawImage(target, self.reference)
        if self.mode == "mask" or self.reference.isNull():
            painter.drawImage(target, self._binary)
        elif self.mode == "overlay":
            painter.setOpacity(self.opacity)
            painter.drawImage(target, self._overlay)
            painter.setOpacity(1.)
        if self._cursor is not None and not self._panning and self.isEnabled():
            painter.setPen(QPen(QColor("white"), 1 / self.scale))
            painter.drawEllipse(self.image_point(self._cursor), self.effective_diameter / 2,
                                self.effective_diameter / 2)

    def wheelEvent(self, event):
        if self.document is None or self._drawing:
            return
        point = self.image_point(event.position())
        self._fit_view = False
        self.scale = min(32., max(.005, self.scale * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15)))
        self.offset = event.position() - point * self.scale
        self.update()

    def mousePressEvent(self, event):
        if self.document is None:
            return
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.finish_stroke()
            self._fit_view = False
            self._panning = True
            self._pan_button = event.button()
            self._last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self._panning:
                return
            point = self.image_point(event.position())
            height, width = self.document.mask.shape
            if not (0 <= point.x() < width and 0 <= point.y() < height):
                return
            self._drawing = True
            self._last_image = point
            self.document.begin_stroke()
            self._paint(point)

    def _paint(self, point):
        box = self.document.paint_segment(
            (self._last_image.x(), self._last_image.y()), (point.x(), point.y()),
            self.effective_diameter, self.tool == "sky")
        self._last_image = point
        if box:
            self.sync(box)

    def mouseMoveEvent(self, event):
        self._cursor = event.position()
        if self._panning:
            self.offset += event.position() - self._last
            self._last = event.position()
        elif self._drawing:
            self._paint(self.image_point(event.position()))
        self.update()

    def finish_stroke(self):
        if self._drawing:
            self.document.end_stroke()
            self._drawing = False
            self.changed.emit()
        self._panning = False
        self._pan_button = None
        self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.LeftButton:
            self._paint(self.image_point(event.position()))
            self.finish_stroke()
        elif self._panning and event.button() == self._pan_button:
            self.finish_stroke()

    def leaveEvent(self, event):
        self._cursor = None
        self.update()

    def resizeEvent(self, event):
        if self._fit_view:
            self.fit()
        elif self.document is not None and event.oldSize().isValid():
            self.offset += QPointF((event.size().width() - event.oldSize().width()) / 2,
                                    (event.size().height() - event.oldSize().height()) / 2)
        super().resizeEvent(event)
