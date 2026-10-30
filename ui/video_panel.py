"""视频画布：自绘画面、关键点、空状态与进度覆盖层。

使用 QPainter 统一绘制，避免 OpenCV/PIL 的字体与缩放问题，
窗口缩放时保持比例且无需重新生成占位图。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ui.theme import theme_manager, tokens
from utils.constants import (
    BEHAVIOR_DISPLAY_NAMES,
    KEYPOINT_CONNECTIONS,
)
from utils.paths import app_icon_path


def _font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont("Microsoft YaHei UI")
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


KEYPOINT_COLORS = {
    "nose": "#F8FAFC",
    "left_eye": "#38BDF8",
    "right_eye": "#38BDF8",
    "left_ear": "#FBBF24",
    "right_ear": "#FBBF24",
    "withers": "#F87171",
    "tail_base": "#C084FC",
    "left_front_shoulder": "#34D399",
    "right_front_shoulder": "#34D399",
    "left_front_elbow": "#34D399",
    "right_front_elbow": "#34D399",
    "left_front_paw": "#A78BFA",
    "right_front_paw": "#A78BFA",
    "left_back_shoulder": "#F472B6",
    "right_back_shoulder": "#F472B6",
    "left_back_elbow": "#F472B6",
    "right_back_elbow": "#F472B6",
    "left_back_paw": "#FB923C",
    "right_back_paw": "#FB923C",
}

# AP-10K 模型额外输出的背线/尾根关键点连接。
# 仅用于界面绘制，不加入 utils.constants.KEYPOINT_CONNECTIONS，
# 以免改变 81 维姿态特征向量。
EXTRA_KEYPOINT_CONNECTIONS = [
    ("nose", "withers"),
    ("withers", "tail_base"),
    ("withers", "left_front_shoulder"),
    ("withers", "right_front_shoulder"),
    ("tail_base", "left_back_shoulder"),
    ("tail_base", "right_back_shoulder"),
]


class VideoPanel(QWidget):
    """视频展示控件，保持原有对外 API。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_frame: Optional[np.ndarray] = None
        self.keypoints: Optional[Dict[str, Tuple[float, float, float]]] = None
        self.current_behavior = ""
        self.current_confidence = 0.0
        self.show_keypoints = True

        self._pixmap: Optional[QPixmap] = None
        self._backdrop: Optional[QPixmap] = None
        self._backdrop_key: Optional[Tuple[int, int]] = None
        self._backdrop_age = 0
        self._mode_text = ""
        self._mode_live = False
        self._status_text = ""
        self._progress: Optional[int] = None
        self._progress_text = ""
        self._no_animal = False
        self._icon = QPixmap(app_icon_path()) if app_icon_path() else QPixmap()

        self.setMinimumSize(520, 230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(
            "支持拖拽导入：MP4 / AVI / MOV / JPG / PNG / BMP\n"
            "快捷键：Ctrl+O 导入视频，Ctrl+I 导入图片"
        )
        theme_manager.theme_changed.connect(self._on_theme_changed)

    # ------------------------------------------------------------------ API
    def update_frame(self, frame: np.ndarray) -> None:
        self.update_with_keypoints(frame, {}, "", 0.0)

    def update_with_keypoints(
        self,
        frame: np.ndarray,
        keypoints: Dict,
        behavior: str = "",
        confidence: float = 0.0,
    ) -> None:
        if frame is None:
            return
        self.current_frame = frame.copy()
        self.keypoints = keypoints or {}
        self.current_behavior = behavior or ""
        self.current_confidence = confidence or 0.0
        self._no_animal = not bool(self.keypoints)
        self._pixmap = self._to_pixmap(frame)
        self._backdrop_age += 1
        self.update()

    def set_mode(self, text: str, live: bool = False) -> None:
        self._mode_text = text
        self._mode_live = live
        self.update()

    def set_status_text(self, text: str) -> None:
        self._status_text = text
        self.update()

    def set_progress(self, value: Optional[int], text: str = "") -> None:
        self._progress = None if value is None else max(0, min(100, int(value)))
        self._progress_text = text
        self.update()

    def set_no_animal_hint(self, visible: bool) -> None:
        self._no_animal = visible
        self.update()

    def clear(self) -> None:
        self.current_frame = None
        self.keypoints = None
        self.current_behavior = ""
        self.current_confidence = 0.0
        self._pixmap = None
        self._backdrop = None
        self._backdrop_key = None
        self._backdrop_age = 0
        self._no_animal = False
        self._status_text = ""
        self._progress = None
        self._progress_text = ""
        self.update()

    def has_frame(self) -> bool:
        return self.current_frame is not None

    # -------------------------------------------------------------- internals
    def _on_theme_changed(self, _theme: str) -> None:
        self.update()

    @staticmethod
    def _to_pixmap(frame: np.ndarray) -> QPixmap:
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            QImage.Format.Format_BGR888,
        ).copy()
        return QPixmap.fromImage(image)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        p = tokens()
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 14.0, 14.0)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(rect, QColor(p["video_bg"]))

        if self._pixmap is not None and not self._pixmap.isNull():
            self._paint_frame(painter, rect)
        else:
            self._paint_empty_state(painter, rect)

        painter.restore()

        border = QColor(p["border"])
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 14.0, 14.0)

        # 顶部覆盖层在圆角之外也保持裁剪效果
        painter.save()
        painter.setClipPath(clip)
        self._paint_top_badges(painter, rect)
        if self._no_animal and self._pixmap is not None:
            self._paint_hint_pill(painter, rect, "未检测到宠物 · 请将猫/狗完整置于画面中")
        self._paint_progress(painter, rect)
        painter.restore()

    def _paint_frame(self, painter: QPainter, rect: QRectF) -> None:
        pixmap = self._pixmap
        if pixmap is None:
            return
        source_w = pixmap.width()
        source_h = pixmap.height()
        if source_w <= 0 or source_h <= 0:
            return
        area = rect.adjusted(12.0, 12.0, -12.0, -12.0)
        scale = min(area.width() / source_w, area.height() / source_h)
        draw_w = source_w * scale
        draw_h = source_h * scale
        target = QRectF(
            area.center().x() - draw_w / 2,
            area.center().y() - draw_h / 2,
            draw_w,
            draw_h,
        )

        letterboxed = (
            area.width() - draw_w > 6.0 or area.height() - draw_h > 6.0
        )
        if letterboxed:
            self._paint_backdrop(painter, rect)

        frame_path = QPainterPath()
        frame_path.addRoundedRect(target, 10.0, 10.0)
        painter.save()
        painter.setClipPath(frame_path)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        if self.show_keypoints and self.keypoints:
            self._paint_keypoints(painter, target, source_w, source_h)
        painter.restore()

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
        painter.drawRoundedRect(
            target.adjusted(0.5, 0.5, -0.5, -0.5), 10.0, 10.0
        )

    def _paint_backdrop(self, painter: QPainter, rect: QRectF) -> None:
        """竖屏或小画幅时用模糊放大的同帧铺底，避免大面积纯黑。"""
        pixmap = self._pixmap
        if pixmap is None:
            return
        key = (pixmap.width(), pixmap.height())
        if (
            self._backdrop is None
            or self._backdrop_key != key
            or self._backdrop_age % 15 == 0
        ):
            small_w = 56
            small_h = max(1, round(56 * pixmap.height() / pixmap.width()))
            self._backdrop = pixmap.scaled(
                small_w,
                small_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._backdrop_key = key
            self._backdrop_age = 1

        cover = self._backdrop
        ratio = max(
            rect.width() / cover.width(), rect.height() / cover.height()
        )
        width = cover.width() * ratio
        height = cover.height() * ratio
        painter.drawPixmap(
            QRectF(
                rect.center().x() - width / 2,
                rect.center().y() - height / 2,
                width,
                height,
            ),
            cover,
            QRectF(cover.rect()),
        )
        painter.fillRect(rect, QColor(6, 10, 17, 170))

    def _paint_keypoints(
        self,
        painter: QPainter,
        target: QRectF,
        source_w: int,
        source_h: int,
    ) -> None:
        if not self.keypoints:
            return
        scale_x = target.width() / source_w
        scale_y = target.height() / source_h

        def mapped(name: str):
            data = self.keypoints.get(name) if self.keypoints else None
            if not data:
                return None
            x, y, conf = data
            if conf <= 0.3:
                return None
            return target.x() + x * scale_x, target.y() + y * scale_y, conf

        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.6))
        for start, end in KEYPOINT_CONNECTIONS + EXTRA_KEYPOINT_CONNECTIONS:
            a = mapped(start)
            b = mapped(end)
            if a and b:
                painter.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        for name in KEYPOINT_COLORS:
            point = mapped(name)
            if not point:
                continue
            x, y, _conf = point
            color = QColor(KEYPOINT_COLORS.get(name, "#FFFFFF"))
            painter.setPen(QPen(QColor(6, 10, 17, 200), 1.4))
            painter.setBrush(color)
            painter.drawEllipse(QRectF(x - 4.2, y - 4.2, 8.4, 8.4))

        self._paint_detection_box(painter, target, source_w, source_h)

    def _paint_detection_box(
        self,
        painter: QPainter,
        target: QRectF,
        source_w: int,
        source_h: int,
    ) -> None:
        if not self.keypoints:
            return
        valid = [(x, y) for x, y, conf in self.keypoints.values() if conf > 0.3]
        if len(valid) < 3:
            return
        xs = [point[0] for point in valid]
        ys = [point[1] for point in valid]
        scale_x = target.width() / source_w
        scale_y = target.height() / source_h
        x1 = target.x() + (min(xs) - 18) * scale_x
        y1 = target.y() + (min(ys) - 18) * scale_y
        x2 = target.x() + (max(xs) + 18) * scale_x
        y2 = target.y() + (max(ys) + 18) * scale_y
        box = QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)).intersected(target)

        accent = QColor("#38BDF8")
        pen = QPen(accent, 1.8)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(box, 8.0, 8.0)

        label = "宠物"
        if self.current_behavior and self.current_behavior not in {"idle", "error"}:
            label = BEHAVIOR_DISPLAY_NAMES.get(self.current_behavior, self.current_behavior)
        font = _font(11, QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label) + 14
        text_h = metrics.height() + 4
        label_y = max(target.y(), box.y() - text_h - 3)
        label_rect = QRectF(box.x(), label_y, text_w, text_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(label_rect, 5.0, 5.0)
        painter.setPen(QColor("#06121E"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_top_badges(self, painter: QPainter, rect: QRectF) -> None:
        if self._mode_text:
            font = _font(11, QFont.Weight.DemiBold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(self._mode_text)
            chip = QRectF(rect.x() + 16, rect.y() + 14, text_w + 34, 26)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(6, 10, 17, 190))
            painter.drawRoundedRect(chip, 8.0, 8.0)
            dot_color = QColor("#F87171") if self._mode_live else QColor("#38BDF8")
            painter.setBrush(dot_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(chip.x() + 11, chip.center().y() - 3.5, 7, 7))
            painter.setPen(QColor("#E8F0FF"))
            painter.drawText(
                chip.adjusted(24, 0, -8, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._mode_text,
            )

        if (
            self._pixmap is not None
            and self.current_behavior
            and self.current_behavior not in {"idle", "error"}
        ):
            display = BEHAVIOR_DISPLAY_NAMES.get(self.current_behavior, self.current_behavior)
            text = f"{display} · {self.current_confidence:.0%}"
            font = _font(11, QFont.Weight.DemiBold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(text)
            chip = QRectF(
                rect.right() - text_w - 32,
                rect.y() + 14,
                text_w + 18,
                26,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(37, 99, 235, 205))
            painter.drawRoundedRect(chip, 8.0, 8.0)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_hint_pill(self, painter: QPainter, rect: QRectF, text: str) -> None:
        font = _font(12, QFont.Weight.Medium)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text)
        pill = QRectF(
            rect.center().x() - text_w / 2 - 16,
            rect.bottom() - 54,
            text_w + 32,
            30,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 12, 20, 200))
        painter.drawRoundedRect(pill, 15.0, 15.0)
        painter.setPen(QColor("#FBBF24"))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_progress(self, painter: QPainter, rect: QRectF) -> None:
        if self._progress is None:
            return
        bar = QRectF(rect.x() + 18, rect.bottom() - 18, rect.width() - 36, 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 40))
        painter.drawRoundedRect(bar, 3.0, 3.0)
        fill_width = max(0.0, bar.width() * self._progress / 100.0)
        if fill_width > 0:
            painter.setBrush(QColor("#5B9CFF"))
            painter.drawRoundedRect(
                QRectF(bar.x(), bar.y(), max(6.0, fill_width), bar.height()),
                3.0,
                3.0,
            )
        if self._progress_text:
            font = _font(10)
            painter.setFont(font)
            painter.setPen(QColor("#A9B6C8"))
            painter.drawText(
                QRectF(rect.x() + 18, bar.y() - 22, rect.width() - 36, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._progress_text,
            )

    def _paint_empty_state(self, painter: QPainter, rect: QRectF) -> None:
        p = tokens()
        # 背景网格
        painter.setPen(QPen(QColor(p["video_grid"]), 1.0))
        step = 56
        x = rect.x() + step
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = rect.y() + step
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

        # 扫描角标
        corner_pen = QPen(QColor(91, 156, 255, 110), 2.0)
        painter.setPen(corner_pen)
        inset = 30.0
        length = 34.0
        left, top = rect.left() + inset, rect.top() + inset
        right, bottom = rect.right() - inset, rect.bottom() - inset
        for x, y, dx, dy in (
            (left, top, 1, 1),
            (right, top, -1, 1),
            (left, bottom, 1, -1),
            (right, bottom, -1, -1),
        ):
            painter.drawLine(QPointF(x, y), QPointF(x + dx * length, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + dy * length))

        center_x = rect.center().x()
        compact = rect.height() < 260
        content_top = rect.center().y() - (78 if compact else 120)
        if not self._icon.isNull():
            icon_size = 44 if compact else 62
            icon_rect = QRectF(center_x - icon_size / 2, content_top, icon_size, icon_size)
            painter.setOpacity(0.92)
            painter.drawPixmap(icon_rect.toRect(), self._icon)
            painter.setOpacity(1.0)
        else:
            content_top += 20

        painter.setFont(_font(18 if compact else 21, QFont.Weight.Bold))
        painter.setPen(QColor(p["video_text"]))
        painter.drawText(
            QRectF(rect.x() + 30, content_top + 56, rect.width() - 60, 30),
            Qt.AlignmentFlag.AlignCenter,
            "等待视频源",
        )

        subtitle = self._status_text or "将视频 / 图片拖入窗口，或使用下方操作开始分析"
        painter.setFont(_font(11 if compact else 12))
        painter.setPen(QColor(p["video_muted"]))
        painter.drawText(
            QRectF(rect.x() + 30, content_top + (88 if compact else 116), rect.width() - 60, 24),
            Qt.AlignmentFlag.AlignCenter,
            subtitle,
        )

        if compact:
            return

        # 格式标签
        chips = ["MP4 · AVI · MOV", "JPG · PNG · BMP"]
        font = _font(10, QFont.Weight.Medium)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding = 12
        gap = 10
        widths = [metrics.horizontalAdvance(text) + padding * 2 for text in chips]
        total = sum(widths) + gap
        start_x = center_x - total / 2
        chip_y = content_top + 158
        for text, width in zip(chips, widths):
            chip = QRectF(start_x, chip_y, width, 24)
            painter.setPen(QPen(QColor(p["video_grid"]), 1.0))
            painter.setBrush(QColor(p["video_surface"]))
            painter.drawRoundedRect(chip, 12.0, 12.0)
            painter.setPen(QColor(p["video_muted"]))
            painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
            start_x += width + gap

        painter.setFont(_font(10))
        painter.setPen(QColor(143, 163, 191, 190))
        painter.drawText(
            QRectF(rect.x() + 30, rect.bottom() - 52, rect.width() - 60, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Ctrl+O 导入视频    Ctrl+I 导入图片",
        )
