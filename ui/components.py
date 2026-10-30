"""通用 UI 组件：卡片、空状态、提示浮层等。"""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QTimer, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def make_label(
    text: str = "",
    role: str | None = None,
    *,
    wrap: bool = False,
    selectable: bool = False,
    parent: QWidget | None = None,
) -> QLabel:
    label = QLabel(text, parent)
    if role:
        label.setProperty("role", role)
    label.setWordWrap(wrap)
    if selectable:
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def make_chip(text: str, tone: str | None = None, parent: QWidget | None = None) -> QLabel:
    chip = QLabel(text, parent)
    chip.setProperty("chip", "true")
    if tone:
        chip.setProperty("tone", tone)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return chip


class Card(QFrame):
    """带内边距与统一圆角的标准卡片容器。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margins: tuple[int, int, int, int] = (16, 14, 16, 14),
        spacing: int = 10,
        object_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("card", "true")
        if object_name:
            self.setObjectName(object_name)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget


class EmptyState(QWidget):
    """轻量空状态：主文案 + 辅助文案。"""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EmptyState")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label = make_label(title, "h3", parent=self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label = make_label(subtitle, "caption", wrap=True, parent=self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

    def set_text(self, title: str, subtitle: str = "") -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))


class Toast(QLabel):
    """主窗口底部居中的短暂提示，不打断操作。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setDuration(180)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        parent.installEventFilter(self)

    def show_message(self, text: str, duration_ms: int = 2600) -> None:
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._animate(0.0, 1.0)
        self._timer.start(duration_ms)

    def _fade_out(self) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._effect.opacity())
        self._animation.setEndValue(0.0)
        try:
            self._animation.finished.disconnect()
        except TypeError:
            pass
        self._animation.finished.connect(self.hide)
        self._animation.start()

    def _animate(self, start: float, end: float) -> None:
        self._animation.stop()
        try:
            self._animation.finished.disconnect()
        except TypeError:
            pass
        self._effect.setOpacity(start)
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        margin = 20
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - margin
        self.move(max(0, x), max(0, y))

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)


class Divider(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


def horizontal_row(*widgets, spacing: int = 8, stretch_last: bool = True) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            layout.addStretch(1)
        else:
            layout.addWidget(widget)
    if stretch_last:
        layout.addStretch(1)
    return layout


def tinted_dot(color_hex: str, size: int = 8) -> QLabel:
    dot = QLabel()
    dot.setFixedSize(size, size)
    dot.setStyleSheet(
        f"background-color: {QColor(color_hex).name()}; border-radius: {size // 2}px;"
    )
    return dot
