"""主操作栏：分析控制器 + 摄像头选择 + 实时状态。"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.components import Card, make_label
from ui.theme import repolish


class ControlPanel(QWidget):
    start_recognition_clicked = pyqtSignal()
    stop_recognition_clicked = pyqtSignal()
    import_video_clicked = pyqtSignal()
    import_image_clicked = pyqtSignal()
    open_settings_clicked = pyqtSignal()
    view_history_clicked = pyqtSignal()
    snapshot_clicked = pyqtSignal()
    cancel_requested = pyqtSignal()
    camera_changed = pyqtSignal(int)
    refresh_cameras_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.is_recognizing = False
        self._busy = False
        self._enabled = True
        self._cameras_available = False
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sheet = Card(self, margins=(16, 12, 16, 12), spacing=0)
        outer.addWidget(sheet)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.start_btn = QPushButton("开始识别")
        self.start_btn.setProperty("variant", "success")
        self.start_btn.setMinimumWidth(112)
        self.start_btn.setFixedHeight(36)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setToolTip("启动摄像头实时分析（Ctrl+R）")
        self.start_btn.clicked.connect(self.on_start_clicked)

        self.stop_btn = QPushButton("停止识别")
        self.stop_btn.setProperty("variant", "danger")
        self.stop_btn.setMinimumWidth(96)
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("结束实时分析并保存记录（Ctrl+T）")
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        self.cancel_btn = QPushButton("取消分析")
        self.cancel_btn.setProperty("variant", "soft")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedWidth(1)
        separator.setStyleSheet("background: transparent; border: none;")

        self.video_btn = QPushButton("导入视频")
        self.video_btn.setFixedHeight(36)
        self.video_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.video_btn.setToolTip("分析本地视频文件（Ctrl+O）")
        self.video_btn.clicked.connect(self.import_video_clicked.emit)

        self.image_btn = QPushButton("导入图片")
        self.image_btn.setFixedHeight(36)
        self.image_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_btn.setToolTip("分析单张宠物图片（Ctrl+I）")
        self.image_btn.clicked.connect(self.import_image_clicked.emit)

        self.snapshot_btn = QPushButton("保存画面")
        self.snapshot_btn.setProperty("variant", "ghost")
        self.snapshot_btn.setFixedHeight(36)
        self.snapshot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.snapshot_btn.setToolTip("将当前画面保存为图片（Ctrl+S）")
        self.snapshot_btn.clicked.connect(self.snapshot_clicked.emit)

        self.busy_label = make_label("正在分析…", "warning")
        self.busy_label.setProperty("chip", "true")
        self.busy_label.setProperty("tone", "warning")
        self.busy_label.setVisible(False)

        camera_label = make_label("摄像头", "caption")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(150)
        self.camera_combo.setFixedHeight(32)
        self.camera_combo.setToolTip("选择用于实时分析的摄像头")
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.refresh_cameras_btn = QPushButton("刷新")
        self.refresh_cameras_btn.setProperty("variant", "ghost")
        self.refresh_cameras_btn.setFixedHeight(32)
        self.refresh_cameras_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_cameras_btn.setToolTip("重新检测可用摄像头")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)

        self.fps_label = make_label("FPS --", "caption")
        self.fps_label.setObjectName("StatusChip")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setMinimumWidth(66)

        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(separator)
        row.addWidget(self.video_btn)
        row.addWidget(self.image_btn)
        row.addWidget(self.snapshot_btn)
        row.addSpacing(6)
        row.addWidget(self.busy_label)
        row.addStretch(1)

        sheet.body().addLayout(row)
        sheet.body().addSpacing(8)

        context_row = QHBoxLayout()
        context_row.setSpacing(8)
        context_row.addWidget(camera_label)
        context_row.addWidget(self.camera_combo)
        context_row.addWidget(self.refresh_cameras_btn)
        context_row.addStretch(1)
        self.ready_hint = make_label("分析结果会自动保存至历史记录", "caption")
        context_row.addWidget(self.ready_hint)
        context_row.addWidget(self.fps_label)
        sheet.body().addLayout(context_row)

        self.set_scanning(True)
        self.snapshot_btn.setEnabled(False)

    # ------------------------------------------------------------- behavior
    def set_scanning(self, scanning: bool) -> None:
        self.refresh_cameras_btn.setEnabled(not scanning)
        if scanning:
            self.camera_combo.blockSignals(True)
            self.camera_combo.clear()
            self.camera_combo.addItem("正在检测…", -1)
            self.camera_combo.blockSignals(False)
            self._cameras_available = False
            self._update_start_enabled()
        else:
            self.refresh_cameras_btn.setEnabled(True)

    def set_cameras(self, cameras: Iterable[int]) -> None:
        cameras = list(cameras)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if cameras:
            for index in cameras:
                self.camera_combo.addItem(f"摄像头 {index}", index)
            self._cameras_available = True
        else:
            self.camera_combo.addItem("未检测到摄像头", -1)
            self._cameras_available = False
        self.camera_combo.blockSignals(False)
        self.refresh_cameras_btn.setEnabled(True)
        self._update_start_enabled()

    def refresh_cameras(self) -> None:
        self.set_scanning(True)
        self.refresh_cameras_requested.emit()

    def get_selected_camera(self) -> int:
        data = self.camera_combo.currentData()
        return int(data) if data is not None else -1

    def _on_camera_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self.camera_combo.currentData()
        if data is not None and int(data) >= 0:
            self.camera_changed.emit(int(data))

    def on_start_clicked(self) -> None:
        if not self._enabled or self._busy:
            return
        self.is_recognizing = True
        self._sync_action_state()
        self.start_recognition_clicked.emit()

    def on_stop_clicked(self) -> None:
        if not self.is_recognizing:
            return
        self.is_recognizing = False
        self._sync_action_state()
        self.stop_recognition_clicked.emit()

    def set_recognizing(self, recognizing: bool) -> None:
        self.is_recognizing = recognizing
        self._sync_action_state()

    def set_busy(self, busy: bool, text: str = "正在分析…", cancelable: bool = True) -> None:
        self._busy = busy
        self.busy_label.setText(text)
        self.busy_label.setVisible(busy)
        self.cancel_btn.setVisible(busy and cancelable)
        self._sync_action_state()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._sync_action_state()

    def set_snapshot_enabled(self, enabled: bool) -> None:
        self.snapshot_btn.setEnabled(enabled and self._enabled and not self._busy)

    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS {fps:.1f}" if fps > 0 else "FPS --")

    def _update_start_enabled(self) -> None:
        if not hasattr(self, "start_btn"):
            return
        self._sync_action_state()

    def _sync_action_state(self) -> None:
        idle = self._enabled and not self._busy
        self.start_btn.setEnabled(
            idle and not self.is_recognizing and self._cameras_available
        )
        self.stop_btn.setEnabled(idle and self.is_recognizing)
        self.cancel_btn.setEnabled(self._busy)
        self.video_btn.setEnabled(idle and not self.is_recognizing)
        self.image_btn.setEnabled(idle and not self.is_recognizing)
        self.snapshot_btn.setEnabled(idle and self.snapshot_btn.property("hasFrame") is True)
        self.camera_combo.setEnabled(idle and not self.is_recognizing)
        self.refresh_cameras_btn.setEnabled(idle and not self.is_recognizing)
        if not self._cameras_available and not self.is_recognizing:
            self.start_btn.setToolTip("未检测到可用摄像头，请连接后点击「刷新」")
        else:
            self.start_btn.setToolTip("启动摄像头实时分析（Ctrl+R）")
        repolish(self.start_btn)
        repolish(self.stop_btn)
        repolish(self.cancel_btn)

    def set_frame_available(self, available: bool) -> None:
        self.snapshot_btn.setProperty("hasFrame", available)
        self.snapshot_btn.setEnabled(
            available and self._enabled and not self._busy
        )
