"""实时日志面板：级别筛选、复制、导出与自动滚动。"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.components import make_label
from ui.theme import repolish, theme_manager, tokens
from utils.constants import BEHAVIOR_DISPLAY_NAMES

LOG_TYPES = {
    "info": "信息",
    "success": "成功",
    "warning": "警告",
    "error": "错误",
    "behavior": "行为",
}

FILTERS = [
    ("全部", {"info", "success", "warning", "error", "behavior"}),
    ("仅重要", {"success", "warning", "error", "behavior"}),
    ("仅警告/错误", {"warning", "error"}),
]


class LogPanel(QWidget):
    log_clicked = pyqtSignal(str, dict)
    status_message = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_log_lines = 1000
        self.entries: Deque[Tuple[str, str, str]] = deque(maxlen=self.max_log_lines)
        self._visible_count = 0
        self.init_ui()
        theme_manager.theme_changed.connect(lambda _theme: self._render())

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.count_label = make_label("0 条", "caption")
        self.count_label.setProperty("chip", "true")
        repolish(self.count_label)
        self.filter_combo = QComboBox()
        self.filter_combo.setFixedWidth(112)
        self.filter_combo.setFixedHeight(30)
        for label, _types in FILTERS:
            self.filter_combo.addItem(label)
        self.filter_combo.currentIndexChanged.connect(self._render)

        self.copy_btn = QPushButton("复制")
        self.copy_btn.setProperty("variant", "ghost")
        self.copy_btn.setFixedHeight(30)
        self.copy_btn.setToolTip("复制当前显示的日志")
        self.copy_btn.clicked.connect(self._copy_visible)

        self.export_btn = QPushButton("导出")
        self.export_btn.setProperty("variant", "ghost")
        self.export_btn.setFixedHeight(30)
        self.export_btn.setToolTip("导出完整日志为文本文件")
        self.export_btn.clicked.connect(self.export_logs)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setProperty("variant", "ghost")
        self.clear_btn.setFixedHeight(30)
        self.clear_btn.clicked.connect(self.clear_logs)

        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.filter_combo)
        toolbar.addWidget(self.copy_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.clear_btn)

        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("LogView")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1200)
        self.log_text.setPlaceholderText("分析过程中的运行信息会显示在这里…")
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout.addLayout(toolbar)
        layout.addWidget(self.log_text, 1)

    # ------------------------------------------------------------------ API
    def append_log(self, message: str, log_type: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_type = log_type if log_type in LOG_TYPES else "info"
        self.entries.append((timestamp, log_type, message))
        self.count_label.setText(f"{len(self.entries)} 条")
        if self._matches_filter(log_type):
            at_bottom = self._is_at_bottom()
            self._append_line(timestamp, log_type, message, scroll=at_bottom)
            self._visible_count += 1
            self._update_count_text()

    def append_behavior_log(self, behavior: str, confidence: float, intent: str) -> None:
        display = BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)
        self.append_log(
            f"识别到「{display}」 · 置信度 {confidence:.0%} · {intent}",
            "behavior",
        )

    def clear_logs(self) -> None:
        self.entries.clear()
        self.log_text.clear()
        self._visible_count = 0
        self.count_label.setText("0 条")

    def export_logs(self) -> None:
        if not self.entries:
            self.status_message.emit("当前没有可导出的日志")
            return
        default_name = f"pet_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出日志",
            str(Path.home() / default_name),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            lines = []
            for timestamp, log_type, message in self.entries:
                label = LOG_TYPES.get(log_type, log_type)
                lines.append(f"[{timestamp}] [{label}] {message}")
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.status_message.emit(f"日志已导出：{Path(path).name}")
        except OSError as exc:
            self.status_message.emit(f"日志导出失败：{exc}")

    # -------------------------------------------------------------- internals
    def _matches_filter(self, log_type: str) -> bool:
        index = max(0, self.filter_combo.currentIndex())
        return log_type in FILTERS[index][1]

    def _is_at_bottom(self) -> bool:
        scrollbar = self.log_text.verticalScrollBar()
        return scrollbar.value() >= scrollbar.maximum() - 4

    def _color_for(self, log_type: str) -> str:
        palette = tokens()
        return {
            "info": palette["text_secondary"],
            "success": palette["success"],
            "warning": palette["warning"],
            "error": palette["danger"],
            "behavior": palette["primary"],
        }.get(log_type, palette["text_secondary"])

    def _append_line(
        self,
        timestamp: str,
        log_type: str,
        message: str,
        scroll: bool = True,
    ) -> None:
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        time_format = QTextCharFormat()
        time_format.setForeground(QColor(tokens()["text_muted"]))
        cursor.insertText(f"{timestamp}  ", time_format)

        type_format = QTextCharFormat()
        type_format.setForeground(QColor(self._color_for(log_type)))
        type_format.setFontWeight(600)
        cursor.insertText(f"{LOG_TYPES.get(log_type, log_type):<2}  ", type_format)

        body_format = QTextCharFormat()
        body_format.setForeground(QColor(tokens()["text"] if log_type != "info" else tokens()["text_secondary"]))
        cursor.insertText(message + "\n", body_format)

        if scroll:
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _render(self) -> None:
        self.log_text.clear()
        self._visible_count = 0
        for timestamp, log_type, message in self.entries:
            if self._matches_filter(log_type):
                self._append_line(timestamp, log_type, message, scroll=False)
                self._visible_count += 1
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._update_count_text()

    def _update_count_text(self) -> None:
        total = len(self.entries)
        if self._visible_count == total:
            self.count_label.setText(f"{total} 条")
        else:
            self.count_label.setText(f"{self._visible_count}/{total} 条")

    def _copy_visible(self) -> None:
        if not self.log_text.toPlainText():
            self.status_message.emit("当前没有可复制的日志")
            return
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.log_text.toPlainText())
        self.status_message.emit("日志已复制到剪贴板")
