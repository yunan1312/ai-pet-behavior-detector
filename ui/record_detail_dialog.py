"""单条分析记录详情窗口。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.components import Card, Divider, make_chip, make_label
from ui.theme import theme_manager
from utils.constants import BEHAVIOR_DISPLAY_NAMES
from utils.exporters import export_record_markdown
from utils.paths import app_icon_path
from utils.record_view import (
    behavior_distribution,
    detection_count,
    dominant_behavior,
    format_duration,
    format_offset,
    format_timestamp,
    record_summary,
    source_label,
    top_confidence,
)


class RecordDetailDialog(QDialog):
    def __init__(self, record: Dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setWindowTitle("分析详情")
        self.setModal(True)
        self.resize(760, 640)
        self.setMinimumSize(640, 520)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(10)
        icon_label = QLabel()
        icon = QPixmap(app_icon_path())
        if not icon.isNull():
            icon_label.setPixmap(
                icon.scaled(
                    40,
                    40,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        titles = QVBoxLayout()
        titles.setSpacing(1)
        title = make_label("分析详情", "h1")
        subtitle = make_label(
            f"{format_timestamp(self.record.get('timestamp'))} · {source_label(self.record)}",
            "caption",
        )
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addWidget(icon_label)
        header.addLayout(titles)
        header.addStretch(1)
        header.addWidget(make_chip(f"识别 {detection_count(self.record)} 次", "primary"))
        header.addWidget(make_chip(f"时长 {format_duration(self.record.get('duration'))}"))
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("Page")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 8)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._build_summary())
        content_layout.addWidget(self._build_distribution())
        content_layout.addWidget(self._build_timeline())
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        layout.addWidget(Divider())

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        export_btn = QPushButton("导出 Markdown 报告")
        export_btn.setProperty("variant", "soft")
        export_btn.setFixedHeight(34)
        export_btn.clicked.connect(self.export_report)
        close_btn = QPushButton("关闭")
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(export_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _build_summary(self) -> QWidget:
        card = Card()
        header = QHBoxLayout()
        header.addWidget(make_label("意图解读", "h3"))
        header.addStretch(1)
        header.addWidget(make_chip(dominant_behavior(self.record), "primary"))
        header.addWidget(make_chip(f"最高置信度 {top_confidence(self.record):.0%}", "success"))
        summary = QLabel(record_summary(self.record, limit=500))
        summary.setObjectName("IntentText")
        summary.setWordWrap(True)
        summary.setMinimumHeight(48)
        card.body().addLayout(header)
        card.body().addWidget(summary)
        return card

    def _build_distribution(self) -> QWidget:
        card = Card()
        card.body().addWidget(make_label("行为分布", "h3"))
        distribution = behavior_distribution(self.record)
        if not distribution:
            empty = make_label("该记录中没有识别到有效行为。", "caption")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(48)
            card.body().addWidget(empty)
            return card

        for item in distribution[:8]:
            row = QHBoxLayout()
            row.setSpacing(10)
            name = make_label(item["display"], "body")
            name.setFixedWidth(88)
            bar = QFrame()
            bar.setFixedHeight(8)
            bar.setStyleSheet("background: rgba(128,128,128,0.22); border-radius: 4px;")
            bar_layout = QHBoxLayout(bar)
            bar_layout.setContentsMargins(0, 0, 0, 0)
            fill = QFrame()
            fill.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                " stop:0 #5B9CFF, stop:1 #A78BFA); border-radius: 4px;"
            )
            fill.setMinimumWidth(6)
            bar_layout.addWidget(fill, max(1, int(item["percent"] * 100)))
            bar_layout.addStretch(max(0, int((1 - item["percent"]) * 100)))
            meta = make_label(f"{item['count']} 次 · {item['percent']:.0%}", "caption")
            meta.setFixedWidth(80)
            meta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(meta)
            card.body().addLayout(row)
        return card

    def _build_timeline(self) -> QWidget:
        card = Card()
        count = sum(
            1
            for item in self.record.get("behaviors", []) or []
            if item.get("behavior") not in {None, "", "idle", "error"}
        )
        card.body().addWidget(make_label(f"行为时间线 · {count} 条", "h3"))

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["时间", "行为", "置信度", "意图"])
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setMinimumHeight(210)
        table.setMaximumHeight(280)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        entries = [
            item
            for item in self.record.get("behaviors", []) or []
            if item.get("behavior") not in {None, "", "idle", "error"}
        ]
        table.setRowCount(len(entries))
        for row, item in enumerate(entries):
            timestamp = format_offset(item.get("timestamp", 0))
            behavior = BEHAVIOR_DISPLAY_NAMES.get(
                item.get("behavior", ""), item.get("behavior", "—")
            )
            confidence = float(item.get("confidence", 0.0) or 0.0)
            for column, value in enumerate(
                [timestamp, behavior, f"{confidence:.0%}", item.get("intent", "")]
            ):
                cell = QTableWidgetItem(value)
                if column == 2:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, cell)
            table.setRowHeight(row, 32)

        if not entries:
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem("—"))
            table.setItem(0, 1, QTableWidgetItem("无行为明细"))
            table.setItem(0, 2, QTableWidgetItem("—"))
            table.setItem(0, 3, QTableWidgetItem("—"))
        card.body().addWidget(table)
        return card

    def export_report(self) -> None:
        default_name = (
            f"pet_record_{format_timestamp(self.record.get('timestamp'), '%Y%m%d_%H%M')}.md"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出分析报告",
            str(Path.home() / default_name),
            "Markdown 文件 (*.md);;文本文件 (*.txt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            export_record_markdown(self.record, path)
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", f"无法写入文件：\n{exc}")
            return
        QMessageBox.information(self, "导出完成", f"报告已导出：\n{path}")
