"""历史记录面板：搜索、筛选、详情、删除与 CSV 导出。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.record_manager import RecordManager
from ui.components import EmptyState, make_label
from ui.record_detail_dialog import RecordDetailDialog
from ui.theme import repolish
from utils.exporters import export_records_csv
from utils.record_view import (
    dominant_behavior,
    format_timestamp,
    record_summary,
    source_kind,
    source_label,
    top_confidence,
)

SOURCE_FILTERS = [
    ("全部来源", ""),
    ("摄像头", "camera"),
    ("视频", "video"),
    ("图片", "image"),
]


class HistoryPanel(QWidget):
    record_selected = pyqtSignal(str)
    record_deleted = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        record_manager: Optional[RecordManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.record_manager = record_manager or RecordManager()
        self.all_records: List[Dict] = []
        self.visible_records: List[Dict] = []
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(280)
        self._search_timer.timeout.connect(self.reload)
        self._initialized = False
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.count_label = make_label("0 条", "caption")
        self.count_label.setProperty("chip", "true")
        repolish(self.count_label)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索意图、文件或来源…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(220)
        self.search_edit.setFixedHeight(30)
        self.search_edit.setProperty("search", "true")
        self.search_edit.textChanged.connect(lambda _text: self._search_timer.start())

        self.source_combo = QComboBox()
        self.source_combo.setFixedWidth(112)
        self.source_combo.setFixedHeight(30)
        for label, value in SOURCE_FILTERS:
            self.source_combo.addItem(label, value)
        self.source_combo.currentIndexChanged.connect(lambda _index: self.reload())

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setProperty("variant", "ghost")
        self.refresh_btn.setFixedHeight(30)
        self.refresh_btn.setToolTip("重新加载历史记录（F5）")
        self.refresh_btn.clicked.connect(self.reload)

        self.export_btn = QPushButton("导出")
        self.export_btn.setProperty("variant", "ghost")
        self.export_btn.setFixedHeight(30)
        self.export_btn.setToolTip("导出当前筛选结果为 CSV")
        self.export_btn.clicked.connect(self.export_records)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.setProperty("variant", "ghost")
        self.delete_btn.setFixedHeight(30)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip("删除选中的记录")
        self.delete_btn.clicked.connect(self.delete_selected)

        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.search_edit)
        toolbar.addWidget(self.source_combo)
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.delete_btn)

        self.table = self._build_table()
        self.empty_state = EmptyState(
            "暂无分析记录",
            "完成一次摄像头、视频或图片分析后，记录会自动出现在这里。",
        )
        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.empty_state)

        layout.addLayout(toolbar)
        layout.addWidget(self.stack, 1)

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["时间", "类型", "来源", "主要行为", "置信度", "意图摘要"]
        )
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setSortingEnabled(True)
        table.setWordWrap(False)
        table.setMinimumHeight(150)
        table.doubleClicked.connect(self._open_detail)
        table.itemSelectionChanged.connect(self._on_selection_changed)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._show_context_menu)

        header = table.horizontalHeader()
        header.setSortIndicatorShown(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)
        return table

    # ------------------------------------------------------------------ data
    def reload(self) -> None:
        keyword = self.search_edit.text().strip()
        if keyword:
            records = self.record_manager.search_records(keyword, limit=300)
        else:
            records = self.record_manager.get_all_records(limit=300)
        source_type = self.source_combo.currentData()
        if source_type:
            records = [r for r in records if r.get("source_type") == source_type]

        self.all_records = self.record_manager.get_all_records(limit=300)
        self.visible_records = records
        self._populate(records)
        total = self.record_manager.get_records_count()
        self.count_label.setText(
            f"{len(records)} 条" if len(records) == total else f"{len(records)}/{total} 条"
        )
        self._initialized = True

    def _populate(self, records: List[Dict]) -> None:
        table = self.table
        table.setSortingEnabled(False)
        table.setRowCount(len(records))
        for row, record in enumerate(records):
            timestamp = format_timestamp(record.get("timestamp"), "%Y-%m-%d %H:%M:%S")
            kind = source_kind(record)
            source = source_label(record)
            behavior = dominant_behavior(record)
            confidence = top_confidence(record)
            summary = record_summary(record, limit=80)

            time_item = QTableWidgetItem(timestamp)
            time_item.setData(Qt.ItemDataRole.UserRole, record.get("id", ""))
            values = [
                time_item,
                QTableWidgetItem(kind),
                QTableWidgetItem(source),
                QTableWidgetItem(behavior),
                QTableWidgetItem(f"{confidence:.0%}" if confidence else "—"),
                QTableWidgetItem(summary),
            ]
            for column, item in enumerate(values):
                if column == 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 2:
                    item.setToolTip(source)
                if column == 5:
                    item.setToolTip(summary)
                table.setItem(row, column, item)
            table.setRowHeight(row, 32)

        table.setSortingEnabled(True)
        table.sortItems(0, Qt.SortOrder.DescendingOrder)
        self.stack.setCurrentIndex(0 if records else 1)
        self.export_btn.setEnabled(bool(records))
        self.delete_btn.setEnabled(False)

    def _on_selection_changed(self) -> None:
        self.delete_btn.setEnabled(bool(self.table.selectionModel().selectedRows()))

    def _selected_ids(self) -> List[str]:
        ids = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item:
                record_id = item.data(Qt.ItemDataRole.UserRole)
                if record_id:
                    ids.append(record_id)
        return ids

    def _selected_records(self) -> List[Dict]:
        ids = set(self._selected_ids())
        return [record for record in self.visible_records if record.get("id") in ids]

    def _open_detail(self, index) -> None:
        item = self.table.item(index.row(), 0)
        if not item:
            return
        record_id = item.data(Qt.ItemDataRole.UserRole)
        record = self.record_manager.get_record(record_id)
        if not record:
            self.status_message.emit("记录不存在或已被删除")
            self.reload()
            return
        self.record_selected.emit(record_id)
        dialog = RecordDetailDialog(record, self.window())
        dialog.exec()
        dialog.deleteLater()

    def delete_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {len(ids)} 条记录吗？\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for record_id in ids:
            if self.record_manager.delete_record(record_id):
                self.record_deleted.emit(record_id)
        self.status_message.emit(f"已删除 {len(ids)} 条记录")
        self.reload()

    def export_records(self) -> None:
        records = self._selected_records() or self.visible_records
        if not records:
            self.status_message.emit("当前没有可导出的记录")
            return
        default_name = "pet_behavior_records.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出历史记录",
            str(Path.home() / default_name),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return
        try:
            count = export_records_csv(records, path)
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", f"无法写入文件：\n{exc}")
            return
        self.status_message.emit(f"已导出 {count} 条记录：{Path(path).name}")

    def _show_context_menu(self, position) -> None:
        from PyQt6.QtWidgets import QMenu

        index = self.table.indexAt(position)
        if not index.isValid():
            return
        if not self.table.selectionModel().isRowSelected(index.row(), index.parent()):
            self.table.selectRow(index.row())
        menu = QMenu(self)
        detail_action = menu.addAction("查看详情")
        export_action = menu.addAction("导出所选记录")
        menu.addSeparator()
        delete_action = menu.addAction("删除所选记录")
        action = menu.exec(self.table.viewport().mapToGlobal(position))
        if action == detail_action:
            self._open_detail(index)
        elif action == export_action:
            self.export_records()
        elif action == delete_action:
            self.delete_selected()

    def refresh_if_needed(self) -> None:
        self.reload()
