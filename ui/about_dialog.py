"""关于窗口。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.components import Card, make_chip, make_label
from utils.constants import APP_NAME, APP_NAME_EN, APP_VERSION
from utils.paths import DATA_DIR, app_icon_path, open_path


class AboutDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setModal(True)
        self.setFixedSize(480, 520)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(14)
        icon = QLabel()
        pixmap = QPixmap(app_icon_path())
        if not pixmap.isNull():
            icon.setPixmap(
                pixmap.scaled(
                    64,
                    64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        icon.setFixedSize(64, 64)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = make_label(APP_NAME, "h1")
        version = make_label(f"版本 {APP_VERSION} · {APP_NAME_EN}", "caption")
        titles.addWidget(title)
        titles.addWidget(version)
        titles.addStretch(1)
        header.addWidget(icon)
        header.addLayout(titles, 1)
        layout.addLayout(header)

        intro = make_label(
            "结合计算机视觉与语音交互，实时解读猫狗的行为、情绪与可能的需求。"
            "所有分析在本地完成，历史记录保存在本机。",
            "subtitle",
            wrap=True,
        )
        layout.addWidget(intro)

        tech_card = Card()
        tech_card.body().addWidget(make_label("技术栈", "h3"))
        chips_row = QHBoxLayout()
        chips_row.setSpacing(6)
        for text in ["RTMPose-M AP-10K", "YOLOv8n", "PyQt6", "SQLite", "Edge TTS"]:
            chips_row.addWidget(make_chip(text))
        chips_row.addStretch(1)
        tech_card.body().addLayout(chips_row)
        layout.addWidget(tech_card)

        features_card = Card()
        features_card.body().addWidget(make_label("能力范围", "h3"))
        features_card.body().addWidget(
            make_label(
                "13 种行为识别 · 实时摄像头分析 · 视频与图片分析\n"
                "关键点可视化 · 意图与情绪解读 · 语音播报\n"
                "历史搜索与详情 · CSV / Markdown 报告导出",
                "subtitle",
                wrap=True,
            )
        )
        layout.addWidget(features_card)
        layout.addStretch(1)

        data_row = QHBoxLayout()
        display_path = str(DATA_DIR)
        home = str(Path.home())
        if display_path.startswith(home):
            display_path = "~" + display_path[len(home):]
        data_path = make_label(display_path, "caption", wrap=True)
        data_path.setToolTip(str(DATA_DIR))
        data_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_data_btn = QPushButton("打开数据目录")
        open_data_btn.setProperty("variant", "ghost")
        open_data_btn.setFixedHeight(30)
        open_data_btn.clicked.connect(lambda: open_path(DATA_DIR))
        data_row.addWidget(data_path, 1)
        data_row.addWidget(open_data_btn)
        layout.addLayout(data_row)

        footer = QHBoxLayout()
        copyright_label = make_label("© 2024-2026 PetBehaviorAnalyzer", "caption")
        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "primary")
        close_btn.setFixedHeight(34)
        close_btn.setMinimumWidth(96)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(copyright_label)
        footer.addStretch(1)
        footer.addWidget(close_btn)
        layout.addLayout(footer)
