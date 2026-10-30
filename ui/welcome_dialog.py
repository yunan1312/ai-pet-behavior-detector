"""欢迎页：首次启动引导，可记住“不再显示”。"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.components import Card, make_chip, make_label
from utils.constants import APP_VERSION
from utils.paths import app_icon_path


class WelcomeDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("欢迎使用")
        self.setModal(True)
        self.resize(760, 560)
        self.setMinimumSize(680, 520)
        self.init_ui()

    @property
    def dont_show_again(self) -> bool:
        return self.dont_show_checkbox.isChecked()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        # 品牌头部
        hero = QFrame()
        hero.setObjectName("InfoHero")
        hero.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        hero_layout.setSpacing(16)

        icon_label = QLabel()
        pixmap = QPixmap(app_icon_path())
        if not pixmap.isNull():
            icon_label.setPixmap(
                pixmap.scaled(
                    64,
                    64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        icon_label.setFixedSize(64, 64)

        titles = QVBoxLayout()
        titles.setSpacing(3)
        title = QLabel("让 AI 读懂你的宠物")
        title.setObjectName("HeroBehavior")
        subtitle = QLabel(
            "结合姿态估计与行为识别，实时解读猫狗的动作与情绪，并生成可回顾的分析记录。"
        )
        subtitle.setObjectName("HeroConfidence")
        subtitle.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        titles.addStretch(1)
        hero_layout.addWidget(icon_label)
        hero_layout.addLayout(titles, 1)
        layout.addWidget(hero)

        # 功能卡片
        features = QGridLayout()
        features.setHorizontalSpacing(10)
        features.setVerticalSpacing(10)
        feature_data = [
            ("实时识别", "连接摄像头，画面内直接标注关键点与行为，语音同步播报。"),
            ("视频与图片", "拖入文件即可分析，支持 MP4 / AVI / MOV 与 JPG / PNG / BMP。"),
            ("记录与报告", "自动保存分析历史，可搜索、查看详情并导出 CSV / Markdown。"),
        ]
        for column, (heading, description) in enumerate(feature_data):
            card = Card(margins=(14, 12, 14, 12), spacing=4)
            card.body().addWidget(make_label(heading, "h3"))
            body = make_label(description, "caption", wrap=True)
            card.body().addWidget(body)
            features.addWidget(card, 0, column)
        layout.addLayout(features)

        # 快速开始
        steps_card = Card(margins=(18, 14, 18, 14), spacing=7)
        steps_card.body().addWidget(make_label("三步开始", "h3"))
        steps = [
            ("1", "选择摄像头并点击「开始识别」，或拖入视频 / 图片"),
            ("2", "在右侧查看当前行为、置信度和意图解读"),
            ("3", "在底部「历史记录」中搜索、查看详情或导出报告"),
        ]
        for number, text in steps:
            row = QHBoxLayout()
            row.setSpacing(10)
            badge = QLabel(number)
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                "background: rgba(91,156,255,0.18); color: #5B9CFF;"
                "border-radius: 11px; font-weight: 700; font-size: 11px;"
            )
            label = make_label(text, "subtitle", wrap=True)
            row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(label, 1)
            steps_card.body().addLayout(row)
        layout.addWidget(steps_card)

        caps_card = Card(margins=(18, 12, 18, 12), spacing=8)
        caps_card.body().addWidget(make_label("可识别的行为", "h3"))
        caps = QGridLayout()
        caps.setHorizontalSpacing(6)
        caps.setVerticalSpacing(6)
        behaviors = [
            "站立", "坐着", "趴卧", "睡觉", "行走", "奔跑", "伸懒腰",
            "理毛", "踩奶", "蹭物体", "捕猎", "玩耍", "发呆",
        ]
        for index, name in enumerate(behaviors):
            caps.addWidget(make_chip(name), index // 7, index % 7)
        caps_card.body().addLayout(caps)
        layout.addWidget(caps_card)
        layout.addStretch(1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.dont_show_checkbox = QCheckBox("以后启动时不再显示此页面")
        self.dont_show_checkbox.setChecked(True)
        version = make_label(f"v{APP_VERSION}", "caption")
        start_btn = QPushButton("开始使用")
        start_btn.setProperty("variant", "primary")
        start_btn.setFixedHeight(38)
        start_btn.setMinimumWidth(130)
        start_btn.setDefault(True)
        start_btn.clicked.connect(self.accept)

        footer.addWidget(self.dont_show_checkbox)
        footer.addStretch(1)
        footer.addWidget(version)
        footer.addWidget(start_btn)
        layout.addLayout(footer)
