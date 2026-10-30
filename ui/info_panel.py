"""右侧洞察面板：当前行为、意图解读、最近行为与会话统计。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.components import Card, make_label
from ui.theme import repolish
from utils.constants import (
    BEHAVIOR_DISPLAY_NAMES,
    BEHAVIOR_TO_EMOTION,
    EMOTION_DISPLAY_NAMES,
)
from utils.paths import app_icon_path

EMOTION_COLORS = {
    "happy": "#22C55E",
    "excited": "#F59E0B",
    "curious": "#38BDF8",
    "affectionate": "#F472B6",
    "relaxed": "#34D399",
    "playful": "#A78BFA",
    "hungry": "#FB923C",
    "sleepy": "#818CF8",
    "nervous": "#FACC15",
    "agitated": "#F87171",
    "neutral": "#94A3B8",
}


class InfoPanel(QScrollArea):
    speech_toggled = pyqtSignal(bool)
    history_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_behavior = ""
        self.current_confidence = 0.0
        self.current_intent = ""
        self.recent_behaviors: List[Tuple[str, float, str]] = []
        self.max_recent = 3
        self._speech_enabled = True
        self._last_recent_key: Optional[str] = None
        self._last_recent_time = 0.0
        self._recent_rows: List[QWidget] = []
        self.init_ui()

    def init_ui(self) -> None:
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(318)
        self.setMaximumWidth(392)
        self.setObjectName("InfoScroll")

        container = QWidget()
        container.setObjectName("InfoContainer")
        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 2, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_hero())
        layout.addWidget(self._build_intent_card())
        layout.addWidget(self._build_recent_card())
        layout.addStretch(1)

        self.setWidget(container)

    # ------------------------------------------------------------------ UI
    def _build_hero(self) -> QWidget:
        hero = QFrame()
        hero.setObjectName("InfoHero")
        hero.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hero.setMinimumHeight(156)
        layout = QVBoxLayout(hero)
        layout.setContentsMargins(18, 12, 18, 10)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        state_label = make_label("当前识别", "caption")
        state_label.setObjectName("HeroTitle")
        self.state_label = make_label("等待视频源", "caption")
        self.state_label.setObjectName("HeroConfidence")
        top_row.addWidget(state_label)
        top_row.addStretch(1)
        top_row.addWidget(self.state_label)

        self.behavior_name_label = QLabel("等待识别")
        self.behavior_name_label.setObjectName("HeroBehavior")
        self.behavior_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.behavior_name_label.setWordWrap(True)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setObjectName("HeroProgress")
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.confidence_label = make_label("置信度 --", "caption")
        self.confidence_label.setObjectName("HeroConfidence")
        self.emotion_label = make_label("", "caption")
        self.emotion_label.setObjectName("HeroEmotion")
        self.emotion_label.setVisible(False)
        bottom_row.addWidget(self.confidence_label)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.emotion_label)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(0)
        self.stat_frames = self._make_stat("--", "分析帧")
        self.stat_detections = self._make_stat("0", "识别次数")
        self.stat_duration = self._make_stat("00:00", "时长")
        for index, stat in enumerate(
            (self.stat_frames, self.stat_detections, self.stat_duration)
        ):
            if index:
                stats_row.addWidget(self._hero_divider())
            stats_row.addWidget(stat[0], 1)

        layout.addLayout(top_row)
        layout.addWidget(self.behavior_name_label)
        layout.addWidget(self.confidence_bar)
        layout.addLayout(bottom_row)
        layout.addSpacing(3)
        divider = QFrame()
        divider.setObjectName("HeroDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        layout.addLayout(stats_row)
        return hero

    @staticmethod
    def _hero_divider() -> QFrame:
        line = QFrame()
        line.setObjectName("HeroDivider")
        line.setFixedWidth(1)
        line.setFixedHeight(26)
        return line

    def _build_intent_card(self) -> QWidget:
        card = Card(margins=(16, 12, 16, 12), spacing=8)
        header = QHBoxLayout()
        header.setSpacing(8)
        title = make_label("意图解读", "h3")
        self.speech_btn = QPushButton()
        self.speech_btn.setProperty("variant", "ghost")
        self.speech_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.speech_btn.setFixedHeight(26)
        self.speech_btn.clicked.connect(self._on_speech_clicked)
        self._update_speech_button()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.speech_btn)

        self.intent_label = QLabel("识别到行为后，将在这里解读宠物的情绪与需求。")
        self.intent_label.setObjectName("IntentText")
        self.intent_label.setWordWrap(True)
        self.intent_label.setMinimumHeight(52)
        self.intent_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.intent_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.emotion_detail_label = make_label("", "caption")
        self.emotion_detail_label.setVisible(False)

        card.body().addLayout(header)
        card.body().addWidget(self.intent_label)
        card.body().addWidget(self.emotion_detail_label)
        return card

    def _build_recent_card(self) -> QWidget:
        card = Card(margins=(16, 12, 16, 12), spacing=8)
        header = QHBoxLayout()
        header.setSpacing(8)
        title = make_label("最近行为", "h3")
        self.recent_count = make_label("0", "caption")
        self.recent_count.setProperty("chip", "true")
        self.recent_all_btn = QPushButton("查看全部")
        self.recent_all_btn.setProperty("variant", "ghost")
        self.recent_all_btn.setFixedHeight(24)
        self.recent_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recent_all_btn.setToolTip("在底部「历史记录」中查看完整记录")
        self.recent_all_btn.clicked.connect(self.history_requested.emit)
        header.addWidget(title)
        header.addWidget(self.recent_count)
        header.addStretch(1)
        header.addWidget(self.recent_all_btn)

        self.recent_container = QWidget()
        self.recent_layout = QVBoxLayout(self.recent_container)
        self.recent_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_layout.setSpacing(4)

        self.history_placeholder = make_label("暂无识别记录", "caption")
        self.history_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_placeholder.setMinimumHeight(44)

        card.body().addLayout(header)
        card.body().addWidget(self.recent_container)
        card.body().addWidget(self.history_placeholder)
        return card

    @staticmethod
    def _make_stat(value: str, caption: str) -> Tuple[QWidget, QLabel]:
        cell = QWidget()
        cell.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(0)
        value_label = QLabel(value)
        value_label.setObjectName("HeroStatValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        caption_label = QLabel(caption)
        caption_label.setObjectName("HeroStatCaption")
        caption_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return cell, value_label

    # -------------------------------------------------------------- behavior
    def update_behavior(
        self,
        behavior: str,
        confidence: float,
        intent: str,
        emotion: str = "",
        state_text: Optional[str] = None,
    ) -> None:
        self.current_behavior = behavior or ""
        self.current_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        self.current_intent = intent or ""

        valid = self.current_behavior not in {"", "idle", "error"}
        if valid:
            display = BEHAVIOR_DISPLAY_NAMES.get(self.current_behavior, self.current_behavior)
            self.behavior_name_label.setText(display)
        else:
            self.behavior_name_label.setText("等待识别")

        self.confidence_bar.setValue(int(self.current_confidence * 100))
        self.confidence_label.setText(
            f"置信度 {self.current_confidence:.0%}" if valid else "置信度 --"
        )
        self.state_label.setText(
            state_text or ("实时识别中" if valid else "等待视频源")
        )

        behavior_emotion = emotion or BEHAVIOR_TO_EMOTION.get(self.current_behavior, "neutral")
        emotion_name = EMOTION_DISPLAY_NAMES.get(behavior_emotion, behavior_emotion)
        if valid and emotion_name:
            self.emotion_label.setText(f"情绪 · {emotion_name}")
            self.emotion_label.setVisible(True)
            self.emotion_detail_label.setText(
                f"情绪判断：{emotion_name}    |    识别时间：{datetime.now().strftime('%H:%M:%S')}"
            )
            self.emotion_detail_label.setVisible(True)
        else:
            self.emotion_label.setVisible(False)
            self.emotion_detail_label.setVisible(False)

        if self.current_intent and valid:
            self.intent_label.setText(self.current_intent)
        else:
            self.intent_label.setText("识别到行为后，将在这里解读宠物的情绪与需求。")

        if valid:
            self._add_to_recent(self.current_behavior, self.current_confidence, self.current_intent)

    def _add_to_recent(self, behavior: str, confidence: float, intent: str) -> None:
        now = time.monotonic()
        key = behavior
        if key == self._last_recent_key and (now - self._last_recent_time) < 3.0:
            # 同一行为的高频刷新不重复刷屏
            return
        self._last_recent_key = key
        self._last_recent_time = now

        self.history_placeholder.setVisible(False)
        display = BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)
        emotion = BEHAVIOR_TO_EMOTION.get(behavior, "neutral")
        color = EMOTION_COLORS.get(emotion, "#94A3B8")

        row = QFrame()
        row.setProperty("recent", "true")
        row.setMinimumHeight(44)
        row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 5, 10, 5)
        row_layout.setSpacing(8)

        dot = QLabel()
        dot.setFixedSize(7, 7)
        dot.setStyleSheet(f"background:{color}; border-radius:3px;")

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(1)
        name_label = QLabel(display)
        name_label.setProperty("role", "h3")
        intent_short = intent.strip()
        if len(intent_short) > 20:
            intent_short = intent_short[:19] + "…"
        detail = QLabel(intent_short or "—")
        detail.setProperty("role", "caption")
        detail.setWordWrap(False)
        detail.setToolTip(intent.strip())
        row.setToolTip(intent.strip() or display)
        text_box.addWidget(name_label)
        text_box.addWidget(detail)

        meta = QVBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(1)
        conf_label = QLabel(f"{confidence:.0%}")
        conf_label.setProperty("role", "caption")
        conf_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setProperty("role", "caption")
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        meta.addWidget(conf_label)
        meta.addWidget(time_label)

        row_layout.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop)
        row_layout.addLayout(text_box, 1)
        row_layout.addLayout(meta)

        self.recent_behaviors.insert(0, (behavior, confidence, intent))
        self._recent_rows.insert(0, row)
        self.recent_layout.insertWidget(0, row)

        while len(self._recent_rows) > self.max_recent:
            old = self._recent_rows.pop()
            self.recent_layout.removeWidget(old)
            old.deleteLater()
        while len(self.recent_behaviors) > self.max_recent:
            self.recent_behaviors.pop()
        self.recent_count.setText(str(len(self.recent_behaviors)))

    def update_session_stats(
        self,
        frames: int = 0,
        detections: int = 0,
        duration: float = 0.0,
    ) -> None:
        self.stat_frames[1].setText(f"{frames:,}" if frames else "--")
        self.stat_detections[1].setText(str(detections))
        total = int(duration)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            text = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            text = f"{minutes:02d}:{seconds:02d}"
        self.stat_duration[1].setText(text)

    def set_session_state(self, text: str) -> None:
        self.state_label.setText(text or "等待视频源")

    def set_speech_enabled(self, enabled: bool) -> None:
        self._speech_enabled = bool(enabled)
        self._update_speech_button()

    def _update_speech_button(self) -> None:
        self.speech_btn.setText("语音 开" if self._speech_enabled else "语音 关")
        self.speech_btn.setToolTip(
            "关闭后不再进行语音播报（Ctrl+M）"
            if self._speech_enabled
            else "开启语音播报（Ctrl+M）"
        )
        self.speech_btn.setProperty(
            "variant", "soft" if self._speech_enabled else "ghost"
        )
        repolish(self.speech_btn)

    def _on_speech_clicked(self) -> None:
        self._speech_enabled = not self._speech_enabled
        self._update_speech_button()
        self.speech_toggled.emit(self._speech_enabled)

    def clear(self) -> None:
        self.current_behavior = ""
        self.current_confidence = 0.0
        self.current_intent = ""
        self.behavior_name_label.setText("等待识别")
        self.confidence_bar.setValue(0)
        self.confidence_label.setText("置信度 --")
        self.state_label.setText("等待视频源")
        self.emotion_label.setVisible(False)
        self.emotion_detail_label.setVisible(False)
        self.intent_label.setText("识别到行为后，将在这里解读宠物的情绪与需求。")

        self._last_recent_key = None
        self._last_recent_time = 0.0
        self.recent_behaviors.clear()
        while self._recent_rows:
            row = self._recent_rows.pop()
            self.recent_layout.removeWidget(row)
            row.deleteLater()
        self.recent_count.setText("0")
        self.history_placeholder.setVisible(True)
        self.update_session_stats(0, 0, 0)
