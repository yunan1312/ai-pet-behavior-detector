"""设置窗口：识别、语音、外观与存储。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.speech_synthesizer import VOICE_DISPLAY_NAMES
from ui.components import Card, make_label
from ui.theme import DARK, LIGHT, THEME_NAMES, current_theme, theme_manager
from utils.config_loader import ConfigLoader
from utils.paths import BASE_DIR, CONFIG_DIR, DATA_DIR, open_path

PAGE_TITLES = ["识别", "语音", "外观", "存储"]


class SettingsDialog(QDialog):
    settings_applied = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        config_loader: Optional[ConfigLoader] = None,
        speech_synthesizer=None,
        record_manager=None,
    ) -> None:
        super().__init__(parent)
        self.config_loader = config_loader or ConfigLoader()
        self.speech_synthesizer = speech_synthesizer
        self.record_manager = record_manager
        self._original_theme = current_theme()
        self.setWindowTitle("设置")
        self.setModal(True)
        self.resize(780, 600)
        self.setMinimumSize(700, 540)
        self.init_ui()
        self.load_settings()

    # ------------------------------------------------------------------ UI
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        title = make_label("设置", "h1")
        subtitle = make_label("分析参数与窗口布局会保存在本地", "caption")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(12)
        self.nav = QListWidget()
        self.nav.setFixedWidth(148)
        self.nav.addItems(PAGE_TITLES)
        self.nav.currentRowChanged.connect(self._switch_page)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_recognition_page())
        self.stack.addWidget(self._build_speech_page())
        self.stack.addWidget(self._build_appearance_page())
        self.stack.addWidget(self._build_storage_page())
        content.addWidget(self.nav)
        content.addWidget(self.stack, 1)
        layout.addLayout(content, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        restore_btn = QPushButton("恢复默认")
        restore_btn.setProperty("variant", "ghost")
        restore_btn.setFixedHeight(34)
        restore_btn.clicked.connect(self.restore_defaults)
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedHeight(34)
        cancel_btn.setMinimumWidth(84)
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存设置")
        save_btn.setProperty("variant", "primary")
        save_btn.setFixedHeight(34)
        save_btn.setMinimumWidth(104)
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.apply_settings)
        footer.addWidget(restore_btn)
        footer.addStretch(1)
        footer.addWidget(cancel_btn)
        footer.addWidget(save_btn)
        layout.addLayout(footer)

        self.nav.setCurrentRow(0)

    def _page(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("Page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        heading = make_label(title, "h2")
        caption = make_label(description, "caption", wrap=True)
        layout.addWidget(heading)
        layout.addWidget(caption)
        return page, layout

    def _build_recognition_page(self) -> QWidget:
        page, layout = self._page(
            "识别设置",
            "置信度越高，结果越保守；调整后将在下一次分析中生效。",
        )

        card = Card()
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        confidence_row = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(75)
        self.confidence_slider.valueChanged.connect(self._update_confidence_label)
        self.confidence_value = make_label("0.75", "h3")
        self.confidence_value.setFixedWidth(48)
        self.confidence_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        confidence_row.addWidget(self.confidence_slider, 1)
        confidence_row.addWidget(self.confidence_value)
        form.addRow(make_label("置信度阈值", "subtitle"), confidence_row)

        self.fps_combo = QComboBox()
        for value in (10, 15, 20, 25, 30):
            self.fps_combo.addItem(f"{value} FPS", value)
        form.addRow(make_label("实时分析帧率", "subtitle"), self.fps_combo)

        hint = make_label(
            "提示：实时识别与视频分析均在后台线程执行；视频会按时间采样以兼顾覆盖度与耗时。",
            "caption",
            wrap=True,
        )
        form.addRow(hint)
        card.body().addLayout(form)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_speech_page(self) -> QWidget:
        page, layout = self._page(
            "语音设置",
            "使用 Microsoft Edge TTS 播报意图解读，语音合成需要网络连接。",
        )
        card = Card()
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        self.speech_enabled_check = QCheckBox("启用语音播报")
        form.addRow(self.speech_enabled_check)

        self.language_combo = QComboBox()
        self.language_combo.addItem("中文（zh_CN）", "zh_CN")
        self.language_combo.addItem("English（en_US）", "en_US")
        self.language_combo.currentIndexChanged.connect(self._refresh_voices)
        form.addRow(make_label("播报语言", "subtitle"), self.language_combo)

        self.voice_combo = QComboBox()
        form.addRow(make_label("音色", "subtitle"), self.voice_combo)

        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._update_speed_label)
        self.speed_value = make_label("1.0x", "h3")
        self.speed_value.setFixedWidth(48)
        self.speed_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_value)
        form.addRow(make_label("语速", "subtitle"), speed_row)

        volume_row = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self._update_volume_label)
        self.volume_value = make_label("80%", "h3")
        self.volume_value.setFixedWidth(48)
        self.volume_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        volume_row.addWidget(self.volume_slider, 1)
        volume_row.addWidget(self.volume_value)
        form.addRow(make_label("音量", "subtitle"), volume_row)

        card.body().addLayout(form)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_appearance_page(self) -> QWidget:
        page, layout = self._page(
            "外观",
            "启动默认浅色；主题切换仅对当前会话生效，窗口布局会自动保存。",
        )
        card = Card()
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("浅色", LIGHT)
        self.theme_combo.addItem("深色", DARK)
        self.theme_combo.currentIndexChanged.connect(self._preview_theme)
        form.addRow(make_label("界面主题", "subtitle"), self.theme_combo)

        self.show_welcome_check = QCheckBox("启动时显示欢迎页")
        form.addRow(self.show_welcome_check)
        card.body().addLayout(form)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_storage_page(self) -> QWidget:
        page, layout = self._page(
            "存储与数据",
            "历史记录与日志均保存在本机，不会上传。",
        )
        card = Card()
        card.body().addWidget(make_label("数据位置", "h3"))

        for label_text, path in (
            ("数据目录", DATA_DIR),
            ("配置文件", CONFIG_DIR / "app_config.yaml"),
        ):
            row = QHBoxLayout()
            row.setSpacing(8)
            name = make_label(label_text, "caption")
            name.setFixedWidth(64)
            value = make_label(str(path), "caption")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setWordWrap(True)
            open_btn = QPushButton("打开")
            open_btn.setProperty("variant", "ghost")
            open_btn.setFixedHeight(26)
            open_btn.clicked.connect(lambda _checked=False, p=path: open_path(p))
            row.addWidget(name)
            row.addWidget(value, 1)
            row.addWidget(open_btn)
            card.body().addLayout(row)

        card.body().addWidget(make_label("历史记录", "h3"))
        count_row = QHBoxLayout()
        self.record_count_label = make_label("-- 条记录", "subtitle")
        clear_btn = QPushButton("清空全部历史记录")
        clear_btn.setProperty("variant", "ghost")
        clear_btn.setFixedHeight(30)
        clear_btn.clicked.connect(self.clear_history)
        count_row.addWidget(self.record_count_label)
        count_row.addStretch(1)
        count_row.addWidget(clear_btn)
        card.body().addLayout(count_row)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    # -------------------------------------------------------------- behavior
    def _switch_page(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)

    def load_settings(self) -> None:
        confidence = float(self.config_loader.get("recognition.confidence_threshold", 0.75))
        self.confidence_slider.setValue(int(round(confidence * 100)))
        self._update_confidence_label(self.confidence_slider.value())

        fps = int(self.config_loader.get("recognition.target_fps", 15))
        index = self.fps_combo.findData(fps)
        self.fps_combo.setCurrentIndex(index if index >= 0 else 1)

        self.speech_enabled_check.setChecked(bool(self.config_loader.get("speech.enabled", True)))
        language = self.config_loader.get("speech.language", "zh_CN")
        lang_index = self.language_combo.findData(language)
        self.language_combo.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        self._refresh_voices()
        voice = self.config_loader.get("speech.voice", "female_warm")
        voice_index = self.voice_combo.findData(voice)
        if voice_index >= 0:
            self.voice_combo.setCurrentIndex(voice_index)

        speed = float(self.config_loader.get("speech.speed", 1.0))
        self.speed_slider.setValue(int(round(speed * 100)))
        self._update_speed_label(self.speed_slider.value())
        volume = float(self.config_loader.get("speech.volume", 0.8))
        self.volume_slider.setValue(int(round(volume * 100)))
        self._update_volume_label(self.volume_slider.value())

        theme = current_theme()
        theme_index = self.theme_combo.findData(theme)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(theme_index if theme_index >= 0 else 0)
        self.theme_combo.blockSignals(False)
        self.show_welcome_check.setChecked(
            bool(self.config_loader.get("app.show_welcome", True))
        )

        if self.record_manager:
            self.record_count_label.setText(
                f"{self.record_manager.get_records_count()} 条记录"
            )
        self.speech_enabled_check.setEnabled(True)

    def _refresh_voices(self) -> None:
        language = self.language_combo.currentData() or "zh_CN"
        current = self.voice_combo.currentData()
        names: Dict[str, str] = VOICE_DISPLAY_NAMES.get(language, VOICE_DISPLAY_NAMES["zh_CN"])
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for voice_id, display in names.items():
            self.voice_combo.addItem(display, voice_id)
        if current:
            index = self.voice_combo.findData(current)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
        self.voice_combo.blockSignals(False)

    def _update_confidence_label(self, value: int) -> None:
        self.confidence_value.setText(f"{value / 100:.2f}")

    def _update_speed_label(self, value: int) -> None:
        self.speed_value.setText(f"{value / 100:.1f}x")

    def _update_volume_label(self, value: int) -> None:
        self.volume_value.setText(f"{value}%")

    def _preview_theme(self, _index: int) -> None:
        theme = self.theme_combo.currentData() or LIGHT
        theme_manager.set_theme(theme)

    def restore_defaults(self) -> None:
        self.confidence_slider.setValue(75)
        index = self.fps_combo.findData(15)
        self.fps_combo.setCurrentIndex(index if index >= 0 else 1)
        self.speech_enabled_check.setChecked(True)
        lang_index = self.language_combo.findData("zh_CN")
        self.language_combo.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        self._refresh_voices()
        voice_index = self.voice_combo.findData("female_warm")
        if voice_index >= 0:
            self.voice_combo.setCurrentIndex(voice_index)
        self.speed_slider.setValue(100)
        self.volume_slider.setValue(80)
        theme_index = self.theme_combo.findData(LIGHT)
        self.theme_combo.setCurrentIndex(theme_index if theme_index >= 0 else 0)
        self._preview_theme(self.theme_combo.currentIndex())
        self.show_welcome_check.setChecked(True)

    def clear_history(self) -> None:
        if not self.record_manager:
            return
        reply = QMessageBox.question(
            self,
            "清空历史记录",
            "确定要删除全部历史记录吗？\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.record_manager.clear_all_records()
        self.record_count_label.setText("0 条记录")
        self.settings_applied.emit()

    def apply_settings(self) -> None:
        confidence = self.confidence_slider.value() / 100.0
        fps = int(self.fps_combo.currentData() or 15)
        language = self.language_combo.currentData() or "zh_CN"
        voice = self.voice_combo.currentData() or "female_warm"
        speed = self.speed_slider.value() / 100.0
        volume = self.volume_slider.value() / 100.0
        speech_enabled = self.speech_enabled_check.isChecked()
        theme = self.theme_combo.currentData() or LIGHT

        updates = {
            "recognition.confidence_threshold": confidence,
            "recognition.target_fps": fps,
            "speech.enabled": speech_enabled,
            "speech.language": language,
            "speech.voice": voice,
            "speech.speed": speed,
            "speech.volume": volume,
            "app.show_welcome": self.show_welcome_check.isChecked(),
        }
        self.config_loader.update_many(updates)
        theme_manager.set_theme(theme)

        if self.speech_synthesizer:
            self.speech_synthesizer.set_language(language)
            self.speech_synthesizer.set_voice(voice)
            self.speech_synthesizer.set_speed(speed)
            self.speech_synthesizer.set_volume(volume)
            if not speech_enabled:
                self.speech_synthesizer.stop()

        self.settings_applied.emit()
        self.accept()

    def reject(self) -> None:
        if current_theme() != self._original_theme:
            theme_manager.set_theme(self._original_theme)
        super().reject()
