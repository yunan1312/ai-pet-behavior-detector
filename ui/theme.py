"""集中式设计系统：颜色令牌、全局样式表与主题管理。

所有界面只使用这里的语义化颜色与组件角色，避免在业务代码中散落
硬编码色值，从而保证浅色/深色主题一致、可维护。
"""

from __future__ import annotations

from string import Template
from typing import Dict

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QWidget

from utils.paths import RESOURCES_DIR

LIGHT = "light"
DARK = "dark"
THEME_NAMES = {LIGHT: "浅色", DARK: "深色"}
SETTINGS_ORG = "PetBehaviorAnalyzer"
SETTINGS_APP = "PetBehaviorAnalyzer"

FONT_STACK = (
    '"Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", '
    '"Noto Sans CJK SC", "PingFang SC", sans-serif'
)
MONO_STACK = '"Cascadia Mono", "Consolas", "JetBrains Mono", "Courier New", monospace'

PALETTES: Dict[str, Dict[str, str]] = {
    LIGHT: {
        "bg": "#F3F5F9",
        "surface": "#FFFFFF",
        "surface_alt": "#F7F9FC",
        "surface_hover": "#EEF2F8",
        "border": "#E2E7F0",
        "border_strong": "#CDD5E1",
        "text": "#1B2433",
        "text_secondary": "#566274",
        "text_muted": "#8B97A8",
        "text_inverse": "#FFFFFF",
        "primary": "#2563EB",
        "primary_hover": "#1D4ED8",
        "primary_pressed": "#1E40AF",
        "primary_soft": "#E8F0FE",
        "accent": "#7C3AED",
        "success": "#15803D",
        "success_hover": "#166534",
        "success_soft": "#E7F6EC",
        "danger": "#DC2626",
        "danger_hover": "#B91C1C",
        "danger_soft": "#FDECEC",
        "warning": "#B45309",
        "warning_soft": "#FEF3C7",
        "video_bg": "#0B1220",
        "video_surface": "#111C2E",
        "video_grid": "#1B2A43",
        "video_text": "#E8F0FF",
        "video_muted": "#8FA3BF",
        "log_bg": "#F8FAFD",
        "log_text": "#334155",
        "hero_from": "#2563EB",
        "hero_to": "#7C3AED",
        "selection": "#DBEAFE",
        "shadow": "#1B243312",
    },
    DARK: {
        "bg": "#0E141D",
        "surface": "#161F2B",
        "surface_alt": "#1C2735",
        "surface_hover": "#243244",
        "border": "#293647",
        "border_strong": "#3A4B60",
        "text": "#E8EEF7",
        "text_secondary": "#A9B6C8",
        "text_muted": "#738296",
        "text_inverse": "#0B1220",
        "primary": "#5B9CFF",
        "primary_hover": "#78AEFF",
        "primary_pressed": "#3C82F0",
        "primary_soft": "#1E2F4A",
        "accent": "#A78BFA",
        "success": "#4ADE80",
        "success_hover": "#6EE7A0",
        "success_soft": "#12301F",
        "danger": "#F87171",
        "danger_hover": "#FCA5A5",
        "danger_soft": "#3A1B1E",
        "warning": "#FBBF24",
        "warning_soft": "#3A2B0E",
        "video_bg": "#060A11",
        "video_surface": "#0C1420",
        "video_grid": "#16202F",
        "video_text": "#E8F0FF",
        "video_muted": "#8FA3BF",
        "log_bg": "#0B111A",
        "log_text": "#C7D2E0",
        "hero_from": "#1D4ED8",
        "hero_to": "#6D28D9",
        "selection": "#223A5E",
        "shadow": "#00000055",
    },
}


def tokens(theme: str | None = None) -> Dict[str, str]:
    """返回当前主题的颜色令牌，供自绘组件使用。"""
    return PALETTES[theme or current_theme()]


_QSS = Template(
    """
    QMainWindow, QDialog {
        background-color: $bg;
        color: $text;
    }
    QWidget {
        font-family: $font_stack;
        font-size: 13px;
        color: $text;
    }
    QWidget#CentralArea, QWidget#Page, QWidget#Transparent {
        background: transparent;
    }
    QToolTip {
        background-color: $text;
        color: $surface;
        border: none;
        border-radius: 6px;
        padding: 6px 9px;
        font-size: 12px;
    }
    QMenuBar {
        background-color: $surface;
        color: $text_secondary;
        border-bottom: 1px solid $border;
        padding: 2px 6px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 7px 12px;
        border-radius: 6px;
    }
    QMenuBar::item:selected {
        background-color: $primary_soft;
        color: $primary;
    }
    QMenu {
        background-color: $surface;
        color: $text;
        border: 1px solid $border;
        border-radius: 8px;
        padding: 6px;
    }
    QMenu::item {
        padding: 7px 26px 7px 20px;
        border-radius: 5px;
    }
    QMenu::item:selected {
        background-color: $primary_soft;
        color: $primary;
    }
    QMenu::item:disabled {
        color: $text_muted;
    }
    QMenu::separator {
        height: 1px;
        background: $border;
        margin: 5px 8px;
    }
    QMenu::indicator {
        width: 14px;
        height: 14px;
        margin-left: 6px;
    }
    QStatusBar {
        background-color: $surface;
        color: $text_secondary;
        border-top: 1px solid $border;
        min-height: 28px;
    }
    QStatusBar::item {
        border: none;
    }
    QStatusBar QLabel {
        color: $text_secondary;
        font-size: 12px;
    }
    QLabel {
        background: transparent;
    }
    QLabel[role="h1"] {
        font-size: 22px;
        font-weight: 700;
        color: $text;
    }
    QLabel[role="h2"] {
        font-size: 16px;
        font-weight: 700;
        color: $text;
    }
    QLabel[role="h3"] {
        font-size: 13px;
        font-weight: 600;
        color: $text;
    }
    QLabel[role="subtitle"] {
        font-size: 12px;
        color: $text_secondary;
    }
    QLabel[role="caption"] {
        font-size: 11px;
        color: $text_muted;
    }
    QLabel[role="muted"] {
        color: $text_muted;
    }
    QLabel[role="danger"] {
        color: $danger;
    }
    QLabel[role="success"] {
        color: $success;
    }
    QLabel[role="section"] {
        font-size: 11px;
        font-weight: 700;
        color: $text_muted;
        letter-spacing: 1px;
    }
    QLabel[chip="true"] {
        background-color: $surface_alt;
        color: $text_secondary;
        border: 1px solid $border;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }
    QLabel[tone="primary"] {
        background-color: $primary_soft;
        color: $primary;
        border: 1px solid $primary_soft;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }
    QLabel[tone="success"] {
        background-color: $success_soft;
        color: $success;
        border: 1px solid $success_soft;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }
    QLabel[tone="warning"] {
        background-color: $warning_soft;
        color: $warning;
        border: 1px solid $warning_soft;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }
    QLabel[tone="danger"] {
        background-color: $danger_soft;
        color: $danger;
        border: 1px solid $danger_soft;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }
    QFrame[card="true"] {
        background-color: $surface;
        border: 1px solid $border;
        border-radius: 12px;
    }
    QFrame[card="true"] QLabel {
        background: transparent;
    }
    QFrame[recent="true"] {
        background-color: $surface_alt;
        border: 1px solid transparent;
        border-radius: 8px;
    }
    QFrame[recent="true"]:hover {
        border-color: $border_strong;
    }
    QFrame#VideoShell {
        background-color: $video_bg;
        border: 1px solid $border;
        border-radius: 14px;
    }
    QFrame#InfoHero {
        border: none;
        border-radius: 12px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 $hero_from, stop:1 $hero_to);
    }
    QFrame#InfoHero QLabel {
        color: #FFFFFF;
        background: transparent;
    }
    QLabel#HeroBehavior {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
    }
    QLabel#HeroTitle {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.78);
    }
    QLabel#HeroConfidence {
        font-size: 12px;
        color: #E9EEFF;
    }
    QLabel#HeroEmotion {
        font-size: 11px;
        color: #F2F5FF;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 9px;
        padding: 2px 9px;
    }
    QProgressBar#HeroProgress {
        background-color: rgba(255, 255, 255, 0.24);
        border: none;
        border-radius: 4px;
        min-height: 8px;
        max-height: 8px;
    }
    QProgressBar#HeroProgress::chunk {
        background-color: #FFFFFF;
        border-radius: 4px;
    }
    QLabel#IntentText {
        background-color: $surface_alt;
        color: $text;
        border: 1px solid $border;
        border-radius: 10px;
        padding: 13px;
        font-size: 13px;
    }
    QLabel#StatValue {
        font-size: 18px;
        font-weight: 700;
        color: $text;
    }
    QLabel#StatCaption {
        font-size: 10px;
        color: $text_muted;
    }
    QFrame#StatTile {
        background-color: $surface_alt;
        border: 1px solid $border;
        border-radius: 9px;
    }
    QFrame#StatTile QLabel {
        background: transparent;
    }
    QLabel#BrandTitle {
        font-size: 16px;
        font-weight: 700;
        color: $text;
    }
    QLabel#BrandSub {
        font-size: 11px;
        color: $text_muted;
    }
    QPushButton {
        background-color: $surface;
        color: $text;
        border: 1px solid $border_strong;
        border-radius: 8px;
        padding: 7px 14px;
        min-height: 20px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: $surface_hover;
    }
    QPushButton:pressed {
        background-color: $border;
    }
    QPushButton:focus {
        border-color: $primary;
        outline: none;
    }
    QPushButton:disabled {
        background-color: $surface_alt;
        color: $text_muted;
        border-color: $border;
    }
    QPushButton[variant="primary"] {
        background-color: $primary;
        color: #FFFFFF;
        border: 1px solid $primary;
        font-weight: 600;
    }
    QPushButton[variant="primary"]:hover {
        background-color: $primary_hover;
        border-color: $primary_hover;
    }
    QPushButton[variant="primary"]:pressed {
        background-color: $primary_pressed;
    }
    QPushButton[variant="primary"]:disabled {
        background-color: $surface_hover;
        color: $text_muted;
        border-color: transparent;
    }
    QPushButton[variant="success"] {
        background-color: $success;
        color: #FFFFFF;
        border: 1px solid $success;
        font-weight: 600;
    }
    QPushButton[variant="success"]:hover {
        background-color: $success_hover;
        border-color: $success_hover;
    }
    QPushButton[variant="success"]:disabled {
        background-color: $surface_hover;
        color: $text_muted;
        border-color: transparent;
    }
    QPushButton[variant="danger"] {
        background-color: $danger;
        color: #FFFFFF;
        border: 1px solid $danger;
        font-weight: 600;
    }
    QPushButton[variant="danger"]:hover {
        background-color: $danger_hover;
        border-color: $danger_hover;
    }
    QPushButton[variant="danger"]:disabled {
        background-color: $surface_hover;
        color: $text_muted;
        border-color: transparent;
    }
    QPushButton[variant="soft"] {
        background-color: $primary_soft;
        color: $primary;
        border: 1px solid transparent;
        font-weight: 600;
    }
    QPushButton[variant="soft"]:hover {
        background-color: $selection;
    }
    QPushButton[variant="soft"]:disabled {
        background-color: $surface_alt;
        color: $text_muted;
    }
    QPushButton[variant="ghost"] {
        background: transparent;
        color: $text_secondary;
        border: 1px solid transparent;
        padding: 6px 10px;
    }
    QPushButton[variant="ghost"]:hover {
        background-color: $surface_hover;
        color: $text;
    }
    QPushButton[variant="ghost"]:disabled {
        color: $text_muted;
        background: transparent;
    }
    QPushButton#IconButton {
        padding: 4px;
        min-width: 30px;
        max-width: 30px;
        min-height: 26px;
        max-height: 26px;
        border-radius: 7px;
    }
    QToolButton {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 7px;
        padding: 5px 9px;
        color: $text_secondary;
    }
    QToolButton:hover {
        background-color: $surface_hover;
        color: $text;
    }
    QToolButton:pressed {
        background-color: $border;
    }
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: $surface;
        color: $text;
        border: 1px solid $border_strong;
        border-radius: 8px;
        padding: 6px 9px;
        selection-background-color: $primary;
        selection-color: #FFFFFF;
    }
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: $text_muted;
    }
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border-color: $primary;
    }
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
        background-color: $surface_alt;
        color: $text_muted;
        border-color: $border;
    }
    QLineEdit[search="true"] {
        padding-left: 10px;
    }
    QComboBox {
        padding-right: 24px;
    }
    QComboBox::drop-down {
        border: none;
        width: 26px;
        subcontrol-origin: padding;
        subcontrol-position: top right;
    }
    QComboBox::down-arrow {
        image: url("$chevron");
        width: 12px;
        height: 12px;
    }
    QComboBox QAbstractItemView {
        background-color: $surface;
        color: $text;
        border: 1px solid $border;
        border-radius: 6px;
        selection-background-color: $primary_soft;
        selection-color: $primary;
        padding: 4px;
        outline: none;
    }
    QComboBox QAbstractItemView::item {
        min-height: 26px;
        padding: 3px 8px;
        border-radius: 4px;
    }
    QPlainTextEdit#LogView {
        background-color: $log_bg;
        color: $log_text;
        border: 1px solid $border;
        border-radius: 10px;
        font-family: $mono_stack;
        font-size: 12px;
        padding: 8px;
    }
    QPlainTextEdit#LogView QScrollBar:vertical {
        background: transparent;
    }
    QProgressBar {
        background-color: $surface_alt;
        border: none;
        border-radius: 4px;
        min-height: 8px;
        max-height: 8px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk {
        border-radius: 4px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 $primary, stop:1 $accent);
    }
    QProgressBar[level="high"]::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 $success, stop:1 $primary);
    }
    QProgressBar[level="mid"]::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 $warning, stop:1 $success);
    }
    QProgressBar[level="low"]::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 $danger, stop:1 $warning);
    }
    QTabWidget::pane {
        border: none;
        border-top: 1px solid $border;
        background-color: $surface;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
    }
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        background: transparent;
        color: $text_secondary;
        padding: 9px 16px;
        margin-right: 4px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: 12px;
        font-weight: 600;
    }
    QTabBar::tab:hover {
        color: $text;
        background-color: $surface_hover;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:selected {
        color: $primary;
        border-bottom: 2px solid $primary;
        background: transparent;
    }
    QTabWidget::tab-bar {
        alignment: left;
    }
    QScrollArea {
        background: transparent;
        border: none;
    }
    QScrollArea > QWidget#InfoContainer {
        background: transparent;
    }
    QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: $border_strong;
        min-height: 28px;
        border-radius: 5px;
        margin: 2px;
    }
    QScrollBar::handle:vertical:hover {
        background: $text_muted;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
    }
    QScrollBar:horizontal {
        background: transparent;
        height: 10px;
        margin: 0;
    }
    QScrollBar::handle:horizontal {
        background: $border_strong;
        min-width: 28px;
        border-radius: 5px;
        margin: 2px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: transparent;
    }
    QTableWidget, QTableView {
        background-color: $surface;
        alternate-background-color: $surface_alt;
        color: $text;
        border: none;
        gridline-color: $border;
        selection-background-color: $selection;
        selection-color: $text;
        border-radius: 8px;
        font-size: 12px;
    }
    QTableWidget::item, QTableView::item {
        padding: 6px 8px;
        border-bottom: 1px solid $border;
    }
    QTableWidget::item:selected, QTableView::item:selected {
        background-color: $selection;
        color: $text;
    }
    QHeaderView::section {
        background-color: $surface_alt;
        color: $text_secondary;
        padding: 8px;
        border: none;
        border-bottom: 1px solid $border;
        font-weight: 600;
        font-size: 12px;
    }
    QTableCornerButton::section {
        background-color: $surface_alt;
        border: none;
    }
    QListWidget {
        background-color: $surface;
        color: $text;
        border: 1px solid $border;
        border-radius: 9px;
        padding: 6px;
        outline: none;
    }
    QListWidget::item {
        padding: 8px 10px;
        border-radius: 6px;
        color: $text_secondary;
    }
    QListWidget::item:hover {
        background-color: $surface_hover;
        color: $text;
    }
    QListWidget::item:selected {
        background-color: $primary_soft;
        color: $primary;
    }
    QTextBrowser {
        background-color: $surface;
        color: $text;
        border: 1px solid $border;
        border-radius: 9px;
        padding: 8px;
        selection-background-color: $primary_soft;
        selection-color: $text;
    }
    QSplitter::handle {
        background: transparent;
    }
    QSplitter::handle:hover {
        background-color: $primary_soft;
    }
    QSplitter::handle:horizontal {
        width: 8px;
    }
    QSplitter::handle:vertical {
        height: 8px;
    }
    QSlider {
        min-height: 22px;
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: $border;
        border-radius: 2px;
    }
    QSlider::sub-page:horizontal {
        background: $primary;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: $surface;
        border: 2px solid $primary;
        width: 14px;
        height: 14px;
        margin: -7px 0;
        border-radius: 9px;
    }
    QSlider::handle:horizontal:hover {
        background: $primary_soft;
    }
    QCheckBox, QRadioButton {
        spacing: 8px;
        color: $text;
        background: transparent;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid $border_strong;
        background-color: $surface;
    }
    QCheckBox::indicator:hover {
        border-color: $primary;
    }
    QCheckBox::indicator:checked {
        background-color: $primary;
        border-color: $primary;
        image: url("$check");
    }
    QCheckBox::indicator:disabled {
        background-color: $surface_alt;
        border-color: $border;
    }
    QCheckBox:disabled, QRadioButton:disabled {
        color: $text_muted;
    }
    QGroupBox {
        border: 1px solid $border;
        border-radius: 10px;
        margin-top: 14px;
        padding: 14px 12px 10px 12px;
        background-color: $surface;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 5px;
        color: $text_secondary;
    }
    QMessageBox {
        background-color: $surface;
    }
    QMessageBox QLabel {
        color: $text;
    }
    QMessageBox QPushButton {
        min-width: 76px;
    }
    QLabel#Toast {
        background-color: $text;
        color: $surface;
        border-radius: 16px;
        padding: 9px 18px;
        font-size: 12px;
        font-weight: 500;
    }
    QLabel#StatusChip {
        background-color: $surface_alt;
        color: $text_secondary;
        border: 1px solid $border;
        border-radius: 9px;
        padding: 2px 8px;
        font-size: 11px;
    }
    QLabel#StatusChip[tone="success"] {
        background-color: $success_soft;
        color: $success;
        border-color: $success_soft;
    }
    QLabel#StatusChip[tone="warning"] {
        background-color: $warning_soft;
        color: $warning;
        border-color: $warning_soft;
    }
    QLabel#StatusChip[tone="danger"] {
        background-color: $danger_soft;
        color: $danger;
        border-color: $danger_soft;
    }
    QWidget#EmptyState {
        background: transparent;
    }
    QWidget#EmptyState QLabel {
        color: $text_muted;
    }
    """
)


def build_stylesheet(theme: str | None = None) -> str:
    active = theme or current_theme()
    palette = dict(tokens(active))
    palette["font_stack"] = FONT_STACK
    palette["mono_stack"] = MONO_STACK
    palette["chevron"] = _asset_url(f"chevron_{active}.png")
    palette["check"] = _asset_url("check_white.png")
    return _QSS.safe_substitute(palette)


def _asset_url(name: str) -> str:
    path = RESOURCES_DIR / name
    # QSS 的 url() 不解析 file:// 与百分号编码，使用带正斜杠的本地路径
    return str(path).replace("\\", "/")


def build_palette(theme: str | None = None) -> QPalette:
    p = tokens(theme)
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: p["bg"],
        QPalette.ColorRole.WindowText: p["text"],
        QPalette.ColorRole.Base: p["surface"],
        QPalette.ColorRole.AlternateBase: p["surface_alt"],
        QPalette.ColorRole.Text: p["text"],
        QPalette.ColorRole.Button: p["surface"],
        QPalette.ColorRole.ButtonText: p["text"],
        QPalette.ColorRole.BrightText: "#FFFFFF",
        QPalette.ColorRole.Highlight: p["primary"],
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.ToolTipBase: p["text"],
        QPalette.ColorRole.ToolTipText: p["surface"],
        QPalette.ColorRole.PlaceholderText: p["text_muted"],
        QPalette.ColorRole.Link: p["primary"],
    }
    for role, value in colors.items():
        color = QColor(value)
        palette.setColor(role, color)
        palette.setColor(QPalette.ColorGroup.Inactive, role, color)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(p["text_muted"]))
    return palette


class ThemeManager(QObject):
    """应用级主题管理器。

    每次启动都从浅色主题开始，不读取上次会话的选择，避免深色
    偏好被意外带进下一次启动；运行期间仍可随时切换到深色。
    """

    theme_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._theme = LIGHT

    @property
    def theme(self) -> str:
        return self._theme

    def apply(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setPalette(build_palette(self._theme))
            app.setStyleSheet(build_stylesheet(self._theme))
        self.theme_changed.emit(self._theme)

    def set_theme(self, theme: str) -> bool:
        if theme not in PALETTES:
            theme = LIGHT
        changed = theme != self._theme
        self._theme = theme
        if changed:
            self.apply()
        return changed

    def toggle(self) -> str:
        self.set_theme(DARK if self._theme == LIGHT else LIGHT)
        return self._theme


theme_manager = ThemeManager()


def current_theme() -> str:
    return theme_manager.theme


def repolish(widget: QWidget) -> None:
    """动态属性变化后刷新控件样式。"""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_role(widget: QWidget, role: str) -> None:
    widget.setProperty("role", role)
    repolish(widget)


def set_chip(widget: QWidget, tone: str | None = None) -> None:
    widget.setProperty("chip", "true")
    if tone:
        widget.setProperty("tone", tone)
    repolish(widget)
