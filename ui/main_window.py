"""主窗口：整合视频画布、操作栏、洞察面板、运行日志与历史记录。

设计要点：
* 所有耗时任务（模型加载、摄像头扫描、实时/视频/图片分析）均在后台线程执行；
* 双主题、深浅色切换与窗口布局持久化；
* 视频/图片导入、拖拽、最近文件、快捷键与状态栏进度形成完整工作流。
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.intent_mapper import IntentMapper
from core.record_manager import RecordManager
from core.speech_synthesizer import SpeechSynthesizer
from core.video_handler import VideoStreamHandler
from ui.about_dialog import AboutDialog
from ui.components import Card, Toast, make_chip, make_label
from ui.control_panel import ControlPanel
from ui.history_panel import HistoryPanel
from ui.help_dialog import HelpDialog
from ui.info_panel import InfoPanel
from ui.log_panel import LogPanel
from ui.settings_dialog import SettingsDialog
from ui.theme import (
    DARK,
    SETTINGS_APP,
    SETTINGS_ORG,
    repolish,
    theme_manager,
)
from ui.video_panel import VideoPanel
from ui.welcome_dialog import WelcomeDialog
from ui.workers import (
    CameraScanWorker,
    ImageAnalysisWorker,
    ModelLoader,
    RealtimeAnalysisWorker,
    VideoAnalysisWorker,
)
from utils.config_loader import ConfigLoader
from utils.constants import APP_NAME, APP_VERSION, BEHAVIOR_DISPLAY_NAMES
from utils.logger import get_logger
from utils.paths import DATA_DIR, app_icon_path, open_path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class MainWindow(QMainWindow):
    def __init__(
        self,
        show_welcome: Optional[bool] = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__()
        self.config_loader = ConfigLoader()
        self.logger = get_logger()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # ---- 运行时状态 -------------------------------------------------
        self.pose_model = None
        self.pose_processor = None
        self.behavior_model = None
        self.behavior_classifier = None
        self.frame_analyzer = None
        self.file_analyzer = None

        self._initialized = False
        self._loading = False
        self._busy_kind: Optional[str] = None
        self._force_show_welcome = show_welcome
        self._auto_start = auto_start
        self._camera_scanning = False
        self._bottom_collapsed = False
        self._splitter_sizes_before_collapse: List[int] = []
        self._frame_seen = False
        self._current_media_path = ""

        self.model_loader: Optional[ModelLoader] = None
        self.camera_worker: Optional[CameraScanWorker] = None
        self.realtime_worker: Optional[RealtimeAnalysisWorker] = None
        self.video_worker: Optional[VideoAnalysisWorker] = None
        self.image_worker: Optional[ImageAnalysisWorker] = None
        self.video_handler: Optional[VideoStreamHandler] = None

        self.current_record_id: Optional[str] = None
        self._session_started = 0.0
        self._session_frames = 0
        self._session_results: List[Dict] = []
        self._last_logged_behavior: Optional[str] = None
        self._last_logged_at = 0.0
        self._last_spoken_behavior: Optional[str] = None
        self._last_spoken_at = 0.0

        self._speech_enabled = bool(self.config_loader.get("speech.enabled", True))

        self.intent_mapper = IntentMapper(self.config_loader)
        self.speech_synthesizer = SpeechSynthesizer(self.config_loader)
        self.record_manager = RecordManager()
        self._recent_files = self._load_recent_files()

        self._session_timer = QTimer(self)
        self._session_timer.setInterval(500)
        self._session_timer.timeout.connect(self._update_session_tick)

        self.init_ui()
        self._build_menus()
        self._init_status_bar()
        self.toast = Toast(self)

        self.history_panel.reload()
        self._restore_layout()
        self._sync_speech_action()
        self._on_theme_changed(theme_manager.theme)
        self._refresh_actions()

        self.log_panel.append_log(f"欢迎使用 {APP_NAME} v{APP_VERSION}", "success")
        self.log_panel.append_log("界面已就绪，正在加载模型并检测摄像头…", "info")

        if auto_start:
            QTimer.singleShot(120, self._start_camera_scan)
            QTimer.singleShot(200, self.load_models_async)
        if show_welcome is not False:
            QTimer.singleShot(320, self._maybe_show_welcome)

    # ------------------------------------------------------------------ UI
    def init_ui(self) -> None:
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        icon_path = app_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1180, 700)
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        central = QWidget()
        central.setObjectName("CentralArea")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        root.addWidget(self._build_header())

        self.outer_splitter = QSplitter(Qt.Orientation.Vertical)
        self.outer_splitter.setChildrenCollapsible(False)
        self.outer_splitter.setHandleWidth(8)

        work = QWidget()
        work.setObjectName("Page")
        work_layout = QHBoxLayout(work)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(12)

        left = QWidget()
        left.setObjectName("Page")
        left.setMinimumWidth(560)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.video_panel = VideoPanel()
        self.control_panel = ControlPanel()
        left_layout.addWidget(self.video_panel, 1)
        left_layout.addWidget(self.control_panel, 0)

        self.info_panel = InfoPanel()
        self.info_panel.setFixedWidth(384)

        work_layout.addWidget(left, 1)
        work_layout.addWidget(self.info_panel, 0)
        self.outer_splitter.addWidget(work)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.log_panel = LogPanel()
        self.history_panel = HistoryPanel(self.record_manager)
        self.bottom_tabs.addTab(self.log_panel, "运行日志")
        self.bottom_tabs.addTab(self.history_panel, "历史记录")

        self._collapse_btn = QPushButton("收起")
        self._collapse_btn.setProperty("variant", "ghost")
        self._collapse_btn.setFixedHeight(26)
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.setToolTip("收起底部面板，扩大视频区域（Ctrl+L）")
        self._collapse_btn.clicked.connect(self._toggle_bottom)
        self.bottom_tabs.setCornerWidget(
            self._collapse_btn, Qt.Corner.TopRightCorner
        )
        self.bottom_tabs.setMinimumHeight(190)

        self.outer_splitter.addWidget(self.bottom_tabs)
        # 两个区域按比例分配多余空间：窗口变小时视频与底部同步收缩，
        # 避免底部固定高度把视频压到最小尺寸。
        self.outer_splitter.setStretchFactor(0, 1)
        self.outer_splitter.setStretchFactor(1, 1)
        self.outer_splitter.setSizes([540, 320])
        root.addWidget(self.outer_splitter, 1)

        # ---- 信号连接 ---------------------------------------------------
        self.control_panel.start_recognition_clicked.connect(
            self.start_realtime_recognition
        )
        self.control_panel.stop_recognition_clicked.connect(
            self.stop_realtime_recognition
        )
        self.control_panel.import_video_clicked.connect(self.import_video)
        self.control_panel.import_image_clicked.connect(self.import_image)
        self.control_panel.open_settings_clicked.connect(self.open_settings)
        self.control_panel.view_history_clicked.connect(
            lambda: self._show_bottom_tab(1)
        )
        self.control_panel.snapshot_clicked.connect(self.save_snapshot)
        self.control_panel.cancel_requested.connect(self._cancel_current_task)
        self.control_panel.refresh_cameras_requested.connect(self.refresh_cameras)
        self.control_panel.camera_changed.connect(lambda _index: self._refresh_actions())

        self.info_panel.speech_toggled.connect(self._on_speech_toggled)
        self.log_panel.status_message.connect(self._on_panel_status)
        self.history_panel.status_message.connect(self._on_panel_status)
        self.history_panel.record_selected.connect(self._on_record_selected)
        self.history_panel.record_deleted.connect(self._on_record_deleted)

        theme_manager.theme_changed.connect(self._on_theme_changed)

        self.control_panel.set_scanning(True)
        self.control_panel.set_enabled(False)
        self.control_panel.set_frame_available(False)

    def _build_header(self) -> QWidget:
        header = Card(margins=(16, 10, 16, 10), spacing=0)
        row = QHBoxLayout()
        row.setSpacing(12)

        icon_label = QLabel()
        pixmap = QPixmap(app_icon_path())
        if not pixmap.isNull():
            icon_label.setPixmap(
                pixmap.scaled(
                    38,
                    38,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        icon_label.setFixedSize(38, 38)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel(APP_NAME)
        title.setObjectName("BrandTitle")
        badge = make_chip(f"v{APP_VERSION}", "primary")
        title_row.addWidget(title)
        title_row.addWidget(badge)
        title_row.addStretch(1)
        subtitle = make_label(
            "姿态识别 · 行为解读 · 情绪语音 · 分析记录", "caption"
        )
        title_box.addLayout(title_row)
        title_box.addWidget(subtitle)

        self._theme_btn = QPushButton()
        self._theme_btn.setProperty("variant", "ghost")
        self._theme_btn.setFixedHeight(32)
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.clicked.connect(lambda: theme_manager.toggle())

        help_btn = QPushButton("使用指南")
        help_btn.setProperty("variant", "ghost")
        help_btn.setFixedHeight(32)
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        help_btn.clicked.connect(self.show_help)

        settings_btn = QPushButton("设置")
        settings_btn.setProperty("variant", "soft")
        settings_btn.setFixedHeight(32)
        settings_btn.setMinimumWidth(74)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(self.open_settings)

        row.addWidget(icon_label)
        row.addLayout(title_box, 1)
        row.addWidget(self._theme_btn)
        row.addWidget(help_btn)
        row.addWidget(settings_btn)
        header.body().addLayout(row)
        return header

    def _init_status_bar(self) -> None:
        status_bar = QStatusBar()
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

        self.status_chip = QLabel("启动中")
        self.status_chip.setObjectName("StatusChip")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setMinimumWidth(76)

        self.status_message_label = QLabel("正在启动…")
        self.status_message_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.status_message_label.setMinimumWidth(0)

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setTextVisible(False)
        self.status_progress.setFixedWidth(180)
        self.status_progress.hide()

        self.status_cancel_btn = QPushButton("取消")
        self.status_cancel_btn.setProperty("variant", "ghost")
        self.status_cancel_btn.setFixedHeight(22)
        self.status_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_cancel_btn.clicked.connect(self._cancel_current_task)
        self.status_cancel_btn.hide()

        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setProperty("role", "caption")

        status_bar.addWidget(self.status_chip)
        status_bar.addWidget(self.status_message_label, 1)
        status_bar.addPermanentWidget(self.status_progress)
        status_bar.addPermanentWidget(self.status_cancel_btn)
        status_bar.addPermanentWidget(version_label)

        self._set_status("启动中", "warning")
        self._set_message("正在初始化界面…")

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        self.import_video_action = QAction("导入视频…", self)
        self.import_video_action.setShortcut("Ctrl+O")
        self.import_video_action.setStatusTip("选择视频文件进行分析")
        self.import_video_action.triggered.connect(self.import_video)
        file_menu.addAction(self.import_video_action)

        self.import_image_action = QAction("导入图片…", self)
        self.import_image_action.setShortcut("Ctrl+I")
        self.import_image_action.setStatusTip("选择图片文件进行分析")
        self.import_image_action.triggered.connect(self.import_image)
        file_menu.addAction(self.import_image_action)

        self.snapshot_action = QAction("保存当前画面…", self)
        self.snapshot_action.setShortcut("Ctrl+S")
        self.snapshot_action.setStatusTip("将当前画面保存为图片")
        self.snapshot_action.triggered.connect(self.save_snapshot)
        file_menu.addAction(self.snapshot_action)

        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("最近打开")
        self.recent_menu.setToolTipsVisible(True)
        self._rebuild_recent_menu()

        open_data_action = QAction("打开数据目录", self)
        open_data_action.triggered.connect(lambda: open_path(DATA_DIR))
        file_menu.addAction(open_data_action)

        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        analysis_menu = menubar.addMenu("分析")
        self.start_action = QAction("开始实时识别", self)
        self.start_action.setShortcut("Ctrl+R")
        self.start_action.triggered.connect(self.start_realtime_recognition)
        analysis_menu.addAction(self.start_action)

        self.stop_action = QAction("停止实时识别", self)
        self.stop_action.setShortcut("Ctrl+T")
        self.stop_action.triggered.connect(self.stop_realtime_recognition)
        analysis_menu.addAction(self.stop_action)

        analysis_menu.addSeparator()
        self.refresh_cameras_action = QAction("重新检测摄像头", self)
        self.refresh_cameras_action.setShortcut("F6")
        self.refresh_cameras_action.triggered.connect(self.refresh_cameras)
        analysis_menu.addAction(self.refresh_cameras_action)

        view_menu = menubar.addMenu("视图")
        log_tab_action = QAction("运行日志", self)
        log_tab_action.setShortcut("Ctrl+1")
        log_tab_action.triggered.connect(lambda: self._show_bottom_tab(0))
        view_menu.addAction(log_tab_action)

        history_tab_action = QAction("历史记录", self)
        history_tab_action.setShortcut("Ctrl+2")
        history_tab_action.triggered.connect(lambda: self._show_bottom_tab(1))
        view_menu.addAction(history_tab_action)

        self.refresh_history_action = QAction("刷新历史记录", self)
        self.refresh_history_action.setShortcut("F5")
        self.refresh_history_action.triggered.connect(self._refresh_history)
        view_menu.addAction(self.refresh_history_action)

        self.bottom_action = QAction("显示底部面板", self)
        self.bottom_action.setShortcut("Ctrl+L")
        self.bottom_action.setCheckable(True)
        self.bottom_action.setChecked(True)
        self.bottom_action.triggered.connect(self._set_bottom_visible)
        view_menu.addAction(self.bottom_action)
        view_menu.addSeparator()

        theme_menu = view_menu.addMenu("主题")
        self._theme_actions: Dict[str, QAction] = {}
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for theme, label in (("light", "浅色主题"), ("dark", "深色主题")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=theme: theme_manager.set_theme(value)
            )
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme] = action

        toggle_theme_action = QAction("切换主题", self)
        toggle_theme_action.setShortcut("Ctrl+Shift+D")
        toggle_theme_action.triggered.connect(lambda: theme_manager.toggle())
        view_menu.addAction(toggle_theme_action)

        speech_menu_action = QAction("语音播报", self)
        speech_menu_action.setShortcut("Ctrl+M")
        speech_menu_action.setCheckable(True)
        speech_menu_action.setChecked(self._speech_enabled)
        speech_menu_action.toggled.connect(self._on_speech_toggled)
        view_menu.addAction(speech_menu_action)
        self.speech_action = speech_menu_action

        settings_menu = menubar.addMenu("设置")
        settings_action = QAction("偏好设置…", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)

        welcome_action = QAction("显示欢迎页", self)
        welcome_action.triggered.connect(lambda: self.show_welcome(remember=False))
        settings_menu.addAction(welcome_action)

        help_menu = menubar.addMenu("帮助")
        guide_action = QAction("使用指南", self)
        guide_action.setShortcut("F1")
        guide_action.triggered.connect(self.show_help)
        help_menu.addAction(guide_action)

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------ 主题/状态
    def _on_theme_changed(self, theme: str) -> None:
        if hasattr(self, "_theme_btn"):
            if theme == DARK:
                self._theme_btn.setText("浅色主题")
                self._theme_btn.setToolTip("切换到浅色主题（Ctrl+Shift+D）")
            else:
                self._theme_btn.setText("深色主题")
                self._theme_btn.setToolTip("切换到深色主题（Ctrl+Shift+D）")
        if hasattr(self, "_theme_actions"):
            for value, action in self._theme_actions.items():
                action.blockSignals(True)
                action.setChecked(value == theme)
                action.blockSignals(False)

    def _set_status(self, text: str, tone: str = "") -> None:
        if not hasattr(self, "status_chip"):
            return
        self.status_chip.setText(text)
        self.status_chip.setProperty("tone", tone)
        repolish(self.status_chip)

    def _set_message(self, text: str) -> None:
        if not hasattr(self, "status_message_label"):
            return
        self.status_message_label.setText(text)
        self.status_message_label.setToolTip(text)

    def _toast(self, text: str) -> None:
        if hasattr(self, "toast"):
            self.toast.show_message(text)

    def _on_panel_status(self, text: str) -> None:
        self._set_message(text)
        self._toast(text)

    # ------------------------------------------------------------- 模型加载
    def load_models_async(self) -> None:
        if self._initialized or self._loading:
            return
        self._loading = True
        self._set_status("加载模型", "warning")
        self._set_message("正在加载姿态估计与行为分类模型…")
        self.control_panel.set_enabled(False)
        self._refresh_actions()

        self.model_loader = ModelLoader(
            sequence_length=int(self.config_loader.get("recognition.sequence_length", 8))
        )
        self.model_loader.status.connect(self._set_message)
        self.model_loader.loaded.connect(self._on_models_loaded)
        self.model_loader.failed.connect(self._on_models_failed)
        self.model_loader.start()

    def _on_models_loaded(
        self,
        pose_model,
        pose_processor,
        behavior_model,
        behavior_classifier,
    ) -> None:
        self.pose_model = pose_model
        self.pose_processor = pose_processor
        self.behavior_model = behavior_model
        self.behavior_classifier = behavior_classifier

        from core.file_analyzer import FileAnalyzer, FrameAnalyzer

        confidence = float(
            self.config_loader.get("recognition.confidence_threshold", 0.75)
        )
        self.frame_analyzer = FrameAnalyzer(
            pose_processor, behavior_classifier, self.intent_mapper, confidence
        )
        self.file_analyzer = FileAnalyzer(
            pose_processor, behavior_classifier, self.intent_mapper, confidence
        )

        self._initialized = True
        self._loading = False
        self.control_panel.set_enabled(True)
        self._set_status("就绪", "success")
        self._set_message("就绪 · 选择摄像头开始识别，或导入视频 / 图片")
        self.info_panel.set_session_state("等待视频源")
        self.info_panel.set_speech_enabled(self._speech_enabled)

        model_msg = ""
        try:
            info = pose_processor.get_model_info()
            model_msg = (
                f"检测器：{info.get('detector', '无')} · "
                f"姿态模型：{info.get('pose_model', '几何模式')}"
            )
        except Exception:
            model_msg = "姿态模型已加载"
        self.log_panel.append_log("模型加载完成", "success")
        self.log_panel.append_log(model_msg, "info")
        self._toast("模型已就绪，可以开始分析")
        self._refresh_actions()
        self.logger.info("Application initialized successfully")

    def _on_models_failed(self, message: str) -> None:
        self._loading = False
        self._set_status("加载失败", "danger")
        self._set_message("模型加载失败 · 请检查依赖与模型文件")
        self.log_panel.append_log(f"模型加载失败：{message}", "error")
        self._refresh_actions()
        QMessageBox.critical(
            self,
            "初始化失败",
            f"模型加载失败：\n\n{message}\n\n"
            "请确认已安装 PyTorch、ultralytics 等依赖，且模型文件完整。",
        )

    # ---------------------------------------------------------------- 摄像头
    def _start_camera_scan(self) -> None:
        if self._camera_scanning:
            return
        self._camera_scanning = True
        self.control_panel.set_scanning(True)
        self.camera_worker = CameraScanWorker(max_devices=6)
        self.camera_worker.found.connect(self._on_cameras_found)
        self.camera_worker.start()

    def _on_cameras_found(self, cameras: list) -> None:
        self._camera_scanning = False
        self.control_panel.set_cameras(cameras)
        if cameras:
            self.log_panel.append_log(
                "检测到摄像头：" + "、".join(f"摄像头 {index}" for index in cameras),
                "success",
            )
            if self._initialized:
                self._set_message("摄像头已就绪，点击「开始识别」")
        else:
            self.log_panel.append_log(
                "未检测到摄像头，仍可导入视频或图片进行分析", "warning"
            )
            if self._initialized:
                self._set_message("未检测到摄像头 · 可导入视频或图片")
        self._refresh_actions()

    def refresh_cameras(self) -> None:
        if self.control_panel.is_recognizing:
            self._toast("请先停止实时识别，再重新检测摄像头")
            return
        if self._busy_kind:
            self._toast("当前有分析任务正在进行")
            return
        self._start_camera_scan()
        self._set_message("正在重新检测摄像头…")

    # -------------------------------------------------------------- 实时识别
    def start_realtime_recognition(self) -> None:
        if not self._initialized:
            self._toast("模型正在加载，请稍候…")
            return
        if self._busy_kind:
            self._toast("当前有分析任务正在进行")
            return
        if self.control_panel.is_recognizing:
            return

        camera_index = self.control_panel.get_selected_camera()
        if camera_index < 0:
            QMessageBox.warning(
                self,
                "没有可用摄像头",
                "未检测到可用摄像头。\n\n"
                "请连接摄像头后点击操作栏的「刷新」，\n"
                "或直接导入视频 / 图片进行分析。",
            )
            return

        target_fps = int(self.config_loader.get("recognition.target_fps", 15))
        handler = VideoStreamHandler(source=camera_index, target_fps=target_fps)
        if not handler.start():
            self.log_panel.append_log("摄像头连接失败", "error")
            self._set_status("连接失败", "danger")
            self._set_message("无法打开摄像头 · 请检查设备连接与占用情况")
            QMessageBox.critical(
                self,
                "摄像头错误",
                "无法打开摄像头。\n\n请检查：\n"
                "1. 摄像头是否正确连接；\n"
                "2. 是否被其他程序占用；\n"
                "3. 系统摄像头权限是否已开启。",
            )
            return

        self.video_handler = handler
        self.frame_analyzer.reset()
        self._reset_session()
        self._session_started = time.monotonic()
        self._session_frames = 0
        self._frame_seen = False

        self.info_panel.clear()
        self.info_panel.set_session_state("实时识别中")
        self.video_panel.clear()
        self.video_panel.set_mode(f"实时 · 摄像头 {camera_index}", live=True)
        self.video_panel.set_status_text("正在读取摄像头画面…")
        self.control_panel.set_frame_available(False)

        self.current_record_id = self.record_manager.add_record(
            source_type="camera",
            source_path=f"Camera {camera_index}",
            duration=0.0,
        )

        worker = RealtimeAnalysisWorker(
            self.video_handler, self.frame_analyzer, target_fps=target_fps
        )
        worker.result_ready.connect(self._on_realtime_result)
        worker.stats.connect(self.control_panel.update_fps)
        worker.failed.connect(self._on_realtime_failed)
        worker.finished.connect(self._on_realtime_finished)
        self.realtime_worker = worker
        worker.start()

        self.control_panel.set_recognizing(True)
        self.control_panel.update_fps(0)
        self._session_timer.start()
        self._set_status("实时识别", "success")
        self._set_message("实时识别中 · 右侧查看行为与意图，点击「停止识别」结束")
        self.log_panel.append_log(
            f"开始实时识别（摄像头 {camera_index}，目标 {target_fps} FPS）", "info"
        )
        self._toast("实时识别已启动")
        self._refresh_actions()

    def _on_realtime_result(self, frame, result: Dict) -> None:
        worker = self.sender()
        if worker is not self.realtime_worker:
            return

        behavior = result.get("behavior", "")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        keypoints = result.get("keypoints", {})
        triggered = bool(result.get("triggered", False))

        self.video_panel.update_with_keypoints(
            frame, keypoints, behavior if triggered else "", confidence if triggered else 0.0
        )
        if not self._frame_seen:
            self._frame_seen = True
            self.control_panel.set_frame_available(True)
            self._refresh_actions()

        self._session_frames = worker.processed_frames
        if not triggered:
            return

        intent = result.get("intent", "")
        emotion = self.intent_mapper.get_emotion_for_behavior(behavior)
        self.info_panel.update_behavior(
            behavior, confidence, intent, emotion, state_text="实时识别中"
        )

        now = time.monotonic()
        if behavior != self._last_logged_behavior or now - self._last_logged_at >= 1.2:
            self.log_panel.append_behavior_log(behavior, confidence, intent)
            self._last_logged_behavior = behavior
            self._last_logged_at = now

        if (
            self._speech_enabled
            and intent
            and (
                behavior != self._last_spoken_behavior
                or now - self._last_spoken_at >= 4.0
            )
        ):
            self.speech_synthesizer.speak_in_thread(intent, emotion=emotion)
            self._last_spoken_behavior = behavior
            self._last_spoken_at = now

    def _on_realtime_failed(self, message: str) -> None:
        worker = self.sender()
        if worker is not self.realtime_worker:
            return
        self.log_panel.append_log(f"实时分析中断：{message}", "error")
        self.stop_realtime_recognition(silent=True)
        self._set_status("识别中断", "danger")
        self._set_message("摄像头读取中断 · 请检查设备后重试")
        self._toast("摄像头读取中断，已保存当前记录")

    def _on_realtime_finished(self) -> None:
        # 主动停止时 worker 已被清空，这里只处理非预期的结束
        if self.sender() is self.realtime_worker and self.realtime_worker is not None:
            self.stop_realtime_recognition(silent=True)

    def stop_realtime_recognition(self, silent: bool = False) -> None:
        worker = self.realtime_worker
        if worker is None and self.video_handler is None:
            return

        self._session_timer.stop()
        self.control_panel.set_recognizing(False)
        self.control_panel.update_fps(0)

        results: List[Dict] = []
        if worker is not None:
            self.realtime_worker = None
            worker.stop()
            results = list(worker.results)
        if self.video_handler is not None:
            self.video_handler.stop()
            self.video_handler = None

        elapsed = (
            time.monotonic() - self._session_started if self._session_started else 0.0
        )
        self._session_started = 0.0
        self._session_results = results
        self.info_panel.update_session_stats(
            max(self._session_frames, worker.processed_frames if worker else 0),
            len(results),
            elapsed,
        )
        self.info_panel.set_session_state("已停止")
        self._finalize_session_record(results, elapsed)
        self.video_panel.set_mode("实时 · 已停止")
        self.video_panel.set_status_text("再次点击「开始识别」可继续分析")
        self._set_status("就绪", "success")
        if not silent:
            self._set_message(
                f"已停止实时识别 · 识别 {len(results)} 次 · 记录已保存"
            )
            self.log_panel.append_log(
                f"实时识别结束，共识别 {len(results)} 次，记录已保存", "success"
            )
            self._log_breakdown(results)
            self._toast("识别已停止，记录已保存")
        else:
            self._set_message(
                f"实时识别已结束 · 识别 {len(results)} 次 · 记录已保存"
            )
        self._refresh_actions()

    # ----------------------------------------------------------------- 视频
    def import_video(self) -> None:
        if not self._ensure_file_analysis_ready():
            return
        path = self._choose_file(
            "选择视频文件",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.webm);;所有文件 (*)",
        )
        if path:
            self.analyze_video(path)

    def analyze_video(self, path: str) -> None:
        if not self._ensure_file_analysis_ready():
            return
        if self.file_analyzer is None:
            return

        file_path = str(Path(path))
        self._current_media_path = file_path
        self._remember_recent(file_path)
        name = Path(file_path).name

        self.info_panel.clear()
        self.info_panel.set_session_state("视频分析")
        self.video_panel.clear()
        self.video_panel.set_mode("视频分析")
        self.video_panel.set_status_text(f"正在分析 {name}")
        self.video_panel.set_progress(0, "准备分析…")

        self._begin_busy("video", f"视频分析中 · {name}", indeterminate=False)
        self._set_status("视频分析", "warning")
        self.log_panel.append_log(f"开始分析视频：{name}", "info")

        confidence = float(
            self.config_loader.get("recognition.confidence_threshold", 0.75)
        )
        worker = VideoAnalysisWorker(
            file_path,
            self.file_analyzer,
            confidence_threshold=confidence,
            analysis_fps=float(self.config_loader.get("recognition.analysis_fps", 3.0)),
            max_frames=int(
                self.config_loader.get("recognition.max_analysis_frames", 1500)
            ),
        )
        worker.progress.connect(self._on_video_progress)
        worker.preview.connect(self._on_video_preview)
        worker.finished.connect(self._on_video_finished)
        worker.failed.connect(self._on_video_failed)
        worker.cancelled.connect(self._on_video_cancelled)
        self.video_worker = worker
        worker.start()

    def _on_video_progress(self, percent: int, text: str) -> None:
        if self.sender() is not self.video_worker:
            return
        self.status_progress.setValue(percent)
        self.video_panel.set_progress(percent, text)
        self._set_message(text)

    def _on_video_preview(self, frame, result: Dict) -> None:
        if self.sender() is not self.video_worker:
            return
        behavior = result.get("behavior", "")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        self.video_panel.update_with_keypoints(
            frame, result.get("keypoints", {}), behavior, confidence
        )
        if behavior not in {"", "idle", "error"}:
            emotion = self.intent_mapper.get_emotion_for_behavior(behavior)
            self.info_panel.update_behavior(
                behavior,
                confidence,
                result.get("intent", ""),
                emotion,
                state_text="视频分析",
            )

    def _on_video_finished(self, results: list, payload: Dict) -> None:
        worker = self.sender()
        if worker is not self.video_worker:
            return
        self.video_worker = None
        self._end_busy()

        results = list(results or [])
        payload = payload or {}
        source_path = payload.get("source_path", "")
        duration = float(payload.get("duration", 0.0) or 0.0)
        analyzed = int(payload.get("analyzed_frames", 0) or 0)
        summary = payload.get("summary", "") or self._summarize(results)

        self._session_results = results
        self._session_frames = analyzed
        self.info_panel.update_session_stats(analyzed, len(results), duration)
        self.info_panel.set_session_state("视频分析")

        top = self._top_result(results)
        if top:
            emotion = self.intent_mapper.get_emotion_for_behavior(top["behavior"])
            self.info_panel.update_behavior(
                top["behavior"],
                top["confidence"],
                summary or top.get("intent", ""),
                emotion,
                state_text="视频分析",
            )

        record_id = self.record_manager.add_record(
            source_type="video",
            source_path=source_path,
            duration=duration,
            behaviors=results,
            summary=summary,
        )
        self.history_panel.reload()

        self.video_panel.set_progress(None)
        self.video_panel.set_mode("视频 · 分析完成")
        self.video_panel.set_status_text(
            f"采样 {analyzed} 帧 · 识别 {len(results)} 次"
        )
        self._set_status("就绪", "success")
        self._set_message(f"视频分析完成 · 识别 {len(results)} 次 · 记录已保存")
        self.log_panel.append_log(
            f"视频分析完成：采样 {analyzed} 帧，识别 {len(results)} 次", "success"
        )
        self._log_breakdown(results)
        self._toast(
            f"视频分析完成：识别 {len(results)} 次"
            + ("，记录已保存" if record_id else "")
        )
        self._refresh_actions()

    def _on_video_failed(self, message: str) -> None:
        if self.sender() is not self.video_worker:
            return
        self.video_worker = None
        self._end_busy()
        self.video_panel.set_progress(None)
        self.video_panel.set_mode("视频 · 分析失败")
        self.video_panel.set_status_text("无法完成视频分析")
        self.info_panel.set_session_state("视频分析失败")
        self._set_status("分析失败", "danger")
        self._set_message("视频分析失败 · 请检查文件格式")
        self.log_panel.append_log(f"视频分析失败：{message}", "error")
        QMessageBox.warning(self, "视频分析失败", message)
        self._refresh_actions()

    def _on_video_cancelled(self) -> None:
        if self.sender() is not self.video_worker:
            return
        self.video_worker = None
        self._end_busy()
        self.video_panel.set_progress(None)
        self.video_panel.set_mode("视频 · 已取消")
        self.video_panel.set_status_text("分析已取消，未保存记录")
        self.info_panel.set_session_state("已取消")
        self._set_status("已取消", "warning")
        self._set_message("已取消视频分析")
        self.log_panel.append_log("用户取消了视频分析", "warning")
        self._toast("已取消视频分析")
        self._refresh_actions()

    # ----------------------------------------------------------------- 图片
    def import_image(self) -> None:
        if not self._ensure_file_analysis_ready():
            return
        path = self._choose_file(
            "选择图片文件",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
        )
        if path:
            self.analyze_image(path)

    def analyze_image(self, path: str) -> None:
        if not self._ensure_file_analysis_ready():
            return
        if self.file_analyzer is None:
            return

        file_path = str(Path(path))
        self._current_media_path = file_path
        self._remember_recent(file_path)
        name = Path(file_path).name

        self.info_panel.clear()
        self.info_panel.set_session_state("图片分析")
        self.video_panel.clear()
        self.video_panel.set_mode("图片分析")
        self.video_panel.set_status_text(f"正在分析 {name}")

        self._begin_busy("image", f"图片分析中 · {name}", indeterminate=True)
        self._set_status("图片分析", "warning")
        self.log_panel.append_log(f"开始分析图片：{name}", "info")

        worker = ImageAnalysisWorker(file_path, self.file_analyzer)
        worker.finished.connect(self._on_image_finished)
        worker.failed.connect(self._on_image_failed)
        self.image_worker = worker
        worker.start()

    def _on_image_finished(self, payload: Dict) -> None:
        if self.sender() is not self.image_worker:
            return
        self.image_worker = None
        self._end_busy()

        payload = payload or {}
        image = payload.get("image")
        result = payload.get("result", {}) or {}
        behavior = result.get("behavior", "")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        intent = result.get("intent", "")
        valid = behavior not in {"", "idle", "error"}

        if image is not None:
            self.video_panel.update_with_keypoints(
                image, result.get("keypoints", {}), behavior, confidence
            )
        self.video_panel.set_mode("图片 · 分析完成")
        self.video_panel.set_status_text(
            f"{Path(self._current_media_path).name or '图片'} · "
            f"{BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior) if valid else '未识别到明确行为'}"
        )

        if valid:
            emotion = self.intent_mapper.get_emotion_for_behavior(behavior)
            self.info_panel.update_behavior(
                behavior,
                confidence,
                intent,
                emotion,
                state_text="图片分析",
            )
            self.log_panel.append_behavior_log(behavior, confidence, intent)
            if self._speech_enabled and intent:
                self.speech_synthesizer.speak_in_thread(intent, emotion=emotion)
            self.record_manager.add_record(
                source_type="image",
                source_path=self._current_media_path,
                duration=0.0,
                behaviors=[
                    {
                        "timestamp": 0,
                        "behavior": behavior,
                        "confidence": confidence,
                        "intent": intent,
                    }
                ],
                summary=intent,
            )
            self.history_panel.reload()
            self.info_panel.update_session_stats(1, 1, 0.0)
            self.info_panel.set_session_state("图片分析")
            self._set_status("就绪", "success")
            self._set_message(
                f"图片分析完成 · {BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)} "
                f"· 置信度 {confidence:.0%}"
            )
            self._toast(
                f"图片分析完成：{BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)}"
            )
        else:
            self.info_panel.update_session_stats(1, 0, 0.0)
            self.info_panel.set_session_state("图片分析")
            self._set_status("未识别", "warning")
            self._set_message("图片分析完成，但未识别到明确行为")
            self.log_panel.append_log("图片分析完成，未识别到明确行为", "warning")
            self._toast("未识别到明确行为，可尝试更清晰的图片")

        self.log_panel.append_log("图片分析完成", "success")
        self._refresh_actions()

    def _on_image_failed(self, message: str) -> None:
        if self.sender() is not self.image_worker:
            return
        self.image_worker = None
        self._end_busy()
        self.video_panel.set_mode("图片 · 分析失败")
        self.video_panel.set_status_text("无法完成图片分析")
        self._set_status("分析失败", "danger")
        self._set_message("图片分析失败 · 请检查文件格式")
        self.log_panel.append_log(f"图片分析失败：{message}", "error")
        QMessageBox.warning(self, "图片分析失败", message)
        self._refresh_actions()

    # ------------------------------------------------------------ 任务与状态
    def _begin_busy(
        self, kind: str, text: str, indeterminate: bool = False
    ) -> None:
        self._busy_kind = kind
        self.control_panel.set_busy(True, text, cancelable=(kind == "video"))
        if indeterminate:
            self.status_progress.setRange(0, 0)
        else:
            self.status_progress.setRange(0, 100)
            self.status_progress.setValue(0)
        self.status_progress.setVisible(True)
        self.status_cancel_btn.setText("取消")
        self.status_cancel_btn.setEnabled(True)
        self.status_cancel_btn.setVisible(kind == "video")
        self._set_message(text)
        self._refresh_actions()

    def _end_busy(self) -> None:
        self._busy_kind = None
        self.control_panel.set_busy(False)
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.hide()
        self.status_cancel_btn.hide()
        self.status_cancel_btn.setText("取消")
        self.status_cancel_btn.setEnabled(True)
        self._refresh_actions()

    def _cancel_current_task(self) -> None:
        if self.video_worker is not None:
            self.video_worker.cancel()
            self.status_cancel_btn.setText("正在取消…")
            self.status_cancel_btn.setEnabled(False)
            self._set_message("正在取消视频分析…")
            self.control_panel.set_busy(True, "正在取消…", cancelable=False)
            return
        if self.image_worker is not None:
            self._toast("图片分析即将完成，请稍候")
            return
        if self.realtime_worker is not None:
            self.stop_realtime_recognition()

    def _ensure_file_analysis_ready(self) -> bool:
        if not self._initialized:
            self._toast("模型正在加载，请稍候…")
            return False
        if self._busy_kind:
            self._toast("已有分析任务正在进行")
            return False
        if self.control_panel.is_recognizing:
            self._toast("请先停止实时识别，再导入文件")
            return False
        return True

    def _refresh_actions(self) -> None:
        if not hasattr(self, "start_action"):
            return
        recognizing = self.control_panel.is_recognizing
        can_analyze = self._initialized and self._busy_kind is None
        self.start_action.setEnabled(can_analyze and not recognizing)
        self.stop_action.setEnabled(recognizing)
        self.import_video_action.setEnabled(can_analyze and not recognizing)
        self.import_image_action.setEnabled(can_analyze and not recognizing)
        self.snapshot_action.setEnabled(
            self.video_panel.has_frame() and self._busy_kind is None
        )
        self.refresh_cameras_action.setEnabled(can_analyze and not recognizing)
        self.refresh_history_action.setEnabled(self._busy_kind is None)

    def _update_session_tick(self) -> None:
        if self.realtime_worker is None:
            return
        elapsed = (
            time.monotonic() - self._session_started if self._session_started else 0.0
        )
        self.info_panel.update_session_stats(
            self.realtime_worker.processed_frames,
            len(self.realtime_worker.results),
            elapsed,
        )

    def _reset_session(self) -> None:
        self._session_results = []
        self._session_frames = 0
        self._last_logged_behavior = None
        self._last_logged_at = 0.0
        self._last_spoken_behavior = None
        self._last_spoken_at = 0.0

    def _finalize_session_record(self, results: List[Dict], duration: float) -> None:
        if not self.current_record_id:
            return
        summary = self._summarize(results)
        self.record_manager.update_record(
            self.current_record_id, results, summary, duration=duration
        )
        self.current_record_id = None
        self.history_panel.reload()

    @staticmethod
    def _summarize(results: List[Dict]) -> str:
        if not results:
            return ""
        top = max(results, key=lambda item: item.get("confidence", 0.0))
        return top.get("intent", "") or ""

    @staticmethod
    def _top_result(results: List[Dict]) -> Optional[Dict]:
        valid = [
            item
            for item in results
            if item.get("behavior") not in {None, "", "idle", "error"}
        ]
        if not valid:
            return None
        return max(valid, key=lambda item: item.get("confidence", 0.0))

    def _log_breakdown(self, results: List[Dict]) -> None:
        counts = Counter(
            item.get("behavior", "")
            for item in results
            if item.get("behavior") not in {None, "", "idle", "error"}
        )
        if not counts:
            return
        parts = [
            f"{BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)}×{count}"
            for behavior, count in counts.most_common(4)
        ]
        self.log_panel.append_log("行为分布：" + " · ".join(parts), "info")

    # ------------------------------------------------------------- 截图/记录
    def save_snapshot(self) -> None:
        frame = self.video_panel.current_frame
        if frame is None:
            self._toast("当前没有可保存的画面")
            return
        last_dir = self.settings.value(
            "paths/snapshot_dir", str(Path.home() / "Pictures")
        )
        default_path = Path(last_dir) / (
            f"pet_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前画面",
            str(default_path),
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png)",
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            path += ".jpg"
            suffix = ".jpg"
        extension = ".png" if suffix == ".png" else ".jpg"
        try:
            ok, buffer = cv2.imencode(extension, frame)
            if not ok:
                raise OSError("编码失败")
            Path(path).write_bytes(buffer.tobytes())
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存画面：\n{exc}")
            return
        self.settings.setValue("paths/snapshot_dir", str(Path(path).parent))
        self.log_panel.append_log(f"画面已保存：{Path(path).name}", "success")
        self._set_message(f"画面已保存：{Path(path).name}")
        self._toast("画面已保存")

    def _on_record_selected(self, record_id: str) -> None:
        record = self.record_manager.get_record(record_id)
        if record:
            self._set_message(
                f"查看记录：{record.get('timestamp', '')} · "
                f"{record.get('source_type', '')}"
            )

    def _on_record_deleted(self, _record_id: str) -> None:
        self._set_message("记录已删除")
        self._refresh_actions()

    def _refresh_history(self) -> None:
        self.history_panel.reload()
        self._show_bottom_tab(1)
        self._set_message("历史记录已刷新")
        self._toast("历史记录已刷新")

    # ------------------------------------------------------------- 设置/帮助
    def open_settings(self) -> None:
        dialog = SettingsDialog(
            self,
            self.config_loader,
            self.speech_synthesizer,
            self.record_manager,
        )
        dialog.settings_applied.connect(self._on_settings_applied)
        dialog.exec()
        dialog.deleteLater()

    def _on_settings_applied(self) -> None:
        confidence = float(
            self.config_loader.get("recognition.confidence_threshold", 0.75)
        )
        if self.frame_analyzer is not None:
            self.frame_analyzer.confidence_threshold = confidence
        if self.file_analyzer is not None:
            self.file_analyzer.confidence_threshold = confidence

        self._speech_enabled = bool(
            self.config_loader.get("speech.enabled", True)
        )
        self.info_panel.set_speech_enabled(self._speech_enabled)
        self._sync_speech_action()
        self.history_panel.reload()
        self.log_panel.append_log("设置已更新", "info")
        self._set_message("设置已保存")
        self._toast("设置已保存")
        self._refresh_actions()

    def _on_speech_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._speech_enabled:
            self._sync_speech_action()
            return
        self._speech_enabled = enabled
        self.config_loader.update_many({"speech.enabled": enabled})
        self.info_panel.set_speech_enabled(enabled)
        if not enabled:
            self.speech_synthesizer.stop()
        self._sync_speech_action()
        self.log_panel.append_log(
            "语音播报已" + ("开启" if enabled else "关闭"), "info"
        )
        self._set_message("语音播报已" + ("开启" if enabled else "关闭"))
        self._toast("语音播报已" + ("开启" if enabled else "关闭"))

    def _sync_speech_action(self) -> None:
        if not hasattr(self, "speech_action"):
            return
        self.speech_action.blockSignals(True)
        self.speech_action.setChecked(self._speech_enabled)
        self.speech_action.blockSignals(False)
        self.info_panel.set_speech_enabled(self._speech_enabled)

    def show_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()
        dialog.deleteLater()

    def show_about(self) -> None:
        dialog = AboutDialog(self)
        dialog.exec()
        dialog.deleteLater()

    def _maybe_show_welcome(self) -> None:
        if self._force_show_welcome is False:
            return
        if self._force_show_welcome is None and not bool(
            self.config_loader.get("app.show_welcome", True)
        ):
            return
        self.show_welcome(remember=True)

    def show_welcome(self, remember: bool = True) -> None:
        dialog = WelcomeDialog(self)
        dialog.exec()
        dont_show_again = dialog.dont_show_again
        dialog.deleteLater()
        if remember:
            self.config_loader.update_many(
                {
                    "app.show_welcome": not dont_show_again,
                    "app.is_first_run": False,
                }
            )

    # --------------------------------------------------------------- 底部面板
    def _toggle_bottom(self) -> None:
        self._set_bottom_visible(self._bottom_collapsed)

    def _set_bottom_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == (not self._bottom_collapsed):
            self._sync_bottom_action(visible)
            return
        if visible:
            self._bottom_collapsed = False
            self.bottom_tabs.setMinimumHeight(190)
            self.bottom_tabs.setMaximumHeight(16777215)
            self._collapse_btn.setText("收起")
            self._collapse_btn.setToolTip("收起底部面板，扩大视频区域（Ctrl+L）")
            if self._splitter_sizes_before_collapse:
                sizes = list(self._splitter_sizes_before_collapse)
                QTimer.singleShot(0, lambda: self.outer_splitter.setSizes(sizes))
        else:
            self._splitter_sizes_before_collapse = self.outer_splitter.sizes()
            self._bottom_collapsed = True
            self.bottom_tabs.setMinimumHeight(44)
            self.bottom_tabs.setMaximumHeight(44)
            self._collapse_btn.setText("展开")
            self._collapse_btn.setToolTip("展开底部面板（Ctrl+L）")
        self._sync_bottom_action(visible)

    def _sync_bottom_action(self, visible: bool) -> None:
        if not hasattr(self, "bottom_action"):
            return
        self.bottom_action.blockSignals(True)
        self.bottom_action.setChecked(visible)
        self.bottom_action.blockSignals(False)

    def _show_bottom_tab(self, index: int) -> None:
        if self._bottom_collapsed:
            self._set_bottom_visible(True)
        self.bottom_tabs.setCurrentIndex(index)

    # -------------------------------------------------------------- 拖拽/文件
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._dragged_media(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragged_media(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        path = self._dragged_media(event)
        if not path:
            event.ignore()
            return
        event.acceptProposedAction()
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            self.analyze_image(path)
        else:
            self.analyze_video(path)

    @staticmethod
    def _dragged_media(event) -> Optional[str]:
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            suffix = Path(path).suffix.lower()
            if suffix in VIDEO_EXTENSIONS or suffix in IMAGE_EXTENSIONS:
                return path
        return None

    def _choose_file(self, caption: str, filters: str) -> str:
        last_dir = self.settings.value("paths/last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, caption, str(last_dir), filters)
        if path:
            self.settings.setValue("paths/last_dir", str(Path(path).parent))
        return path

    def _remember_recent(self, path: str) -> None:
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            resolved = path
        if resolved in self._recent_files:
            self._recent_files.remove(resolved)
        self._recent_files.insert(0, resolved)
        self._recent_files = self._recent_files[:8]
        self.settings.setValue("ui/recent_files", self._recent_files)
        self._rebuild_recent_menu()

    def _load_recent_files(self) -> List[str]:
        raw = self.settings.value("ui/recent_files", [])
        if isinstance(raw, str):
            raw = [raw]
        if not raw:
            return []
        return [str(item) for item in raw if item]

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        if not self._recent_files:
            empty = self.recent_menu.addAction("暂无最近文件")
            empty.setEnabled(False)
            return
        for path in self._recent_files:
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, value=path: self._open_recent(value)
            )

    def _open_recent(self, path: str) -> None:
        if not Path(path).exists():
            QMessageBox.warning(
                self, "文件不存在", f"文件已被移动或删除：\n{path}"
            )
            if path in self._recent_files:
                self._recent_files.remove(path)
                self.settings.setValue("ui/recent_files", self._recent_files)
                self._rebuild_recent_menu()
            return
        suffix = Path(path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            self.analyze_image(path)
        elif suffix in VIDEO_EXTENSIONS:
            self.analyze_video(path)

    # ---------------------------------------------------------------- 持久化
    def _restore_layout(self) -> None:
        geometry = self.settings.value("ui/window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        splitter_state = self.settings.value("ui/splitter_state")
        restored = False
        if splitter_state is not None:
            restored = self.outer_splitter.restoreState(splitter_state)
        if not restored:
            self.outer_splitter.setSizes([540, 320])

        collapsed = self.settings.value("ui/bottom_collapsed", False)
        if isinstance(collapsed, str):
            collapsed = collapsed.lower() == "true"
        if bool(collapsed):
            self._set_bottom_visible(False)
        else:
            self._sync_bottom_action(True)

    def _save_layout(self) -> None:
        self.settings.setValue("ui/window_geometry", self.saveGeometry())
        if not self._bottom_collapsed:
            self.settings.setValue("ui/splitter_state", self.outer_splitter.saveState())
        self.settings.setValue("ui/bottom_collapsed", self._bottom_collapsed)
        self.settings.sync()

    # ------------------------------------------------------------------ 关闭
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._session_timer.stop()
        if self.realtime_worker is not None or self.video_handler is not None:
            self.stop_realtime_recognition(silent=True)

        if self.video_worker is not None:
            self.video_worker.cancel()
        if self.model_loader is not None:
            self.model_loader.cancel()
        if self.speech_synthesizer is not None:
            self.speech_synthesizer.stop()

        self._save_layout()
        self.logger.info("Application closed")
        event.accept()
