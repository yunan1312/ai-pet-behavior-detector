from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsOpacityEffect, QWidget
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont


class IntentDialog(QDialog):
    def __init__(self, parent=None, behavior="", intent="", confidence=0.0):
        super().__init__(parent)
        self.behavior = behavior
        self.intent = intent
        self.confidence = confidence
        self.auto_close = True
        self.close_timer = QTimer()
        self.close_timer.timeout.connect(self.accept)
        self.init_ui()
        self.setup_animation()

    def init_ui(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(400, 200)
        self.setMaximumSize(600, 300)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.title_label = QLabel("🐾 行为意图解读")
        self.title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #ffffff;")

        self.behavior_label = QLabel(f"行为: {self._get_behavior_display_name(self.behavior)}")
        self.behavior_label.setFont(QFont("Microsoft YaHei", 12))
        self.behavior_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.behavior_label.setStyleSheet("color: #e0e0e0;")

        self.confidence_label = QLabel(f"置信度: {self.confidence:.1%}")
        self.confidence_label.setFont(QFont("Microsoft YaHei", 10))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("color: #b0b0b0;")

        self.intent_label = QLabel(self.intent)
        self.intent_label.setFont(QFont("Microsoft YaHei", 14))
        self.intent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.intent_label.setWordWrap(True)
        self.intent_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 15px;
            }
        """)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("关闭")
        close_btn.setFont(QFont("Microsoft YaHei", 11))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 5px;
                padding: 8px 25px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)

        button_layout.addWidget(close_btn)
        button_layout.addStretch()

        layout.addWidget(self.title_label)
        layout.addWidget(self.behavior_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.intent_label)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

        if self.auto_close:
            self.close_timer.start(5000)

    def _get_behavior_display_name(self, behavior: str) -> str:
        from utils.constants import BEHAVIOR_DISPLAY_NAMES
        return BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)

    def setup_animation(self):
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        self.fade_in_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_in_animation.start()

    def set_auto_close(self, enabled: bool, duration: int = 5000):
        self.auto_close = enabled
        if enabled:
            self.close_timer.start(duration)
        else:
            self.close_timer.stop()

    def update_content(self, behavior: str, intent: str, confidence: float):
        self.behavior = behavior
        self.intent = intent
        self.confidence = confidence

        self.behavior_label.setText(f"行为: {self._get_behavior_display_name(self.behavior)}")
        self.confidence_label.setText(f"置信度: {self.confidence:.1%}")
        self.intent_label.setText(self.intent)

        if self.auto_close:
            self.close_timer.start(5000)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
