"""日志工具模块 - 单例模式，支持文件和控制台输出

使用 RotatingFileHandler 实现日志文件自动轮转，
防止单个日志文件过大。
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    _instance: Optional['Logger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, 'data')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')

        self._logger = logging.getLogger('PetBehaviorAnalyzer')
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            # 使用 RotatingFileHandler，单文件最大 5MB，保留 3 个备份
            fh = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
            )
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def set_level(self, level: str):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        self._logger.setLevel(log_level)

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def log_behavior(self, behavior: str, confidence: float, intent: str):
        self._logger.info(
            f"BEHAVIOR_DETECTED | behavior={behavior} | confidence={confidence:.2f} | intent={intent}"
        )

    def log_speech(self, text: str, language: str, voice: str, emotion: str = "neutral"):
        self._logger.info(
            f"SPEECH_SYNTHESIZE | language={language} | voice={voice} | emotion={emotion} | text={text[:50]}..."
        )


def get_logger() -> Logger:
    return Logger()
