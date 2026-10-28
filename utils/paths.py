"""项目内路径解析与常用系统操作。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = DATA_DIR


def asset_path(name: str) -> Path:
    return RESOURCES_DIR / name


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def app_icon_path() -> str:
    path = asset_path("app_icon.png")
    return str(path) if path.exists() else ""


def open_path(path: str | Path) -> bool:
    """使用系统默认程序打开文件或目录。"""
    target = str(path)
    if not os.path.exists(target):
        return False
    try:
        if sys.platform == "win32":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
        return True
    except Exception:
        return False
