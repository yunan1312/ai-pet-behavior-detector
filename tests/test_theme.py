"""设计系统与主题样式表的回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ui.theme import DARK, LIGHT, PALETTES, build_stylesheet, tokens
from utils.paths import RESOURCES_DIR

REQUIRED_TOKENS = {
    "bg",
    "surface",
    "border",
    "text",
    "text_secondary",
    "primary",
    "success",
    "danger",
    "warning",
    "video_bg",
    "hero_from",
    "hero_to",
}


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_stylesheet_has_no_unresolved_placeholders(theme):
    stylesheet = build_stylesheet(theme)
    assert "$" not in stylesheet
    assert "QPushButton" in stylesheet
    assert "QFrame#InfoHero" in stylesheet


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_theme_assets_exist_and_use_plain_paths(theme):
    stylesheet = build_stylesheet(theme)
    chevron = RESOURCES_DIR / f"chevron_{theme}.png"
    check = RESOURCES_DIR / "check_white.png"
    assert chevron.exists()
    assert check.exists()
    # QSS 不解析 file:/// 与百分号编码，必须使用正斜杠本地路径
    assert str(chevron).replace("\\", "/") in stylesheet
    assert "file://" not in stylesheet
    assert "%" not in stylesheet


def test_palettes_are_complete_and_distinct():
    for theme, palette in PALETTES.items():
        assert REQUIRED_TOKENS <= set(palette), theme
    assert PALETTES[LIGHT]["bg"] != PALETTES[DARK]["bg"]
    assert tokens(LIGHT)["surface"] == "#FFFFFF"


def test_theme_names_are_lowercase_constants():
    assert LIGHT == "light"
    assert DARK == "dark"


def test_theme_manager_starts_light_every_session():
    from ui.theme import ThemeManager

    manager = ThemeManager()
    assert manager.theme == LIGHT

    manager.set_theme(DARK)
    assert manager.theme == DARK

    # 新实例模拟下一次启动，应重新回到浅色
    assert ThemeManager().theme == LIGHT
