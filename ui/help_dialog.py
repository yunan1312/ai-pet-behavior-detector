"""使用指南：侧边导航 + 富文本内容。"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui.components import Divider
from ui.theme import theme_manager, tokens
from utils.constants import APP_VERSION

SECTIONS: List[Tuple[str, str]] = [
    (
        "快速开始",
        """
        <h2>三种分析方式</h2>
        <p><b>实时识别</b>：连接摄像头后点击「开始识别」，把宠物完整放入画面即可。
        推理在后台线程运行，界面不会冻结；右侧会实时显示行为、置信度与意图。</p>
        <p><b>视频分析</b>：点击「导入视频」或按 <code>Ctrl+O</code>，支持 MP4 / AVI / MOV。
        分析过程有真实进度与取消按钮，完成后自动写入历史记录。</p>
        <p><b>图片分析</b>：点击「导入图片」或按 <code>Ctrl+I</code>，支持 JPG / PNG / BMP。</p>
        <p><b>拖拽导入</b>：把视频或图片直接拖到窗口任意位置，软件会自动识别文件类型。</p>
        <h2>推荐流程</h2>
        <ol>
          <li>首次使用打开「设置」，确认摄像头、置信度与语音选项；</li>
          <li>实时识别时观察右侧「最近行为」，确认识别结果符合预期；</li>
          <li>分析完成后在底部「历史记录」中查看详情或导出报告。</li>
        </ol>
        """,
    ),
    (
        "界面与操作",
        """
        <h2>主界面结构</h2>
        <ul>
          <li><b>视频画布</b>：显示实时画面或导入的媒体，叠加关键点、检测框、行为标签与进度。</li>
          <li><b>操作栏</b>：开始 / 停止识别、导入视频、导入图片、保存画面、摄像头选择与 FPS。</li>
          <li><b>洞察面板</b>：当前行为、置信度进度、情绪、意图解读、最近行为与会话统计。</li>
          <li><b>底部面板</b>：标签页切换「运行日志 / 历史记录」，可收起以扩大画面。</li>
        </ul>
        <h2>历史记录</h2>
        <p>支持关键词搜索、来源筛选、列排序与多选。双击任意记录可查看行为分布与时间线，
        也可以导出单条 Markdown 报告或所选记录的 CSV 汇总。</p>
        <h2>主题</h2>
        <p>右上角按钮可在浅色 / 深色主题间切换；也可以在「设置 → 外观」中修改。
        窗口大小、分栏位置和底部面板状态会自动记住。</p>
        """,
    ),
    (
        "支持的行为",
        """
        <h2>可识别的 13 种行为</h2>
        <p><b>姿态</b>：站立、坐着、趴卧、睡觉</p>
        <p><b>移动</b>：行走、奔跑、玩耍、捕猎</p>
        <p><b>放松与互动</b>：伸懒腰、理毛、踩奶、蹭物体</p>
        <p><b>默认</b>：发呆（没有明确证据时的状态）</p>
        <p>当前姿态模型只输出 17 个身体关键点，不包含耳部与尾尖，「缓慢眨眼 / 飞机耳 / 尾巴竖起 / 尾巴甩动 / 露肚皮」已不再自动识别。
        判定基于按体长归一化的几何特征与真实时间戳速度，换分辨率或帧率不影响结果；姿态标签需连续 2 帧一致才切换，避免闪烁。</p>
        <h2>意图与情绪</h2>
        <p>每种行为都会映射为一句自然的意图解读，并附带情绪标签（开心、放松、紧张等）。
        可在 <code>config/intent_mapping.yaml</code> 中自定义文案。</p>
        """,
    ),
    (
        "快捷键",
        """
        <h2>常用快捷键</h2>
        <table cellspacing="0" cellpadding="6" width="100%">
          <tr><td><b>Ctrl + O</b></td><td>导入视频</td></tr>
          <tr><td><b>Ctrl + I</b></td><td>导入图片</td></tr>
          <tr><td><b>Ctrl + S</b></td><td>保存当前画面</td></tr>
          <tr><td><b>Ctrl + R</b></td><td>开始实时识别</td></tr>
          <tr><td><b>Ctrl + T</b></td><td>停止实时识别</td></tr>
          <tr><td><b>Ctrl + M</b></td><td>开启 / 关闭语音播报</td></tr>
          <tr><td><b>Ctrl + L</b></td><td>收起 / 展开底部面板</td></tr>
          <tr><td><b>Ctrl + ,</b></td><td>打开设置</td></tr>
          <tr><td><b>F1</b></td><td>使用指南</td></tr>
          <tr><td><b>F5</b></td><td>刷新历史记录</td></tr>
          <tr><td><b>Ctrl + Q</b></td><td>退出程序</td></tr>
        </table>
        """,
    ),
    (
        "常见问题",
        """
        <h2>摄像头无法打开？</h2>
        <p>确认设备已连接且未被其他程序占用；点击操作栏的「刷新」重新检测。
        若系统隐私设置禁止应用访问摄像头，请在系统设置中授权。</p>
        <h2>分析结果不准确？</h2>
        <p>在设置中调整置信度阈值；确保光线充足、宠物完整入镜、背景尽量简洁。
        单张图片分析只使用一张画面，动作类行为建议使用视频或实时识别。</p>
        <h2>语音没有声音？</h2>
        <p>检查系统音量与设置中的音量；Edge TTS 需要联网。右侧「意图解读」卡片上的
        语音开关可以快速静音。</p>
        <h2>分析很慢或界面卡顿？</h2>
        <p>姿态估计依赖 CPU/GPU 性能。视频分析会自动按时间采样并限制最大分析帧数；
        实时识别在后台线程运行，FPS 会显示在操作栏右侧。</p>
        <h2>数据保存在哪里？</h2>
        <p>历史记录位于 <code>data/records.db</code>，日志位于 <code>data/app.log</code>。
        可在「关于」或「设置 → 存储」中打开数据目录。</p>
        """,
    ),
]


def _build_html(title: str, body: str) -> str:
    p = tokens()
    return f"""
    <html><body style="font-family:'Microsoft YaHei UI','Segoe UI',sans-serif;
        font-size:13px; color:{p['text']}; line-height:1.75;">
      <h1 style="font-size:22px; color:{p['text']}; margin:2px 0 18px 0;
        padding-bottom:10px; border-bottom:1px solid {p['border']};">{title}</h1>
      <style>
        h2 {{ font-size:16px; color:{p['text']}; margin:16px 0 8px 0; }}
        p {{ margin:8px 0; color:{p['text_secondary']}; }}
        li {{ margin:6px 0; color:{p['text_secondary']}; }}
        ul {{ margin:8px 0; padding-left:20px; }}
        b {{ color:{p['text']}; }}
        code {{ background:{p['surface_alt']}; color:{p['primary']};
                padding:1px 5px; border-radius:4px; font-family:Consolas,monospace; }}
        table {{ border-collapse:collapse; margin-top:6px; }}
        td {{ padding:6px 10px; border-bottom:1px solid {p['border']};
              color:{p['text_secondary']}; }}
      </style>
      {body}
    </body></html>
    """


class HelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("使用指南")
        self.setModal(True)
        self.resize(820, 640)
        self.setMinimumSize(720, 540)
        self.init_ui()
        theme_manager.theme_changed.connect(self._render_current)

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setFixedWidth(168)
        for title, _body in SECTIONS:
            self.nav.addItem(title)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._render_current)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)

        content_row.addWidget(self.nav)
        content_row.addWidget(self.browser, 1)
        layout.addLayout(content_row, 1)
        layout.addWidget(Divider())

        footer = QHBoxLayout()
        version = QPushButton(f"版本 v{APP_VERSION}")
        version.setProperty("variant", "ghost")
        version.setEnabled(False)
        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "primary")
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(version)
        footer.addStretch(1)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

        self._render_current()

    def _render_current(self, *_args) -> None:
        row = max(0, self.nav.currentRow())
        if row >= len(SECTIONS):
            row = 0
        title, body = SECTIONS[row]
        self.nav.setCurrentRow(row)
        self.browser.setHtml(_build_html(title, body))
