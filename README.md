# 宠物行为智能分析软件

一款结合计算机视觉与语音交互技术的创新应用，能够实时分析宠物的行为动作，并用自然语言解读其可能的「意图表达」。

## 功能特点

- **实时视频分析** - 通过摄像头实时捕捉并分析宠物行为
- **视频文件分析** - 导入 MP4/AVI/MOV 格式视频进行深度分析，带真实进度与取消
- **图片分析** - 对单张宠物图片进行即时解读
- **语音播报** - 自动将行为意图转换为语音播报（支持中英文，可一键静音）
- **历史记录** - 自动保存每次分析，支持搜索、来源筛选、详情、删除与 CSV / Markdown 导出
- **现代化界面** - 明暗双主题、底部标签页日志与历史、可拖拽分栏、响应式操作栏
- **流畅不卡顿** - 模型加载、摄像头检测与视频分析全部后台线程执行，界面实时反馈
- **拖拽导入** - 将视频或图片直接拖入窗口即可开始分析
- **13种行为实时识别** - 站立、坐着、趴卧、睡觉、行走、奔跑、伸懒腰、理毛、踩奶、蹭物体、捕猎、玩耍、发呆
- **尺度/帧率无关** - 所有特征按体长归一化、速度按「体长/秒」计算，换摄像头或改分辨率不影响判定；姿态判定带时序迟滞，避免标签闪烁；单帧漏检在 1 秒内沿用上一标签

## 技术栈

| 模块 | 技术 |
|------|------|
| 姿态估计 | RTMPose-M AP-10K 动物姿态模型 (ONNX)，YOLO-pose 兼容回退 |
| 目标检测 | YOLOv8n（定位画面中的猫/狗） |
| 行为分类 | 基于几何规则的智能分类器 |
| 语音合成 | Microsoft Edge TTS |
| 视频处理 | OpenCV |
| 界面框架 | PyQt6 + 自研 QSS 设计系统（明暗双主题） |
| 数据存储 | SQLite |
| 配置管理 | PyYAML |

## 快速开始

### 环境要求

- Python 3.9+
- Windows 10+ / macOS / Linux
- 摄像头（实时分析功能需要）

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd AI宠物行为识别

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# 标准启动
python main.py

# 跳过欢迎界面
python main.py --no-welcome

# 启用调试模式
python main.py --debug

# 查看版本
python main.py --version
```

### 快速启动脚本

Windows 用户可直接双击 `run.bat` 启动。

### 模型准备

项目默认使用 **RTMPose-M AP-10K** 动物姿态模型（ONNX，约 52MB）。该模型在 AP-10K 数据集上训练，覆盖猫、狗等 54 种动物，模型文件位于 `models/rtmpose_ap10k.onnx`。

若模型文件缺失（例如首次拉取仓库时未包含大文件），运行以下脚本自动下载并校验：

```bash
python download_rtmpose.py
```

目标检测使用 YOLOv8n（约 6MB），首次启动时自动下载。若没有任何姿态模型可用，程序会回退到基于几何特征的基础模式。

## 项目结构

```
AI宠物行为识别/
├── main.py                    # 程序入口
├── setup.py                   # 安装配置
├── Makefile                   # 开发任务脚本
├── requirements.txt           # Python 依赖
├── USER_MANUAL.md             # 使用说明书
├── config/
│   ├── app_config.yaml        # 应用配置
│   └── intent_mapping.yaml    # 行为-意图映射
├── core/                      # 核心业务逻辑
│   ├── video_handler.py       # 视频流处理
│   ├── file_analyzer.py       # 帧分析器
│   ├── emotion_analyzer.py    # 情感分析
│   ├── intent_mapper.py       # 意图映射
│   ├── speech_synthesizer.py  # 语音合成
│   └── record_manager.py      # 记录管理
├── models/                    # AI 模型
│   ├── pose_estimator.py      # 姿态估计器
│   ├── behavior_classifier.py # 行为分类器
│   └── animal_keypoints.json  # 关键点定义
├── ui/                        # 用户界面
│   ├── main_window.py         # 主窗口与业务流程
│   ├── theme.py               # 设计令牌、QSS 与明暗主题管理
│   ├── components.py          # 卡片、空状态、Toast 等通用组件
│   ├── video_panel.py         # 视频画布
│   ├── control_panel.py       # 两行式控制栏
│   ├── info_panel.py          # 右侧行为与意图面板
│   ├── log_panel.py           # 运行日志
│   ├── history_panel.py       # 历史记录（搜索/筛选/导出）
│   ├── record_detail_dialog.py# 记录详情与报告导出
│   ├── settings_dialog.py     # 设置对话框
│   ├── welcome_dialog.py      # 欢迎对话框
│   ├── help_dialog.py         # 帮助对话框
│   ├── about_dialog.py        # 关于对话框
│   └── intent_dialog.py       # 意图弹窗
├── resources/                 # 图标与 QSS 贴图
├── utils/                     # 工具模块
│   ├── config_loader.py       # 配置加载器
│   ├── logger.py              # 日志工具
│   ├── constants.py           # 公共常量
│   ├── paths.py               # 资源与数据路径
│   ├── record_view.py         # 记录展示纯函数
│   └── exporters.py           # CSV / Markdown 导出
├── tests/                     # 测试文件
│   ├── conftest.py            # 测试配置
│   ├── test_behavior_classifier.py
│   ├── test_emotion_analyzer.py
│   ├── test_exporters.py
│   ├── test_file_analyzer.py
│   ├── test_intent_mapper.py
│   ├── test_record_manager.py
│   ├── test_record_view.py
│   └── test_video_handler.py
└── download_rtmpose.py        # 模型下载脚本
```

## 使用指南

### 实时识别

1. 确保摄像头已正确连接
2. 点击「开始识别」按钮
3. 将宠物置于摄像头前
4. 观察画面中的关键点标注和语音播报
5. 点击「停止」结束识别

### 视频分析

1. 点击「导入视频」或使用快捷键 `Ctrl+O`，也可直接把视频拖入窗口
2. 选择视频文件（支持 MP4/AVI/MOV）
3. 分析在后台执行，画布底部显示真实进度，可随时点击「取消分析」
4. 完成后自动保存到历史记录，并展示行为分布与意图摘要

### 图片分析

1. 点击「导入图片」或使用快捷键 `Ctrl+I`，也可直接拖入图片
2. 选择图片文件（支持 JPG/PNG/BMP）
3. 立即查看分析结果，并自动保存到历史记录

### 历史记录与报告

- 底部「历史记录」标签页支持按意图/文件搜索、按来源筛选、排序与多选
- 双击记录或右键「查看详情」可查看行为时间线、分布与原始意图
- 「导出」可将当前筛选结果保存为 CSV；详情页可导出单条 Markdown 报告
- 点击右侧「收起」或按 `Ctrl+L` 可折叠底部面板，获得更大的视频区域

### 主题与画面

- 启动默认浅色主题；右上角「深色主题 / 浅色主题」按钮或 `Ctrl+Shift+D` 可即时切换（仅对当前会话生效）
- 「保存画面」或 `Ctrl+S` 将当前视频帧截图保存到本地
- 语音播报可用面板中的「语音」按钮或 `Ctrl+M` 随时开关

## 配置说明

### 应用配置 (config/app_config.yaml)

```yaml
recognition:
  confidence_threshold: 0.75  # 置信度阈值 (0.50-0.95)
  target_fps: 15              # 目标帧率

speech:
  language: zh_CN             # 语言 (zh_CN/en_US)
  voice: female_warm          # 音色
  speed: 1.0                  # 语速
  volume: 1.0                 # 音量
```

### 自定义行为解读 (config/intent_mapping.yaml)

可编辑此文件自定义每种行为的意图描述，支持中英文。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+R` | 开始实时识别 |
| `Ctrl+T` | 停止实时识别 |
| `Ctrl+O` | 导入视频 |
| `Ctrl+I` | 导入图片 |
| `Ctrl+S` | 保存当前画面 |
| `Ctrl+1` / `Ctrl+2` | 切换到日志 / 历史标签页 |
| `Ctrl+L` | 收起或展开底部面板 |
| `Ctrl+Shift+D` | 切换明暗主题 |
| `Ctrl+M` | 开关语音播报 |
| `F5` | 刷新历史记录 |
| `F6` | 重新检测摄像头 |
| `Ctrl+,` | 打开设置 |
| `F1` | 使用指南 |
| `Ctrl+Q` | 退出程序 |

## 开发指南

### 安装开发依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
# 或
make install-dev
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行测试并查看覆盖率
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# 使用 Makefile
make test
make test-cov
```

### 代码质量检查

```bash
python -m py_compile main.py
python -m py_compile core/*.py

# 或使用 Makefile
make lint
```

## 部署指南

### 本地部署

```bash
# 1. 克隆项目
git clone <repository-url>
cd AI宠物行为识别

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
python main.py
```

### 打包分发

```bash
# 构建分发包
python setup.py sdist bdist_wheel

# 安装
pip install dist/pet-behavior-analyzer-2.0.0.tar.gz
```

## 常见问题

**Q: 摄像头无法打开？**
A: 请检查摄像头是否已连接、是否被其他程序占用、驱动是否正常。

**Q: 识别结果不准确？**
A: 已针对准确率做了系统性修复：特征全部按体长归一化（不受分辨率/距离影响）、速度按「体长/秒」并基于真实时间戳（不受帧率影响）、图像等比缩放不留形变、姿态判定改为打分+时序迟滞、检测阈值放宽到 0.25 并按「面积×置信度」选主目标（漏检 1 秒内沿用上一标签）。仍有误差时，请尽量让宠物全身入镜、避免遮挡、保证光线充足，并可微调置信度阈值。

**Q: 为什么没有「缓慢眨眼 / 飞机耳 / 尾巴甩动」了？**
A: 当前使用的 AP-10K 姿态模型只输出 17 个身体关键点，不包含耳朵细节与尾尖轨迹，这几类行为缺乏可靠证据，强行识别只会产生大量误报，因此已从自动识别列表中移除。相关意图文案仍保留，方便接入带尾巴/耳部关键点的模型后恢复。

**Q: 语音无法播报？**
A: 请检查系统音量、网络连接（edge-tts 需要网络）。

**Q: 如何自定义行为意图？**
A: 编辑 `config/intent_mapping.yaml` 文件，支持中英文双语。

**Q: 如何切换主题或恢复默认参数？**
A: 启动时固定为浅色主题；点击右上角主题按钮或按 `Ctrl+Shift+D` 可即时切换（仅当前会话生效）。窗口尺寸、分栏位置与底部折叠状态会自动恢复；设置窗口左下角的「恢复默认」可将识别、语音与外观参数还原为初始值。

**Q: 分析记录保存在哪里？**
A: 记录存放在本地 SQLite 数据库中，不依赖网络；可在「历史记录」中导出为 CSV 或 Markdown 报告。

## 许可证

MIT License

## 文档

- [使用说明书](USER_MANUAL.md) - 详细的安装、操作和故障排除指南