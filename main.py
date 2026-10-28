"""宠物行为智能分析软件 - 程序入口

支持命令行参数：
    --no-welcome    本次启动跳过欢迎页
    --debug         启用调试日志
    --version       显示版本信息
"""

import os
import sys
import ctypes
import argparse
import traceback

# Windows DLL 路径修复（PyTorch 兼容性）
if sys.platform == 'win32':
    try:
        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        SetDllDirectoryW = getattr(kernel32, 'SetDllDirectoryW', None)
        if SetDllDirectoryW:
            dll_paths = [
                os.path.join(sys.base_prefix, 'Library', 'bin'),
                os.path.join(sys.base_prefix, 'Lib', 'site-packages', 'torch', 'lib'),
            ]
            for dll_path in dll_paths:
                if os.path.exists(dll_path):
                    SetDllDirectoryW(dll_path)
                    break
    except Exception:
        pass

    torch_lib = os.path.join(sys.base_prefix, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.constants import APP_NAME, APP_VERSION  # noqa: E402


def setup_global_exception_handler():
    """设置全局异常处理器，防止未捕获异常导致静默崩溃"""
    def handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n{'='*50}", file=sys.stderr)
        print("未捕获的异常:", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        # 尝试写入日志文件
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'crash.log')
            with open(log_file, 'a', encoding='utf-8') as f:
                from datetime import datetime
                f.write(f"\n{'='*50}\n")
                f.write(f"崩溃时间: {datetime.now().isoformat()}\n")
                f.write(error_msg)
                f.write(f"{'='*50}\n")
        except Exception:
            pass

    sys.excepthook = handle_exception


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} - 通过AI识别宠物行为并解读意图'
    )
    parser.add_argument('--no-welcome', action='store_true', help='跳过欢迎界面')
    parser.add_argument('--debug', action='store_true', help='启用调试日志级别')
    parser.add_argument(
        '--version', action='version', version=f'{APP_NAME} v{APP_VERSION}'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_global_exception_handler()

    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont, QIcon

    from ui.theme import theme_manager
    from utils.paths import app_icon_path

    app = QApplication(sys.argv)
    app.setApplicationName("PetBehaviorAnalyzer")
    app.setOrganizationName("PetBehaviorAnalyzer")
    app.setApplicationDisplayName(APP_NAME)
    app.setQuitOnLastWindowClosed(True)
    app.setStyle("Fusion")

    # 统一字体与主题（每次启动默认浅色，运行中可切换深色）
    app.setFont(QFont("Microsoft YaHei UI", 9))
    icon_path = app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    theme_manager.apply()

    from ui.main_window import MainWindow

    window = MainWindow(show_welcome=False if args.no_welcome else None)

    if args.debug:
        from utils.logger import get_logger

        get_logger().set_level("DEBUG")
        window.log_panel.append_log("调试模式已启用", "warning")

    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
