@echo off
chcp 65001 >nul
echo ========================================
echo   宠物行为智能分析软件 - 启动脚本
echo ========================================
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查依赖
echo [信息] 检查依赖...
pip show PyQt6 >nul 2>&1
if errorlevel 1 (
    echo [警告] 依赖未安装，正在安装...
    pip install -r requirements.txt
)

echo [信息] 启动应用...
python main.py %*

pause
