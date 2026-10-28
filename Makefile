# 宠物行为智能分析软件 - 开发任务脚本

.PHONY: install install-dev test test-cov lint run clean help

help:
	@echo "可用命令:"
	@echo "  make install     - 安装生产依赖"
	@echo "  make install-dev - 安装开发依赖"
	@echo "  make test        - 运行所有测试"
	@echo "  make test-cov    - 运行测试并生成覆盖率报告"
	@echo "  make lint        - 代码质量检查"
	@echo "  make run         - 启动应用"
	@echo "  make clean       - 清理临时文件"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	python -m py_compile main.py
	python -m py_compile core/*.py
	python -m py_compile models/*.py
	python -m py_compile utils/*.py

run:
	python main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete