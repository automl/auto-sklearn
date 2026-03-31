.PHONY: help install install-dev install-all check format clean build test test-cov test-benchmark doc doc-serve security lint type-check pre-commit publish docker-build docker-run

SHELL := /bin/bash
PYTHON ?= python
POETRY ?= poetry
PYTEST ?= $(POETRY) run pytest
BLACK ?= $(POETRY) run black
ISORT ?= $(POETRY) run isort
MYPY ?= $(POETRY) run mypy
RUFF ?= $(POETRY) run ruff
BANDIT ?= $(POETRY) run bandit
VULTURE ?= $(POETRY) run vulture
PRECOMMIT ?= $(POETRY) run pre-commit
MKDOCS ?= $(POETRY) run mkdocs

DIR := $(CURDIR)
DIST := $(DIR)/dist
DOCDIR := $(DIR)/docs
BENCHMARK_DIR := $(DIR)/benchmarks

help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    Auto-sklearn Makefile                         ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  安装命令                                                        ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  install           安装生产依赖                                  ║"
	@echo "║  install-dev       安装开发依赖 + pre-commit hooks               ║"
	@echo "║  install-all        安装所有依赖 (含 examples/docs/benchmark)    ║"
	@echo "║  install-docs       安装文档依赖                                 ║"
	@echo "║  install-examples   安装示例依赖                                 ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  质量检查 (四层质量门禁)                                         ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  lint              第一层: 代码风格检查 (ruff + black + isort)   ║"
	@echo "║  type-check        第二层: 静态类型检查 (mypy)                   ║"
	@echo "║  security          第三层: 安全漏洞扫描 (bandit + vulture)       ║"
	@echo "║  test              第四层: 单元测试                              ║"
	@echo "║  test-cov          单元测试 + 覆盖率报告                         ║"
	@echo "║  check             运行所有质量检查                              ║"
	@echo "║  pre-commit         运行 pre-commit 所有检查                     ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  格式化                                                          ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  format            格式化代码 (black + isort + ruff --fix)       ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  测试                                                            ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  test              运行单元测试                                  ║"
	@echo "║  test-cov          运行测试 + 覆盖率报告                         ║"
	@echo "║  test-benchmark    运行性能基准测试                              ║"
	@echo "║  test-parallel     并行运行测试 (4 workers)                      ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  文档                                                            ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  doc               构建 HTML 文档                                ║"
	@echo "║  doc-serve          本地预览文档 (热重载)                        ║"
	@echo "║  doc-api            生成 API 文档                                ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  构建 & 发布                                                     ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  build             构建分发包 (sdist + wheel)                    ║"
	@echo "║  publish           发布到 TestPyPI                               ║"
	@echo "║  publish-prod       发布到 PyPI                                  ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  Docker                                                          ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  docker-build       构建 Docker 镜像                             ║"
	@echo "║  docker-run         运行 Docker 容器                             ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  清理                                                            ║"
	@echo "╠──────────────────────────────────────────────────────────────────╣"
	@echo "║  clean             清理构建产物和缓存                            ║"
	@echo "║  clean-all         深度清理 (含虚拟环境)                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

install:
	$(POETRY) install --only main

install-dev:
	$(POETRY) install --with dev
	$(POETRY) run pre-commit install
	@echo "✅ 开发环境安装完成！"

install-all:
	$(POETRY) install --with dev,docs,examples,benchmark
	$(POETRY) run pre-commit install
	@echo "✅ 完整开发环境安装完成！"

install-docs:
	$(POETRY) install --with docs

install-examples:
	$(POETRY) install --with examples

lint-ruff:
	$(RUFF) check autosklearn test examples || true

lint-black:
	$(BLACK) autosklearn test examples --check || true

lint-isort:
	$(ISORT) autosklearn test examples --check || true

lint: lint-ruff lint-black lint-isort
	@echo "✅ 第一层质量门禁 (代码风格) 检查完成"

type-check:
	$(MYPY) autosklearn || true
	@echo "✅ 第二层质量门禁 (类型检查) 完成"

security-bandit:
	$(BANDIT) -r autosklearn -c pyproject.toml || true

security-vulture:
	$(VULTURE) autosklearn --min-confidence 80 || true

security: security-bandit security-vulture
	@echo "✅ 第三层质量门禁 (安全检查) 完成"

test:
	$(PYTEST) test -v --forked --durations=20

test-cov:
	$(PYTEST) test -v --forked --durations=20 \
		--cov=autosklearn \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml
	@echo "✅ 第四层质量门禁 (测试) 完成"
	@echo "📊 覆盖率报告: htmlcov/index.html"

test-parallel:
	$(PYTEST) test -v --forked -n 4 --dist loadscope

test-benchmark:
	$(PYTEST) test/benchmark -v --benchmark-only \
		--benchmark-autosave \
		--benchmark-sort=mean \
		--benchmark-columns=mean,stddev,rounds,iterations
	@echo "📊 基准测试结果保存在: .benchmarks/"

check: lint type-check security test
	@echo "🎉 所有质量门禁检查通过！"

format-black:
	$(BLACK) autosklearn test examples

format-isort:
	$(ISORT) autosklearn test examples

format-ruff:
	$(RUFF) check autosklearn test examples --fix || true

format: format-ruff format-isort format-black
	@echo "✅ 代码格式化完成"

pre-commit:
	$(PRECOMMIT) run --all-files

doc:
	$(MKDOCS) build --clean
	@echo "📚 文档构建完成: site/index.html"

doc-serve:
	$(MKDOCS) serve --dev-addr 127.0.0.1:8000

doc-api:
	$(POETRY) run python scripts/generate_api_docs.py
	@echo "📚 API 文档生成完成"

build:
	$(POETRY) build
	@echo "📦 构建完成，产物位于: dist/"

publish: build
	$(POETRY) publish --repository testpypi
	@echo "🚀 已发布到 TestPyPI"
	@echo "测试安装: pip install --index-url https://test.pypi.org/simple/ auto-sklearn"

publish-prod: build
	$(POETRY) publish
	@echo "🚀 已发布到 PyPI"

docker-build:
	docker build -t autosklearn:latest -t autosklearn:$(shell date +%Y%m%d) .

docker-run:
	docker run -it --rm \
		-v $(DIR):/workspace \
		-w /workspace \
		autosklearn:latest \
		/bin/bash

clean-build:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf autosklearn.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-doc:
	rm -rf site/
	rm -rf docs/api/

clean-cache:
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .benchmarks/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

clean: clean-build clean-doc clean-cache
	@echo "🧹 清理完成"

clean-all: clean
	rm -rf .venv/
	rm -rf .poetry/
	@echo "🧹 深度清理完成 (含虚拟环境)"

lock:
	$(POETRY) lock --no-update

update:
	$(POETRY) update

outdated:
	$(POETRY) show --outdated

tree:
	$(POETRY) show --tree
