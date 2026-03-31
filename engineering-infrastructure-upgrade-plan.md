# Auto-Sklearn 工程基建升级方案

## 1. 现状分析

### 1.1 现有工程基建概览

| 维度 | 现状 | 评估 |
|------|------|------|
| 依赖管理 | `requirements.txt` + `setup.py` + `pyproject.toml` (部分配置) | ⚠️ 传统方案，缺乏版本锁定 |
| 构建系统 | `setup.py` + `Makefile` | ⚠️ 传统方案，功能有限 |
| 代码质量 | `black` + `isort` + `flake8` + `mypy` + `pre-commit` | ✅ 基础完善，可进一步增强 |
| CI/CD | GitHub Actions (pytest, pre-commit, docs, docker) | ✅ 基础完善 |
| 文档系统 | Sphinx + sphinx-gallery | ⚠️ 功能可用，可现代化 |
| 容器化 | Dockerfile 存在，但无 Dev Container 配置 | ⚠️ 部分支持 |
| 类型检查 | mypy 已配置，但大量模块被忽略 | ⚠️ 需要渐进式增强 |

### 1.2 现有文件结构

```
auto-sklearn/
├── setup.py                    # 传统打包配置
├── pyproject.toml             # 部分工具配置 (black, isort, mypy, pytest)
├── requirements.txt           # 生产依赖
├── Makefile                   # 开发任务
├── Dockerfile                 # 基础容器镜像
├── .pre-commit-config.yaml    # 预提交钩子
├── .flake8                    # flake8 配置
├── .github/workflows/         # CI/CD 工作流
│   ├── pytest.yml
│   ├── pre-commit.yaml
│   ├── docs.yml
│   ├── docker-publish.yml
│   └── ...
└── doc/                       # Sphinx 文档
    └── Makefile
```

---

## 2. 现代化依赖管理系统

### 2.1 方案对比：Poetry vs PDM

| 特性 | Poetry | PDM |
|------|--------|-----|
| 依赖解析 | ✅ 优秀 | ✅ 优秀 |
| 版本锁定 | ✅ poetry.lock | ✅ pdm.lock |
| PEP 582 支持 | ❌ | ✅ (无需虚拟环境) |
| 构建后端 | 自定义 | 标准 PEP 517 |
| 发布到 PyPI | ✅ 内置 | ✅ 内置 |
| 脚本定义 | ✅ 支持 | ✅ 支持 |
| 社区活跃度 | 高 | 快速增长 |
| 与现有项目迁移难度 | 中等 | 较低 |

### 2.2 推荐方案：PDM

**选择理由：**
1. **PEP 582 支持**：允许在无虚拟环境模式下运行，更适合容器化场景
2. **标准合规**：严格遵循 PEP 517/518/621 标准
3. **迁移友好**：与现有 `pyproject.toml` 配置兼容性好
4. **性能优秀**：依赖解析速度快

### 2.3 ⚠️ 风险评估与缓解策略

#### 风险矩阵

| 风险类别 | 风险描述 | 严重程度 | 发生概率 | 风险等级 |
|---------|---------|---------|---------|---------|
| **依赖解析风险** | PDM 与 pip 解析结果不一致 | 高 | 中 | 🔴 高 |
| **构建兼容性风险** | pdm-backend 与 setuptools 构建产物差异 | 高 | 中 | 🔴 高 |
| **CI/CD 中断风险** | 迁移期间 CI/CD 流程失效 | 高 | 高 | 🔴 高 |
| **子模块兼容性** | `automl_common` 子模块依赖管理 | 高 | 高 | 🔴 高 |
| **团队适应风险** | 开发团队不熟悉 PDM | 中 | 高 | 🟡 中 |
| **工具链兼容性** | readthedocs 等工具支持 | 中 | 中 | 🟡 中 |

#### 缓解策略

**1. 保守迁移方案 (推荐)**

```
阶段 1: 准备期 (2 周)
├── 创建 pdm 分支
├── 添加 pyproject.toml (使用 setuptools 后端)
├── 添加 PDM 配置，但不删除 setup.py
├── CI/CD 同时支持 pip 和 pdm
└── 团队培训

阶段 2: 并行期 (4 周)
├── 保持 setup.py 和 pyproject.toml 同步
├── 所有开发者同时测试两种安装方式
├── 收集问题并修复
└── 构建产物对比验证

阶段 3: 切换期 (1 周)
├── 更新文档，推荐 PDM 作为主要方式
├── CI/CD 默认使用 PDM
├── setup.py 标记为 deprecated
└── 保留 setup.py 至少 1 个版本

阶段 4: 清理期 (1 个版本后)
├── 移除 setup.py
├── 完全切换到 pdm-backend
└── 清理兼容性代码
```

**2. 依赖验证脚本**

```python
# scripts/verify_deps.py
"""验证 PDM 与 pip 依赖解析一致性。"""
import subprocess
import json
import sys

def get_pip_deps():
    result = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True, text=True
    )
    return {pkg["name"].lower(): pkg["version"] 
            for pkg in json.loads(result.stdout)}

def get_pdm_deps():
    result = subprocess.run(
        ["pdm", "list", "--json"],
        capture_output=True, text=True
    )
    return {pkg["name"].lower(): pkg["version"] 
            for pkg in json.loads(result.stdout)}

def compare_deps():
    pip_deps = get_pip_deps()
    pdm_deps = get_pdm_deps()
    
    differences = []
    for name in set(pip_deps.keys()) & set(pdm_deps.keys()):
        if pip_deps[name] != pdm_deps[name]:
            differences.append({
                "package": name,
                "pip_version": pip_deps[name],
                "pdm_version": pdm_deps[name]
            })
    
    if differences:
        print("⚠️ 发现依赖版本差异:")
        for diff in differences:
            print(f"  {diff['package']}: "
                  f"pip={diff['pip_version']}, "
                  f"pdm={diff['pdm_version']}")
        return 1
    else:
        print("✅ 依赖版本一致")
        return 0

if __name__ == "__main__":
    sys.exit(compare_deps())
```

**3. CI/CD 双轨配置**

```yaml
# .github/workflows/pytest.yml
strategy:
  matrix:
    os: [ubuntu-latest]
    python: ["3.8", "3.9", "3.10"]
    install-method: [pip, pdm]  # 同时测试两种安装方式

steps:
  - name: Install with pip
    if: matrix.install-method == 'pip'
    run: pip install -e ".[test]"
  
  - name: Install with PDM
    if: matrix.install-method == 'pdm'
    run: |
      pip install pdm
      pdm install --dev
```

**4. 回滚方案**

```bash
# 如果迁移失败，快速回滚到 setuptools
git checkout HEAD~1 -- setup.py requirements.txt
rm -rf __pypackages__ pdm.lock .pdm.toml
pip install -e ".[test]"
```

### 2.4 实施计划

#### Phase 1: 初始化 PDM 配置 (保守方案)

```toml
# pyproject.toml (迁移后)
[project]
name = "auto-sklearn"
dynamic = ["version"]
description = "Automated machine learning."
readme = "README.md"
license = {text = "BSD-3-Clause"}
authors = [
    {name = "Matthias Feurer", email = "feurerm@informatik.uni-freiburg.de"}
]
classifiers = [
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8,<3.12"
dependencies = [
    "setuptools",
    "typing_extensions",
    "distro",
    "numpy>=1.9.0",
    "scipy>=1.7.0",
    "joblib",
    "scikit-learn>=0.24.0,<0.25.0",
    "dask>=2021.12",
    "distributed>=2021.12",
    "pyyaml",
    "pandas>=1.0",
    "liac-arff",
    "threadpoolctl",
    "tqdm",
    "ConfigSpace>=0.4.21,<0.5",
    "pynisher>=0.6.3,<0.7",
    "pyrfr>=0.8.1,<0.9",
    "smac>=1.2,<1.3",
]

[project.optional-dependencies]
test = [
    "pytest>=4.6",
    "pytest-cov",
    "pytest-forked",
    "pytest-timeout",
    "pytest-cases>=3.6.11",
    "pytest-benchmark",          # 新增：性能测试
    "mypy",
    "isort",
    "black",
    "pydocstyle",
    "openml",
    "pre-commit",
    "ruff",                      # 新增：高性能 Python linter
]
examples = [
    "matplotlib",
    "jupyter",
    "notebook",
    "seaborn",
]
docs = [
    "mkdocs-material",           # 迁移到 mkdocs-material
    "mkdocstrings[python]",
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
]
dev = [
    "auto-sklearn[test,examples,docs]",
    "pdm-bump",                  # 版本管理
    "pdm-publish",               # 发布工具
]

[project.urls]
Homepage = "https://automl.github.io/auto-sklearn"
Documentation = "https://automl.github.io/auto-sklearn"
Repository = "https://github.com/automl/auto-sklearn"
Issues = "https://github.com/automl/auto-sklearn/issues"

[project.scripts]
# 如有 CLI 工具可在此定义
# auto-sklearn = "autosklearn.cli:main"

[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"

[tool.pdm]
version = {source = "file", path = "autosklearn/__version__.py"}

[tool.pdm.scripts]
# 开发任务脚本 (替代 Makefile 部分功能)
test = "pytest --forked"
test-cov = "pytest --forked --cov=autosklearn --cov-report=xml"
lint = {composite = ["ruff check autosklearn test", "ruff format --check autosklearn test"]}
format = {composite = ["ruff format autosklearn test", "isort autosklearn test"]}
typecheck = "mypy autosklearn"
docs-serve = "mkdocs serve"
docs-build = "mkdocs build"
docs-deploy = "mkdocs gh-deploy"
```

#### Phase 2: 迁移步骤

```bash
# 1. 安装 PDM
pip install pdm

# 2. 初始化项目 (复用现有配置)
pdm init

# 3. 导入现有依赖
pdm import requirements.txt

# 4. 安装所有依赖 (包括开发依赖)
pdm install --dev

# 5. 锁定依赖版本
pdm lock

# 6. 验证构建
pdm build
```

---

## 3. 标准化开发任务流

### 3.1 ⚠️ 与现有 Makefile 的冲突分析

**冲突点识别：**

| 冲突点 | 描述 | 影响 |
|-------|------|------|
| **命令命名冲突** | 两者任务名称相同 (`make test` vs `task test`) | 用户混淆 |
| **文档不一致** | README 指向 Makefile | 新用户困惑 |
| **CI/CD 依赖** | GitHub Actions 调用 `make` | 切换成本高 |
| **习惯依赖** | 现有开发者习惯 `make` | 学习成本 |
| **子目录 Makefile** | `doc/Makefile` 被主 Makefile 调用 | 依赖复杂 |

**解决方案：渐进式迁移**

### 3.2 迁移策略：Taskfile 为主，Makefile 为兼容层

**阶段 1: 共存期 (4 周)**
- 添加 Taskfile.yml (新功能)
- 保留 Makefile (向后兼容)
- Makefile 添加提示，引导用户使用 Taskfile
- 更新文档，推荐 Taskfile

**阶段 2: 过渡期 (4 周)**
- Makefile 任务委托给 Taskfile
- 添加安装检测和提示
- 收集用户反馈

**阶段 3: 切换期 (2 周)**
- Makefile 重命名为 Makefile.legacy
- 创建包装器 Makefile (提示迁移)
- 更新所有文档

**阶段 4: 清理期 (2 个月后)**
- 移除 Makefile.legacy
- 完全使用 Taskfile

### 3.3 兼容层 Makefile 设计

```makefile
# Makefile - 修改为 Taskfile 的包装器
# 注意: 此文件保留用于向后兼容，新功能请使用 Taskfile.yml

.PHONY: help install-dev check format pre-commit clean build doc test

# 检测是否安装了 task
TASK := $(shell which task 2>/dev/null)

help:
ifeq ($(TASK),)
	@echo "⚠️  Taskfile 未安装，使用传统 Makefile 命令"
	@echo "建议安装 Task: https://taskfile.dev/installation/"
	@echo ""
	@make legacy-help
else
	@task --list
endif

# 所有任务委托给 Taskfile
install-dev check format pre-commit clean build doc test:
ifeq ($(TASK),)
	$(error "Taskfile 未安装。请运行: npm install -g @go-task/cli")
else
	@task $@
endif
```

### 3.4 Taskfile 配置

```yaml
# Taskfile.yml
version: '3'

vars:
  PYTHON: pdm run python
  PYTEST: pdm run pytest
  RUFF: pdm run ruff
  MYPY: pdm run mypy
  MKDOCS: pdm run mkdocs

env:
  PYTEST_ARGS: --forked --durations=20 --timeout=600 --timeout-method=thread -s
  COV_ARGS: --cov=autosklearn --cov-report=xml --cov-report=html

tasks:
  default:
    desc: 显示所有可用任务
    cmds:
      - task --list

  setup:
    desc: 初始化开发环境
    cmds:
      - pdm install --dev
      - pdm run pre-commit install
      - echo "开发环境初始化完成！"

  update:
    desc: 更新所有依赖
    cmds:
      - pdm update
      - pdm lock

  # ========== 代码质量任务 ==========
  lint:
    desc: 运行所有代码检查
    deps: [lint-ruff, lint-mypy]

  lint-ruff:
    desc: 使用 ruff 检查代码
    cmds:
      - "{{.RUFF}} check autosklearn test"

  lint-mypy:
    desc: 使用 mypy 进行类型检查
    cmds:
      - "{{.MYPY}} autosklearn"

  format:
    desc: 格式化代码
    cmds:
      - "{{.RUFF}} format autosklearn test"
      - pdm run isort autosklearn test

  format-check:
    desc: 检查代码格式
    cmds:
      - "{{.RUFF}} format --check autosklearn test"

  # ========== 测试任务 ==========
  test:
    desc: 运行测试
    cmds:
      - "{{.PYTEST}} {{.PYTEST_ARGS}} test"

  test-cov:
    desc: 运行测试并生成覆盖率报告
    cmds:
      - "{{.PYTEST}} {{.PYTEST_ARGS}} {{.COV_ARGS}} test"

  test-fast:
    desc: 运行快速测试 (标记为 fast 的测试)
    cmds:
      - "{{.PYTEST}} {{.PYTEST_ARGS}} -m fast test"

  test-benchmark:
    desc: 运行性能基准测试
    cmds:
      - "{{.PYTEST}} test --benchmark-only --benchmark-json=output.json"

  # ========== 文档任务 ==========
  docs-serve:
    desc: 本地预览文档
    cmds:
      - "{{.MKDOCS}} serve"

  docs-build:
    desc: 构建文档
    cmds:
      - "{{.MKDOCS}} build"

  docs-deploy:
    desc: 部署文档到 GitHub Pages
    cmds:
      - "{{.MKDOCS}} gh-deploy --force"

  # ========== 构建与发布 ==========
  build:
    desc: 构建分发包
    cmds:
      - pdm build

  publish-test:
    desc: 发布到 Test PyPI
    cmds:
      - pdm publish --repository testpypi

  publish:
    desc: 发布到 PyPI
    cmds:
      - pdm publish

  # ========== 清理任务 ==========
  clean:
    desc: 清理构建产物
    cmds:
      - rm -rf build dist *.egg-info .pytest_cache .mypy_cache htmlcov
      - rm -rf site  # mkdocs 构建输出
      - find . -type d -name __pycache__ -exec rm -rf {} +
      - find . -type f -name "*.pyc" -delete

  # ========== 完整 CI 模拟 ==========
  ci:
    desc: 模拟完整 CI 流程
    deps: [format-check, lint, test-cov, build]
    cmds:
      - echo "CI 流程完成！"
```

### 3.3 与 Makefile 的对比优势

| 特性 | Makefile | Taskfile |
|------|----------|----------|
| 跨平台 | ❌ (需要 Windows 额外配置) | ✅ (原生支持) |
| 可读性 | 中等 | 高 (YAML) |
| 任务依赖 | 基础支持 | 强大的依赖管理 |
| 条件执行 | 复杂 | 简单 |
| 并行执行 | 支持 | 支持 |
| 内置帮助 | 需手动实现 | `task --list` |

---

## 4. 增强型代码质量保障体系

### 4.1 四层质量门禁设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     四层质量门禁体系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 第一层：提交前门禁 (Pre-commit Hooks)                    │   │
│  │  • ruff (lint + format)                                 │   │
│  │  • isort (import 排序)                                   │   │
│  │  • mypy (类型检查)                                       │   │
│  │  • 执行时间: < 30 秒                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 第二层：PR 门禁 (GitHub Actions)                         │   │
│  │  • 完整测试套件 (pytest)                                 │   │
│  │  • 代码覆盖率检查 (codecov)                              │   │
│  │  • 安全扫描 (bandit, safety)                             │   │
│  │  • 执行时间: < 10 分钟                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 第三层：合并前门禁 (Merge Requirements)                  │   │
│  │  • 性能回归测试 (pytest-benchmark)                       │   │
│  │  • 集成测试                                             │   │
│  │  • 文档构建检查                                         │   │
│  │  • 执行时间: < 30 分钟                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 第四层：发布前门禁 (Release Gate)                        │   │
│  │  • 全平台测试矩阵                                       │   │
│  │  • 端到端测试                                           │   │
│  │  • 安全审计                                             │   │
│  │  • 执行时间: < 2 小时                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 第一层：增强型 Pre-commit 配置

```yaml
# .pre-commit-config.yaml
repos:
  # Ruff: 极速 Python linter 和 formatter (替代 flake8 + black)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        files: ^(autosklearn|test)/
      - id: ruff-format
        files: ^(autosklearn|test)/

  # isort: import 排序
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        files: ^(autosklearn|test)/

  # mypy: 类型检查
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
        files: ^autosklearn/
        exclude: ^autosklearn/automl_common/

  # Bandit: 安全扫描
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]
        files: ^autosklearn/

  # Gitleaks: 密钥泄露检测
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks

  # Check JSON/YAML/TOML
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-json
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: mixed-line-ending
```

### 4.3 Ruff 配置 (替代 flake8 + black)

```toml
# pyproject.toml
[tool.ruff]
target-version = "py38"  # Python 3.8+
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "D",   # pydocstyle
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "PTH", # flake8-use-pathlib
]
ignore = [
    "D100",  # Missing docstring in public module
    "D104",  # Missing docstring in public package
    "D203",  # 1 blank line required before class docstring
    "D212",  # Multi-line docstring summary should start at the first line
]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.per-file-ignores]
"test/*" = ["D"]  # 测试文件不强制文档字符串
```

### 4.4 第二层：增强型 CI 工作流

```yaml
# .github/workflows/quality-gate.yml
name: Quality Gate

on:
  push:
    branches: [master, development]
  pull_request:
    branches: [master, development]

jobs:
  # ===== 代码风格检查 =====
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: "3.10"
      - run: pdm install --dev
      - run: pdm run ruff check autosklearn test
      - run: pdm run ruff format --check autosklearn test

  # ===== 类型检查 =====
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: "3.10"
      - run: pdm install --dev
      - run: pdm run mypy autosklearn

  # ===== 安全扫描 =====
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: "3.10"
      - run: pdm install --dev
      - run: pdm run bandit -r autosklearn -f json -o bandit-report.json || true
      - run: pdm audit  # PDM 内置依赖安全审计

  # ===== 测试矩阵 =====
  test:
    needs: [lint, typecheck]
    strategy:
      matrix:
        os: [ubuntu-latest]
        python: ["3.8", "3.9", "3.10", "3.11"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: ${{ matrix.python }}
      - run: pdm install --dev
      - run: pdm run pytest --forked --cov=autosklearn --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  # ===== 性能回归测试 =====
  performance:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: "3.10"
      - run: pdm install --dev
      - run: pdm run pytest test --benchmark-only --benchmark-json=benchmark.json
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: pytest
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

---

## 5. 容器化开发环境

### 5.1 Dev Container 配置

```dockerfile
# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.10

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    git \
    curl \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 设置 locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# 安装 PDM
RUN pip install --no-cache-dir pdm

# 设置工作目录
WORKDIR /workspace

# 复制依赖文件 (利用 Docker 缓存)
COPY pyproject.toml pdm.lock* ./

# 安装项目依赖 (开发模式)
RUN pdm install --dev --no-lock

# 设置环境变量
ENV PYTHONPATH=/workspace/__pypackages__/3.10/lib
ENV PATH=/workspace/__pypackages__/3.10/bin:$PATH
```

```json
// .devcontainer/devcontainer.json
{
  "name": "Auto-Sklearn Dev",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  "features": {
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },
  "customizations": {
    "vscode": {
      "extensions": [
        // Python 开发
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        // 工具
        "redhat.vscode-yaml",
        "tamasfe.even-better-toml",
        "ms-vscode.makefile-tools",
        "task.vscode-task",
        // Git
        "eamodio.gitlens",
        "github.vscode-pull-request-github",
        // 文档
        "yzhang.markdown-all-in-one",
        // 主题
        "pkief.material-icon-theme"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.analysis.typeCheckingMode": "basic",
        "python.formatting.provider": "ruff",
        "python.linting.enabled": true,
        "python.linting.ruffEnabled": true,
        "python.linting.mypyEnabled": true,
        "ruff.organizeImports": true,
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.fixAll.ruff": true,
          "source.organizeImports": true
        }
      }
    }
  },
  "postCreateCommand": "pdm install --dev && pdm run pre-commit install",
  "remoteUser": "vscode",
  "mounts": [
    "source=${localEnv:HOME}/.gitconfig,target=/home/vscode/.gitconfig,type=bind,consistency=cached"
  ]
}
```

### 5.2 docker-compose 配置 (可选)

```yaml
# .devcontainer/docker-compose.yml
version: '3.8'

services:
  dev:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    volumes:
      - ..:/workspace:cached
      - pdm-cache:/home/vscode/.cache/pdm
    environment:
      - OPENBLAS_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OMP_NUM_THREADS=1
    command: sleep infinity

volumes:
  pdm-cache:
```

### 5.3 GitHub Codespaces 支持

```json
// .devcontainer/codespaces.json
{
  "name": "Auto-Sklearn Codespace",
  "image": "mcr.microsoft.com/devcontainers/universal:2",
  "hostRequirements": {
    "cpus": 4,
    "memory": "16gb",
    "storage": "32gb"
  },
  "onCreateCommand": "pip install pdm && pdm install --dev",
  "postCreateCommand": "pdm run pre-commit install",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "charliermarsh.ruff",
        "task.vscode-task"
      ]
    }
  }
}
```

---

## 6. 智能文档生成系统

### 6.1 方案选择：MkDocs Material

**选择理由：**
- **现代化设计**：Material Design 风格，响应式布局
- **搜索功能**：内置全文搜索，无需额外配置
- **API 文档**：mkdocstrings 自动从代码生成 API 文档
- **扩展丰富**：支持图表、代码块高亮、内容标签等
- **部署简单**：内置 GitHub Pages 部署

### 6.2 MkDocs 配置

```yaml
# mkdocs.yml
site_name: Auto-Sklearn
site_description: Automated Machine Learning with scikit-learn
site_url: https://automl.github.io/auto-sklearn/
repo_url: https://github.com/automl/auto-sklearn
repo_name: automl/auto-sklearn
copyright: Copyright &copy; 2024 AutoML Group

theme:
  name: material
  palette:
    # 自动暗/亮模式切换
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.expand
    - navigation.indexes
    - content.code.copy
    - content.code.annotate
    - search.highlight
    - search.share

plugins:
  - search
  - gen-files:
      scripts:
        - docs/gen_ref_pages.py  # 自动生成 API 文档
  - literate-nav:
      nav_file: SUMMARY.md
  - section-index
  - mkdocstrings:
      handlers:
        python:
          paths: [autosklearn]
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true
            show_category_heading: true
          selection:
            docstring_style: numpy

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html
  - tables

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/automl/auto-sklearn
  version:
    provider: mike  # 版本化管理

nav:
  - Home: index.md
  - Installation: installation.md
  - Quick Start: quickstart.md
  - User Guide:
      - guide/index.md
      - guide/api.md
      - guide/examples.md
  - API Reference: reference/
  - Development:
      - dev/contributing.md
      - dev/setup.md
  - Changelog: changelog.md
```

### 6.3 自动生成 API 文档脚本

```python
# docs/gen_ref_pages.py
"""自动生成 API 参考文档。"""
from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("autosklearn").rglob("*.py")):
    module_path = path.relative_to("autosklearn").with_suffix("")
    doc_path = path.relative_to("autosklearn").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
```

### 6.4 文档覆盖率统计

```python
# scripts/doc_coverage.py
"""统计文档覆盖率。"""
import ast
from pathlib import Path
from typing import List, Tuple


def get_docstring(node) -> str:
    """获取 AST 节点的文档字符串。"""
    return ast.get_docstring(node) or ""


def analyze_file(filepath: Path) -> Tuple[int, int]:
    """分析单个文件的文档覆盖情况。

    Returns:
        (有文档的公开对象数, 公开对象总数)
    """
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return 0, 0

    documented = 0
    total = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            # 跳过私有对象
            if node.name.startswith("_"):
                continue
            total += 1
            if get_docstring(node):
                documented += 1

    return documented, total


def main():
    """主函数。"""
    base_path = Path("autosklearn")
    total_documented = 0
    total_objects = 0

    for py_file in base_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        doc, total = analyze_file(py_file)
        total_documented += doc
        total_objects += total

    coverage = (total_documented / total_objects * 100) if total_objects > 0 else 0

    print(f"文档覆盖率: {coverage:.1f}%")
    print(f"已文档化: {total_documented} / {total_objects}")

    # 生成报告
    report = f"""# 文档覆盖率报告

- 覆盖率: {coverage:.1f}%
- 已文档化: {total_documented}
- 总计: {total_objects}
- 缺失: {total_objects - total_documented}
"""

    Path("docs/coverage.md").write_text(report)


if __name__ == "__main__":
    main()
```

---

## 7. 性能监控回归测试

### 7.1 pytest-benchmark 配置

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["test"]
addopts = "--forked"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "fast: marks tests as fast",
    "benchmark: marks tests as performance benchmarks",
]
```

### 7.2 性能测试示例

```python
# test/test_performance/test_automl_performance.py
"""AutoML 性能基准测试。"""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from autosklearn.classification import AutoSklearnClassifier


class TestAutoMLPerformance:
    """AutoML 性能测试类。"""

    @pytest.fixture
    def dataset(self):
        """创建测试数据集。"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        return X, y

    @pytest.mark.benchmark
    def test_automl_fit_performance(self, benchmark, dataset):
        """测试 AutoML fit 性能。"""
        X, y = dataset

        def fit_model():
            automl = AutoSklearnClassifier(
                time_left_for_this_task=60,
                per_run_time_limit=10,
                seed=42
            )
            automl.fit(X, y)
            return automl

        result = benchmark(fit_model)
        assert result is not None

    @pytest.mark.benchmark
    def test_automl_predict_performance(self, benchmark, dataset):
        """测试 AutoML predict 性能。"""
        X, y = dataset

        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            seed=42
        )
        automl.fit(X, y)

        def predict():
            return automl.predict(X)

        result = benchmark(predict)
        assert len(result) == len(y)


# test/test_performance/test_pipeline_performance.py
"""Pipeline 性能基准测试。"""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from autosklearn.pipeline.classification import SimpleClassificationPipeline


class TestPipelinePerformance:
    """Pipeline 性能测试类。"""

    @pytest.fixture
    def dataset(self):
        """创建测试数据集。"""
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            random_state=42
        )
        return X, y

    @pytest.mark.benchmark
    def test_pipeline_fit_performance(self, benchmark, dataset):
        """测试 Pipeline fit 性能。"""
        X, y = dataset

        def fit_pipeline():
            pipeline = SimpleClassificationPipeline()
            pipeline.fit(X, y)
            return pipeline

        benchmark(fit_pipeline)
```

### 7.3 性能基线配置

```yaml
# .github/benchmark-config.yml
# 性能回归阈值配置
thresholds:
  test_automl_fit_performance:
    max_relative_change: 0.15  # 允许 15% 的性能下降
    max_absolute_change: 5.0   # 或 5 秒的绝对变化

  test_automl_predict_performance:
    max_relative_change: 0.10
    max_absolute_change: 0.1

  test_pipeline_fit_performance:
    max_relative_change: 0.20
```

### 7.4 GitHub Actions 性能监控工作流

```yaml
# .github/workflows/performance.yml
name: Performance Regression

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: pdm-project/setup-pdm@v3
        with:
          python-version: "3.10"

      - run: pdm install --dev

      - name: Run benchmarks
        run: |
          pdm run pytest test/test_performance \
            --benchmark-only \
            --benchmark-json=benchmark.json \
            --benchmark-compare=baseline.json || true

      - name: Download baseline
        uses: actions/download-artifact@v3
        with:
          name: performance-baseline
          path: baseline.json
        continue-on-error: true

      - name: Compare benchmarks
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: pytest
          output-file-path: benchmark.json
          external-data-json-path: baseline.json
          fail-on-alert: true
          alert-threshold: '150%'
          comment-on-alert: true
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: ${{ github.ref == 'refs/heads/master' }}

      - name: Upload new baseline
        if: github.ref == 'refs/heads/master'
        uses: actions/upload-artifact@v3
        with:
          name: performance-baseline
          path: benchmark.json
```

---

## 8. 模块化架构支撑

### 8.1 插件化架构设计

```
autosklearn/
├── plugins/                    # 插件系统
│   ├── __init__.py
│   ├── base.py                # 插件基类
│   ├── registry.py            # 插件注册表
│   ├── loader.py              # 动态加载器
│   └── builtin/               # 内置插件
│       ├── algorithms/        # 算法插件
│       ├── optimizers/        # 优化器插件
│       └── metrics/           # 评估指标插件
├── ext/                       # 扩展接口
│   ├── __init__.py
│   ├── algorithm.py           # 算法扩展接口
│   ├── optimizer.py           # 优化器扩展接口
│   └── metric.py              # 指标扩展接口
└── automl.py                  # 主类集成插件系统
```

### 8.2 插件基类实现

```python
# autosklearn/plugins/base.py
"""插件系统基类。"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar
import importlib
import pkgutil

T = TypeVar("T", bound="Plugin")


class Plugin(ABC):
    """插件基类。

    所有 Auto-Sklearn 插件必须继承此类。
    """

    # 插件元数据
    name: str = ""
    version: str = ""
    author: str = ""
    description: str = ""

    @classmethod
    @abstractmethod
    def register(cls) -> None:
        """注册插件到系统。"""
        pass

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """获取插件元数据。"""
        return {
            "name": cls.name,
            "version": cls.version,
            "author": cls.author,
            "description": cls.description,
        }


class PluginRegistry:
    """插件注册表。

    管理所有已注册的插件。
    """

    _plugins: Dict[str, Type[Plugin]] = {}

    @classmethod
    def register(cls, plugin_class: Type[T]) -> Type[T]:
        """注册插件类。

        可用作装饰器。
        """
        if not issubclass(plugin_class, Plugin):
            raise TypeError(f"{plugin_class} 必须继承 Plugin")

        name = plugin_class.name
        if not name:
            raise ValueError(f"插件 {plugin_class.__name__} 必须设置 name 属性")

        cls._plugins[name] = plugin_class
        plugin_class.register()
        return plugin_class

    @classmethod
    def get(cls, name: str) -> Type[Plugin]:
        """获取插件类。"""
        if name not in cls._plugins:
            raise KeyError(f"未找到插件: {name}")
        return cls._plugins[name]

    @classmethod
    def list_plugins(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册插件。"""
        return {
            name: plugin.get_metadata()
            for name, plugin in cls._plugins.items()
        }

    @classmethod
    def discover(cls, package: str) -> None:
        """自动发现指定包中的插件。"""
        try:
            pkg = importlib.import_module(package)
        except ImportError:
            return

        for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            full_name = f"{package}.{name}"
            try:
                importlib.import_module(full_name)
            except ImportError:
                continue
```

### 8.3 算法插件接口

```python
# autosklearn/ext/algorithm.py
"""算法扩展接口。"""
from abc import abstractmethod
from typing import Optional, Dict, Any
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.plugins.base import Plugin, PluginRegistry


class AlgorithmPlugin(Plugin):
    """机器学习算法插件接口。

    允许第三方实现自定义算法并集成到 Auto-Sklearn。
    """

    # 算法属性
    task_type: str = ""  # "classification" 或 "regression"
    supports_sparse: bool = False
    supports_multiclass: bool = False
    supports_multilabel: bool = False

    @abstractmethod
    def get_configuration_space(self, dataset_properties: Dict) -> ConfigurationSpace:
        """返回算法的配置空间。

        Args:
            dataset_properties: 数据集属性

        Returns:
            ConfigurationSpace 对象
        """
        pass

    @abstractmethod
    def fit(self, X, y, config: Optional[Dict] = None):
        """训练模型。

        Args:
            X: 特征矩阵
            y: 目标向量
            config: 超参数配置
        """
        pass

    @abstractmethod
    def predict(self, X):
        """预测。"""
        pass

    @classmethod
    def register(cls) -> None:
        """注册算法到 Auto-Sklearn。"""
        from autosklearn.pipeline.components.base import register_component
        # 将插件转换为 Auto-Sklearn 组件
        register_component(cls)


# 注册装饰器
algorithm = PluginRegistry.register


# 示例：自定义算法插件
# @algorithm
# class MyCustomClassifier(AlgorithmPlugin):
#     name = "my_custom_classifier"
#     version = "1.0.0"
#     task_type = "classification"
#     supports_sparse = True
#
#     def get_configuration_space(self, dataset_properties):
#         ...
```

### 8.4 优化器插件接口

```python
# autosklearn/ext/optimizer.py
"""超参数优化器扩展接口。"""
from abc import abstractmethod
from typing import Callable, Optional, Dict, Any

from autosklearn.plugins.base import Plugin


class OptimizerPlugin(Plugin):
    """超参数优化器插件接口。

    允许集成自定义优化策略 (如贝叶斯优化、遗传算法等)。
    """

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        config_space: Any,
        budget: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行优化。

        Args:
            objective: 目标函数
            config_space: 配置空间
            budget: 优化预算 (如时间、迭代次数)
            **kwargs: 额外参数

        Returns:
            最优配置
        """
        pass

    @abstractmethod
    def get_incumbent(self) -> Dict[str, Any]:
        """获取当前最优解。"""
        pass

    @classmethod
    def register(cls) -> None:
        """注册优化器。"""
        from autosklearn.smbo import register_optimizer
        register_optimizer(cls)
```

### 8.5 评估指标插件接口

```python
# autosklearn/ext/metric.py
"""评估指标扩展接口。"""
from abc import abstractmethod
from typing import Any
import numpy as np

from autosklearn.plugins.base import Plugin


class MetricPlugin(Plugin):
    """评估指标插件接口。

    允许定义自定义评估指标。
    """

    # 指标属性
    greater_is_better: bool = True
    needs_proba: bool = False
    needs_threshold: bool = False

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算指标。

        Args:
            y_true: 真实标签
            y_pred: 预测值

        Returns:
            指标值
        """
        pass

    @abstractmethod
    def sign(self) -> float:
        """返回指标的符号 (用于优化方向)。"""
        return 1.0 if self.greater_is_better else -1.0

    @classmethod
    def register(cls) -> None:
        """注册指标。"""
        from autosklearn.metrics import register_metric
        register_metric(cls)
```

### 8.6 插件加载器

```python
# autosklearn/plugins/loader.py
"""插件加载器。"""
import os
import sys
from pathlib import Path
from typing import List
import importlib.util

from .base import PluginRegistry


def load_plugins_from_directory(directory: str) -> int:
    """从目录加载插件。

    Args:
        directory: 插件目录路径

    Returns:
        加载的插件数量
    """
    count = 0
    path = Path(directory)

    if not path.exists():
        return 0

    for file in path.glob("*.py"):
        if file.name.startswith("_"):
            continue

        spec = importlib.util.spec_from_file_location(
            file.stem, file
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[file.stem] = module
            spec.loader.exec_module(module)
            count += 1

    return count


def load_plugins_from_entry_points() -> int:
    """从 entry_points 加载插件。

    支持通过 pip 安装的第三方插件。
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        from importlib_metadata import entry_points

    count = 0
    eps = entry_points()

    # Python 3.10+ 返回 SelectableGroups，需要不同的处理方式
    if hasattr(eps, "select"):
        autosklearn_eps = eps.select(group="autosklearn.plugins")
    else:
        autosklearn_eps = eps.get("autosklearn.plugins", [])

    for ep in autosklearn_eps:
        try:
            plugin_class = ep.load()
            PluginRegistry.register(plugin_class)
            count += 1
        except Exception as e:
            print(f"加载插件 {ep.name} 失败: {e}")

    return count


def load_all_plugins() -> None:
    """加载所有插件。"""
    # 1. 加载内置插件
    PluginRegistry.discover("autosklearn.plugins.builtin")

    # 2. 从 entry_points 加载第三方插件
    load_plugins_from_entry_points()

    # 3. 从环境变量指定的目录加载
    plugin_dirs = os.environ.get("AUTOSKLEARN_PLUGIN_DIRS", "")
    for directory in plugin_dirs.split(":"):
        if directory:
            load_plugins_from_directory(directory)
```

---

## 9. 实施路线图 (修正版)

### 关键变更说明

基于风险评估，实施路线图已调整为**保守渐进式**方案：

1. **PDM 迁移**：采用双轨并行策略，保留 setup.py 作为后备
2. **Taskfile 迁移**：采用兼容层策略，Makefile 委托给 Taskfile
3. **增加风险评估阶段**：在正式迁移前进行充分验证

---

### Phase 0: 风险评估与准备 (2 周)

**目标**：识别风险并准备缓解措施

| 任务 | 优先级 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|--------|----------|
| 依赖解析一致性验证 | P0 | DevOps | verify_deps.py | 差异 < 5% |
| 构建产物对比 | P0 | DevOps | compare_builds.py | 文件列表一致 |
| CI/CD 双轨配置 | P0 | DevOps | pytest.yml 更新 | pip/pdm 都通过 |
| Taskfile 创建 | P0 | DevOps | Taskfile.yml | 覆盖所有 make 任务 |
| Makefile 兼容层 | P0 | DevOps | Makefile 更新 | 提示迁移 |
| 风险评估报告 | P0 | Tech Lead | 风险文档 | 所有高风险有缓解策略 |

**退出条件**：
- 如果发现无法解决的依赖冲突 → 暂停 PDM 迁移
- 如果 CI 维护成本增加 > 30% → 重新评估方案

---

### Phase 1: 并行验证期 (4 周)

**目标**：新旧系统并行运行，收集反馈

| 任务 | 优先级 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|--------|----------|
| 团队试用 PDM | P0 | 全员 | 反馈报告 | 无阻塞问题 |
| 团队试用 Taskfile | P0 | 全员 | 反馈报告 | Win/Mac/Linux 都可用 |
| 依赖差异修复 | P0 | DevOps | pdm.lock | 与 pip 一致 |
| 文档更新 | P1 | Tech Writer | CONTRIBUTING.md | 包含迁移指南 |
| 培训会议 | P1 | Tech Lead | 培训材料 | 全员参加 |

**每周检查点**：
- 周一：收集上周问题和反馈
- 周三：修复关键问题
- 周五：更新迁移指南

---

### Phase 2: 渐进切换期 (2 周)

**目标**：默认使用新系统，保留旧系统作为后备

| 任务 | 优先级 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|--------|----------|
| CI 默认切换到 PDM | P0 | DevOps | CI 配置 | 构建成功 |
| Makefile 委托模式 | P0 | DevOps | Makefile | 调用 task |
| 文档推荐新工具 | P0 | Tech Writer | README.md | 新用户首选新工具 |
| setup.py 标记弃用 | P1 | DevOps | setup.py | 添加 DeprecationWarning |

**回滚触发条件**：
- 如果生产构建失败 → 立即回滚到 setup.py
- 如果开发者投诉 > 30% → 延长过渡期

---

### Phase 3: 质量增强 (2 周)

| 任务 | 优先级 | 估计工时 |
|------|--------|----------|
| 实施四层质量门禁 | P0 | 3 天 |
| 集成安全扫描 | P1 | 2 天 |
| 增强类型检查覆盖率 | P1 | 5 天 |
| 配置 Dev Container | P1 | 2 天 |

---

### Phase 4: 高级功能 (3 周)

| 任务 | 优先级 | 估计工时 |
|------|--------|----------|
| 实施性能监控回归测试 | P1 | 5 天 |
| 设计并实现插件化架构 | P2 | 7 天 |
| 文档覆盖率统计系统 | P2 | 3 天 |
| GitHub Codespaces 支持 | P2 | 2 天 |

---

### Phase 5: 完全迁移 (1 个月后)

**目标**：移除旧系统，完全切换到新方案

| 任务 | 优先级 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|--------|----------|
| 移除 setup.py | P0 | DevOps | - | 所有 CI 通过 |
| 移除 Makefile.legacy | P0 | DevOps | - | 无用户投诉 |
| 完全切换到 pdm-backend | P0 | DevOps | pyproject.toml | 构建成功 |
| 清理兼容性代码 | P1 | DevOps | - | 代码审查通过 |

---

### 时间线总览

```
Week  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15+
      |--Phase 0--|
                  |-----Phase 1-----|
                                    |--P2--|
                                           |--P3--|
                                                  |--P4--|
                                                         |--P5--|

关键里程碑:
  Week 2: 风险评估完成，决定是否继续
  Week 6: 并行验证完成，团队反馈良好
  Week 8: 渐进切换完成，默认使用新系统
  Week 15+: 完全迁移，旧系统移除
```

---

## 10. 预期收益

### 10.1 开发效率提升

| 指标 | 现状 | 目标 | 提升 |
|------|------|------|------|
| 新成员环境搭建时间 | 2-4 小时 | < 10 分钟 | **95%** |
| 依赖安装时间 | 5-10 分钟 | < 2 分钟 | **80%** |
| 日常开发任务执行 | 多命令/工具 | 统一 Task 命令 | **统一化** |

### 10.2 代码质量提升

| 指标 | 现状 | 目标 | 提升 |
|------|------|------|------|
| 代码风格检查速度 | ~30s | ~5s | **83%** (Ruff) |
| 类型检查覆盖率 | ~40% | > 70% | **75%** |
| 文档覆盖率 | 未知 | > 80% | **可量化** |

### 10.3 交付流程加速

| 指标 | 现状 | 目标 | 提升 |
|------|------|------|------|
| CI 执行时间 | ~20 分钟 | ~10 分钟 | **50%** |
| 发布流程 | 手动 | 一键自动化 | **自动化** |
| 文档部署 | 手动 | 自动 | **自动化** |

### 10.4 协作体验优化

- **零配置开发**：VS Code + Dev Container = 一键启动
- **跨平台一致**：Windows/macOS/Linux 完全一致的环境
- **云原生支持**：GitHub Codespaces 开箱即用
- **插件生态**：第三方扩展简单易集成

---

## 附录

### A. 参考资源

- [PDM 文档](https://pdm.fming.dev/)
- [Taskfile 文档](https://taskfile.dev/)
- [Ruff 文档](https://docs.astral.sh/ruff/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Dev Containers](https://containers.dev/)

### B. 迁移检查清单

- [ ] 备份现有 `setup.py` 和 `requirements.txt`
- [ ] 创建新的 `pyproject.toml` (PDM 格式)
- [ ] 验证所有依赖正确解析
- [ ] 更新 CI/CD 工作流
- [ ] 更新开发文档
- [ ] 通知团队成员迁移步骤
- [ ] 并行运行新旧系统 (过渡期)
- [ ] 完全切换到新系统
