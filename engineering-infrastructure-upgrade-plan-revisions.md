# 工程基建升级方案 - 修正与补充

## 1. PDM 迁移方案风险评估与缓解策略

### 1.1 风险识别矩阵

| 风险类别 | 风险描述 | 严重程度 | 发生概率 | 风险等级 |
|---------|---------|---------|---------|---------|
| **依赖解析风险** | PDM 与 pip 解析结果不一致，导致依赖冲突 | 高 | 中 | 🔴 高 |
| **构建兼容性风险** | pdm-backend 与 setuptools 构建产物差异 | 高 | 中 | 🔴 高 |
| **CI/CD 中断风险** | 迁移期间 CI/CD 流程失效 | 高 | 高 | 🔴 高 |
| **团队适应风险** | 开发团队不熟悉 PDM 工作流程 | 中 | 高 | 🟡 中 |
| **工具链兼容性** | 现有工具 (如 readthedocs) 对 PDM 支持不足 | 中 | 中 | 🟡 中 |
| **回滚困难风险** | 迁移后难以回滚到 setuptools | 中 | 低 | 🟡 中 |
| **子模块兼容性** | autosklearn/automl_common 子模块的依赖管理 | 高 | 高 | 🔴 高 |

### 1.2 详细风险分析与缓解策略

#### 🔴 风险 1: 依赖解析差异风险

**风险描述：**
- PDM 使用 `resolvelib` 进行依赖解析，与 pip 的解析器行为可能存在差异
- 某些依赖在 PDM 中可能解析出不同版本，导致运行时错误
- 特别是 `ConfigSpace`, `pynisher`, `pyrfr`, `smac` 等 AutoML 核心依赖

**缓解策略：**

```bash
# 1. 依赖一致性验证脚本
# scripts/verify_deps.py
"""验证 PDM 与 pip 依赖解析一致性。"""
import subprocess
import json
import sys

def get_pip_deps():
    """获取 pip 解析的依赖树。"""
    result = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True,
        text=True
    )
    return {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}

def get_pdm_deps():
    """获取 PDM 解析的依赖树。"""
    result = subprocess.run(
        ["pdm", "list", "--json"],
        capture_output=True,
        text=True
    )
    return {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}

def compare_deps():
    """对比依赖差异。"""
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
            print(f"  {diff['package']}: pip={diff['pip_version']}, pdm={diff['pdm_version']}")
        return 1
    else:
        print("✅ 依赖版本一致")
        return 0

if __name__ == "__main__":
    sys.exit(compare_deps())
```

**迁移步骤调整：**
```bash
# 阶段 1: 并行验证 (2 周)
# 保持现有 setup.py 不变，同时维护 PDM 配置

# 安装 PDM
pip install pdm

# 创建 PDM 配置 (不删除 setup.py)
pdm init --non-interactive
pdm import requirements.txt

# 锁定依赖
pdm lock

# 验证依赖一致性
python scripts/verify_deps.py

# 运行完整测试套件验证功能
pdm run pytest test/ -x
```

#### 🔴 风险 2: 构建产物兼容性风险

**风险描述：**
- `pdm-backend` 生成的 wheel/sdist 可能与 `setuptools` 产物有差异
- 某些文件可能未正确包含在分发包中
- entry_points 等元数据格式可能不同

**缓解策略：**

```toml
# pyproject.toml - 保守的构建配置
[build-system]
# 方案 A: 使用 setuptools 后端 (推荐过渡期使用)
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

# 方案 B: 验证通过后切换到 pdm-backend
# requires = ["pdm-backend"]
# build-backend = "pdm.backend"

[tool.setuptools]
packages = ["autosklearn"]
include-package-data = true

[tool.setuptools.dynamic]
version = {attr = "autosklearn.__version__"}
readme = {file = ["README.md"]}

[tool.setuptools.package-data]
autosklearn = ["**/*.json", "**/*.arff", "**/*.csv", "**/*.txt"]
```

```bash
# 构建产物对比脚本
# scripts/compare_builds.py
"""对比 setuptools 和 PDM 构建产物。"""
import subprocess
import tempfile
import zipfile
from pathlib import Path

def extract_wheel_contents(wheel_path: Path, extract_dir: Path):
    """解压 wheel 文件。"""
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        zf.extractall(extract_dir)

def compare_builds():
    """对比两种构建方式的产物。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # setuptools 构建
        subprocess.run(["python", "setup.py", "bdist_wheel", "-d", str(tmp / "setuptools")])
        
        # PDM 构建
        subprocess.run(["pdm", "build", "-d", str(tmp / "pdm")])
        
        # 对比文件列表
        setuptools_wheel = next((tmp / "setuptools").glob("*.whl"))
        pdm_wheel = next((tmp / "pdm").glob("*.whl"))
        
        setuptools_files = set(zipfile.ZipFile(setuptools_wheel).namelist())
        pdm_files = set(zipfile.ZipFile(pdm_wheel).namelist())
        
        only_in_setuptools = setuptools_files - pdm_files
        only_in_pdm = pdm_files - setuptools_files
        
        if only_in_setuptools:
            print(f"⚠️ 只在 setuptools 产物中存在: {only_in_setuptools}")
        if only_in_pdm:
            print(f"⚠️ 只在 PDM 产物中存在: {only_in_pdm}")
        
        return len(only_in_setuptools) == 0 and len(only_in_pdm) == 0

if __name__ == "__main__":
    success = compare_builds()
    exit(0 if success else 1)
```

#### 🔴 风险 3: CI/CD 中断风险

**风险描述：**
- 现有 GitHub Actions 工作流依赖 `pip install -e .`
- 切换到 PDM 后可能导致安装失败
- 多平台测试矩阵 (conda/source/dist) 需要重新验证

**缓解策略：**

```yaml
# .github/workflows/pytest.yml - 兼容性更新
name: Tests

on:
  push:
    branches: [master, development]
  pull_request:
    branches: [master, development]

jobs:
  test-matrix:
    strategy:
      matrix:
        os: [ubuntu-latest]
        python: ["3.8", "3.9", "3.10", "3.11"]
        install-method: [pip, pdm]  # 同时测试两种安装方式
        
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}
      
      # 方案 1: 传统 pip 安装 (保持兼容)
      - name: Install with pip
        if: matrix.install-method == 'pip'
        run: |
          pip install -e ".[test]"
      
      # 方案 2: PDM 安装 (新方式)
      - name: Install with PDM
        if: matrix.install-method == 'pdm'
        run: |
          pip install pdm
          pdm install --dev
      
      - name: Run tests
        run: |
          if [ "${{ matrix.install-method }}" == "pdm" ]; then
            pdm run pytest --forked
          else
            pytest --forked
          fi
```

#### 🔴 风险 7: 子模块兼容性风险

**风险描述：**
- `autosklearn/automl_common` 是一个 git 子模块
- 它有自己的 `setup.py` 和依赖
- PDM 可能无法正确处理嵌套的依赖关系

**缓解策略：**

```toml
# pyproject.toml - 子模块处理
[tool.pdm]
# 排除子模块目录，避免重复处理
excludes = ["autosklearn/automl_common"]

[tool.pdm.resolution]
# 确保子模块依赖也被正确解析
respect-source-order = true

# 在 install 脚本中处理子模块
[tool.pdm.scripts]
post_install = "scripts/install_submodules.sh"
```

```bash
# scripts/install_submodules.sh
#!/bin/bash
# 安装子模块依赖

echo "Installing automl_common submodule..."
cd autosklearn/automl_common

if [ -f "pyproject.toml" ]; then
    # 如果子模块也使用 PDM
    pdm install
elif [ -f "setup.py" ]; then
    # 子模块仍使用 setuptools
    pip install -e .
fi

cd ../..
```

### 1.3 保守迁移方案 (推荐)

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

---

## 2. Taskfile 与 Makefile 冲突解决方案

### 2.1 冲突分析

| 冲突点 | 描述 | 影响 |
|-------|------|------|
| **命令命名冲突** | 两者都使用 `make <task>` 和 `task <task>`，但任务名称相同 | 用户混淆 |
| **文档不一致** | README 和 CONTRIBUTING 指向 Makefile | 新用户困惑 |
| **CI/CD 依赖** | GitHub Actions 调用 `make` 命令 | 切换成本高 |
| **习惯依赖** | 现有开发者习惯使用 `make` | 学习成本 |
| **子目录 Makefile** | `doc/Makefile` 被主 Makefile 调用 | 依赖关系复杂 |

### 2.2 解决方案：渐进式迁移策略

#### 方案 A: Taskfile 作为主入口，Makefile 作为兼容层 (推荐)

保留 Makefile，但将其任务委托给 Taskfile：

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
	$(error "Taskfile 未安装。请运行: npm install -g @go-task/cli 或 sh -c \"$$(curl --location https://taskfile.dev/install.sh)\" -- -d -b ~/.local/bin")
else
	@task $@
endif

# 保留传统任务作为后备
legacy-help:
	@echo "传统 Makefile 命令 (已弃用):"
	@echo "  make install-dev    安装开发依赖"
	@echo "  make test           运行测试"
	@echo "  ..."
```

```yaml
# Taskfile.yml - 完整的任务定义
version: '3'

vars:
  PYTHON: python
  PYTEST: python -m pytest

env:
  PYTEST_ARGS: --forked --durations=20 --timeout=600 --timeout-method=thread -s

tasks:
  default:
    desc: 显示所有可用任务
    cmds:
      - task --list

  install-dev:
    desc: 安装开发依赖并配置 pre-commit
    cmds:
      - pip install -e ".[test,examples,docs]"
      - pre-commit install

  check:
    desc: 运行所有代码检查 (black, isort, mypy, flake8)
    cmds:
      - task: check-black
      - task: check-isort
      - task: check-mypy
      - task: check-flake8

  check-black:
    desc: 检查 black 格式
    cmds:
      - black autosklearn examples test --check

  check-isort:
    desc: 检查 isort 格式
    cmds:
      - isort autosklearn test --check

  check-mypy:
    desc: 运行 mypy 类型检查
    cmds:
      - mypy autosklearn

  check-flake8:
    desc: 运行 flake8 检查
    cmds:
      - flake8 autosklearn
      - flake8 test

  format:
    desc: 格式化代码
    cmds:
      - task: format-black
      - task: format-isort

  format-black:
    desc: 使用 black 格式化
    cmds:
      - black autosklearn test examples

  format-isort:
    desc: 使用 isort 格式化
    cmds:
      - isort autosklearn test

  pre-commit:
    desc: 运行 pre-commit 检查
    cmds:
      - pre-commit run --all-files

  test:
    desc: 运行测试套件
    cmds:
      - "{{.PYTEST}} {{.PYTEST_ARGS}} test"

  test-fast:
    desc: 运行快速测试
    cmds:
      - "{{.PYTEST}} {{.PYTEST_ARGS}} -m fast test"

  clean:
    desc: 清理构建产物
    cmds:
      - rm -rf build dist *.egg-info .pytest_cache .mypy_cache htmlcov
      - find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
      - find . -type f -name "*.pyc" -delete 2>/dev/null || true

  clean-doc:
    desc: 清理文档构建产物
    dir: doc
    cmds:
      - make clean

  build:
    desc: 构建分发包
    cmds:
      - python setup.py sdist

  doc:
    desc: 构建文档
    dir: doc
    cmds:
      - make html-noexamples
      - echo "文档已构建: file://{{.PWD}}/doc/build/html/index.html"

  doc-full:
    desc: 构建完整文档 (包含示例)
    dir: doc
    cmds:
      - make html

  links:
    desc: 检查文档链接
    dir: doc
    cmds:
      - make linkcheck

  # 新增任务 (Makefile 中没有的)
  update-deps:
    desc: 更新所有依赖
    cmds:
      - pip install --upgrade -e ".[test,examples,docs]"
      - pre-commit autoupdate

  ci:
    desc: 模拟完整 CI 流程
    cmds:
      - task: check
      - task: test
      - echo "✅ CI 检查通过"
```

#### 方案 B: 完全替换方案 (激进)

如果团队决定完全迁移：

```bash
# 1. 备份 Makefile
mv Makefile Makefile.legacy

# 2. 创建迁移脚本
# scripts/migrate_to_taskfile.sh
#!/bin/bash
echo "====================================="
echo "Taskfile 迁移助手"
echo "====================================="
echo ""
echo "检测到您正在使用 make 命令。"
echo ""
echo "我们正在迁移到 Taskfile (跨平台任务运行器)。"
echo ""

# 检测 task 是否安装
if ! command -v task &> /dev/null; then
    echo "⚠️  Taskfile 未安装。"
    echo ""
    echo "安装方式:"
    echo "  macOS/Linux: sh -c \"\$(curl --location https://taskfile.dev/install.sh)\" -- -d -b ~/.local/bin"
    echo "  Windows: choco install go-task"
    echo "  npm: npm install -g @go-task/cli"
    echo ""
    echo "或者继续使用传统 Makefile:"
    echo "  make -f Makefile.legacy <command>"
    exit 1
fi

echo "✅ Taskfile 已安装"
echo ""
echo "常用命令对照表:"
echo "  make install-dev  →  task install-dev"
echo "  make test         →  task test"
echo "  make check        →  task check"
echo "  make format       →  task format"
echo ""
echo "查看所有任务: task --list"
echo ""

# 提示用户
read -p "是否将 'make $1' 转换为 'task $1'? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    task "$1"
else
    echo "使用传统 Makefile..."
    make -f Makefile.legacy "$1"
fi
```

```makefile
# Makefile - 迁移提示版本
.PHONY: %

%:
	@bash scripts/migrate_to_taskfile.sh $@
```

### 2.3 推荐的渐进式迁移路径

```
阶段 1: 共存期 (4 周)
├── 添加 Taskfile.yml (新功能)
├── 保留 Makefile (向后兼容)
├── Makefile 添加提示，引导用户使用 Taskfile
├── 更新文档，推荐 Taskfile
└── CI/CD 逐步切换到 task 命令

阶段 2: 过渡期 (4 周)
├── Makefile 任务委托给 Taskfile
├── 添加安装检测和提示
├── 收集用户反馈
└── 修复跨平台问题

阶段 3: 切换期 (2 周)
├── Makefile 重命名为 Makefile.legacy
├── 创建包装器 Makefile (提示迁移)
├── 更新所有文档
└── 团队培训

阶段 4: 清理期 (2 个月后)
├── 移除 Makefile.legacy
├── 完全使用 Taskfile
└── 清理迁移脚本
```

### 2.4 团队迁移指南

```markdown
# 开发工具迁移指南

## 背景
我们正在从 Makefile 迁移到 Taskfile，以获得更好的跨平台支持。

## 你需要做什么

### 1. 安装 Taskfile

**macOS/Linux:**
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

**Windows:**
```powershell
choco install go-task
# 或
scoop install task
```

**npm (跨平台):**
```bash
npm install -g @go-task/cli
```

### 2. 验证安装
```bash
task --version
```

### 3. 命令对照表

| 旧命令 (Makefile) | 新命令 (Taskfile) | 说明 |
|------------------|-------------------|------|
| `make install-dev` | `task install-dev` | 安装开发依赖 |
| `make test` | `task test` | 运行测试 |
| `make check` | `task check` | 代码检查 |
| `make format` | `task format` | 格式化代码 |
| `make clean` | `task clean` | 清理构建产物 |
| `make doc` | `task doc` | 构建文档 |

### 4. 查看所有任务
```bash
task --list
```

### 5. 获取帮助
如果在迁移过程中遇到问题，请：
1. 查看详细文档: [CONTRIBUTING.md](CONTRIBUTING.md)
2. 在 Slack #dev-help 频道提问
3. 创建 GitHub Issue

## 过渡期支持
在完全迁移前，你可以继续使用 `make` 命令，
系统会提示你对应的 `task` 命令。
```

---

## 3. 修正后的实施路线图

### Phase 1: 风险评估与准备 (2 周)

| 任务 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|----------|
| 依赖解析一致性验证 | DevOps | verify_deps.py | 差异 < 5% |
| 构建产物对比 | DevOps | compare_builds.py | 文件列表一致 |
| CI/CD 双轨配置 | DevOps | pytest.yml 更新 | pip/pdm 都通过 |
| Taskfile 创建 | DevOps | Taskfile.yml | 覆盖所有 make 任务 |
| Makefile 兼容层 | DevOps | Makefile 更新 | 提示迁移 |

### Phase 2: 并行验证期 (4 周)

| 任务 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|----------|
| 团队试用 PDM | 全员 | 反馈报告 | 无阻塞问题 |
| 团队试用 Taskfile | 全员 | 反馈报告 | Windows/Mac/Linux 都可用 |
| 依赖差异修复 | DevOps | pdm.lock | 与 pip 一致 |
| 文档更新 | Tech Writer | CONTRIBUTING.md | 包含迁移指南 |

### Phase 3: 渐进切换期 (2 周)

| 任务 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|----------|
| CI 默认切换到 PDM | DevOps | CI 配置 | 构建成功 |
| Makefile 委托模式 | DevOps | Makefile | 调用 task |
| 文档推荐新工具 | Tech Writer | README.md | 新用户首选新工具 |

### Phase 4: 完全迁移 (1 个月后)

| 任务 | 负责人 | 输出物 | 验收标准 |
|------|--------|--------|----------|
| 移除 setup.py | DevOps | - | 所有 CI 通过 |
| 移除 Makefile.legacy | DevOps | - | 无用户投诉 |
| 完全切换到 pdm-backend | DevOps | pyproject.toml | 构建成功 |

---

## 4. 关键决策点

### 决策 1: 是否继续推进 PDM 迁移？

**建议:** 继续，但采用保守方案

**理由:**
- 风险可控，通过双轨运行可以逐步验证
- 长期收益明显 (依赖锁定、构建速度)
- 社区趋势 (越来越多的项目采用 pyproject.toml)

**退出条件:**
- 如果 Phase 2 发现无法解决的依赖冲突
- 如果团队接受度 < 50%
- 如果 CI 维护成本增加 > 30%

### 决策 2: 是否继续推进 Taskfile 迁移？

**建议:** 继续，但优先级降低

**理由:**
- 收益相对较小 (主要是 Windows 开发者受益)
- 迁移成本低 (可以无缝兼容)
- 可以作为 Makefile 的增强而非替代

**退出条件:**
- 如果 Windows 开发者比例 < 10%
- 如果团队对 Makefile 满意度 > 80%

---

## 5. 附录: 回滚方案

### PDM 回滚

```bash
# 1. 恢复 setup.py (从 git 历史)
git checkout HEAD~1 -- setup.py requirements.txt

# 2. 清理 PDM 产物
rm -rf __pypackages__ pdm.lock .pdm.toml

# 3. 恢复 pip 安装
pip install -e ".[test]"

# 4. 更新 CI
# 恢复 .github/workflows/pytest.yml 中的 pip 安装步骤
```

### Taskfile 回滚

```bash
# 1. 恢复原始 Makefile
git checkout HEAD~1 -- Makefile

# 2. 可选: 删除 Taskfile
rm Taskfile.yml

# 3. 更新文档
# 恢复 README.md 和 CONTRIBUTING.md 中的 make 命令
```
