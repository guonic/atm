# PyCharm 配置指南

## 问题说明

如果 PyCharm 中显示导入错误（红色波浪线），通常是因为 PyCharm 没有正确识别项目的 Python 路径。

## 快速配置步骤

### 步骤 1: 设置 Python 解释器

1. **打开设置**
   - `File` → `Settings` (Windows/Linux) 或 `PyCharm` → `Preferences` (macOS)
   - 快捷键: `Ctrl+Alt+S` (Windows/Linux) 或 `Cmd+,` (macOS)

2. **配置解释器**
   - `Project: atm` → `Python Interpreter`
   - 点击齿轮图标 → `Add...`
   - 选择 `Existing environment`
   - 解释器路径: `$PROJECT_DIR$/.venv/bin/python`
   - 点击 `OK`

### 步骤 2: 标记源代码根目录

1. **打开项目结构设置**
   - `File` → `Settings` → `Project: atm` → `Project Structure`

2. **标记源代码目录**
   - 找到 `python` 目录
   - 右键点击 → `Mark Directory as` → `Sources Root`
   - 或者选中后点击上方的 `Sources` 按钮

3. **标记内容根目录**
   - 确保项目根目录（`atm`）被标记为 `Content Root`
   - 如果没有，选中项目根目录 → 点击 `Content Root` 按钮

### 步骤 3: 刷新/重启

1. **使缓存失效并重启**
   - `File` → `Invalidate Caches / Restart...`
   - 选择 `Invalidate and Restart`
   - 等待 PyCharm 重启

### 步骤 4: 验证配置

打开任意 Python 文件（如 `python/tools/dataingestor/service/stock_ingestor_service.py`），检查导入语句：

```python
from atm.config import DatabaseConfig
from atm.data.source import TushareSource
from atm.models.stock import StockBasic
from atm.repo import StockBasicRepo
```

如果红色波浪线消失，说明配置成功。

## 详细配置说明

### 项目结构

```
atm/                          # Content Root
├── python/                   # Sources Root ← 重要！
│   ├── atm/                 # 核心包
│   │   ├── config/
│   │   ├── data/
│   │   ├── models/
│   │   └── repo/
│   └── tools/               # 工具包
│       └── dataingestor/
└── .venv/                   # 虚拟环境（已配置）
```

### 环境变量配置（可选）

如果需要，可以在运行配置中添加环境变量：

1. `Run` → `Edit Configurations...`
2. 选择配置或创建新配置
3. 在 `Environment variables` 中添加：
   ```
   PYTHONPATH=$PROJECT_DIR$/python
   ```

### 已创建的配置文件

项目已包含以下 PyCharm 配置文件（在 `.idea/` 目录）：

- `atm.iml` - 模块配置
- `modules.xml` - 模块列表
- `misc.xml` - 杂项配置
- `workspace.xml` - 工作区配置（包含 PYTHONPATH）
- `vcs.xml` - 版本控制配置

**注意**: `.idea` 目录已在 `.gitignore` 中，不会提交到 Git。每个开发者需要在自己的 PyCharm 中配置。

## 常见问题排查

### 1. 导入仍然报错

**检查项**：
- ✅ 虚拟环境是否已激活？
- ✅ `python` 目录是否标记为 `Sources Root`？
- ✅ Python 解释器是否指向 `.venv/bin/python`？
- ✅ 是否已安装所有依赖？运行: `pip install -e ".[dev]"`

**解决方法**：
```bash
# 重新安装依赖
cd python
pip install -e ".[dev]"

# 重启 PyCharm
File → Invalidate Caches / Restart...
```

### 2. 找不到 `nq` 模块

**原因**: `python` 目录没有被识别为源代码根目录

**解决**: 
- 右键 `python` 目录 → `Mark Directory as` → `Sources Root`
- 或者: `File` → `Settings` → `Project Structure` → 选中 `python` → 点击 `Sources`

### 3. 找不到 `tools` 模块

**原因**: `tools` 目录在 `python` 下，如果 `python` 是 Sources Root，`tools` 会自动被识别

**解决**: 确保 `python` 目录是 Sources Root

### 4. 类型检查错误

**配置类型检查**:
- `File` → `Settings` → `Editor` → `Inspections`
- 展开 `Python` → 检查相关检查项是否启用
- 可以禁用某些检查项（如未解析的引用）如果它们不影响功能

### 5. 自动完成不工作

**解决方法**:
- `File` → `Invalidate Caches / Restart...`
- 等待索引重建完成（右下角会显示进度）

## 验证配置的命令

在 PyCharm 的 Python Console 中运行：

```python
import sys
print(sys.path)

# 应该包含项目路径
# 例如: '/Users/guonic/Workspace/OpenSource/nexusquant/python'

# 测试导入
from atm.config import DatabaseConfig
from atm.data.source import TushareSource
from atm.models.stock import StockBasic
from atm.repo import StockBasicRepo
print("✓ All imports successful!")
```

## 其他 IDE 配置

### VSCode

创建 `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/python"
    ],
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

### 命令行使用

如果需要在命令行中使用，设置环境变量：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
```

## 参考

- [PyCharm 官方文档 - 配置 Python 项目](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)
- [PyCharm 官方文档 - 源代码根目录](https://www.jetbrains.com/help/pycharm/content-root.html)
