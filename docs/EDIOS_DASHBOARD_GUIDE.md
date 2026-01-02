# EDiOS 网页端使用指南

## 启动方式

### 方法 1: 使用 Streamlit 命令（推荐）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动 Streamlit 应用
streamlit run python/examples/edios_dashboard.py
```

或者直接运行：

```bash
streamlit run python/nq/analysis/edios/visualization.py
```

### 方法 2: 使用 Python 直接运行

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行启动脚本
python python/examples/edios_dashboard.py
```

### 方法 3: 指定配置文件

如果需要使用特定的配置文件：

```bash
streamlit run python/examples/edios_dashboard.py -- --config_path config/config.yaml
```

## 访问地址

启动后，Streamlit 会自动在浏览器中打开，默认地址为：

```
http://localhost:8501
```

如果浏览器没有自动打开，可以手动访问上述地址。

## 功能说明

### 1. 实验选择器

- 显示所有已保存的回测实验
- 选择实验后可以查看详细信息

### 2. 性能指标面板

显示以下指标：
- **Total Return**: 总收益率
- **Max Drawdown**: 最大回撤
- **Final NAV**: 最终净值
- **Trading Days**: 交易天数

### 3. 交易统计面板

显示交易相关统计：
- 总交易次数
- 买入/卖出次数
- 胜率
- 平均持仓天数

### 4. NAV 图表

显示净值曲线：
- 时间序列图表
- 支持缩放和交互

### 5. 排名轨迹分析

输入股票代码（如 `000001.SZ`），查看该股票的：
- 排名变化轨迹
- 分数变化
- 持仓时间

## 配置要求

### 数据库配置

确保 `config/config.yaml` 中包含正确的数据库配置：

```yaml
database:
  host: localhost
  port: 5432
  database: quant_db
  user: quant
  password: your_password
```

### 依赖安装

确保已安装 Streamlit 和相关依赖：

```bash
pip install streamlit plotly pandas
```

## 常见问题

### 1. 无法连接到数据库

**错误信息**: `Failed to load experiments: ...`

**解决方法**:
- 检查数据库配置是否正确
- 确保 PostgreSQL 服务正在运行
- 检查数据库连接权限

### 2. 没有显示实验

**错误信息**: `No experiments found`

**解决方法**:
- 确保已经运行过回测并启用了 `--enable_edios` 参数
- 检查数据库中是否有实验数据：
  ```sql
  SELECT * FROM edios.bt_experiment;
  ```

### 3. 端口被占用

**错误信息**: `Port 8501 is already in use`

**解决方法**:
- 使用其他端口：
  ```bash
  streamlit run python/examples/edios_dashboard.py --server.port 8502
  ```
- 或者关闭占用端口的进程

### 4. 模块导入错误

**错误信息**: `ModuleNotFoundError: No module named 'nq'`

**解决方法**:
- 确保已激活虚拟环境
- 确保在项目根目录运行
- 检查 Python 路径设置

## 高级用法

### 自定义端口

```bash
streamlit run python/examples/edios_dashboard.py --server.port 8080
```

### 指定主机

```bash
streamlit run python/examples/edios_dashboard.py --server.address 0.0.0.0
```

### 启用主题

在 Streamlit 设置中选择深色或浅色主题。

## 开发模式

如果需要修改可视化代码，可以启用自动重载：

```bash
streamlit run python/examples/edios_dashboard.py --server.runOnSave true
```

## 相关文档

- [EDiOS 使用指南](EDIOS_USAGE.md)
- [EDiOS 数据写入功能](EDIOS_DATA_WRITING.md)
- [EDiOS 与 Structure Expert 集成指南](EDIOS_STRUCTURE_EXPERT_INTEGRATION.md)

