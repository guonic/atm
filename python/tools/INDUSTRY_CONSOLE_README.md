# 行业分类管理控制台

一个基于 Web 的管理控制台，用于查看和编辑股票行业分类数据。

## 功能特性

- 📊 **数据统计**: 显示总记录数、当前股票数、行业数量等统计信息
- 🔍 **股票搜索**: 支持按股票代码或名称搜索，查看行业分类信息
- 📁 **行业浏览**: 按一级、二级、三级行业分类浏览，查看每个行业的股票列表
- 📝 **历史记录**: 查看每个股票的行业分类历史变更记录
- ✏️ **数据编辑**: 支持编辑和更新股票的行业分类信息

## 安装依赖

确保已安装 Flask：

```bash
pip install flask
```

## 使用方法

### 启动服务

```bash
# 使用默认配置
python python/tools/industry_management_console.py

# 指定配置文件
python python/tools/industry_management_console.py --config_path config/config.yaml

# 指定端口和主机
python python/tools/industry_management_console.py --host 0.0.0.0 --port 8080

# 启用调试模式
python python/tools/industry_management_console.py --debug
```

### 命令行参数

- `--config_path`: 配置文件路径（默认: `config/config.yaml`）
- `--schema`: 数据库 schema（默认: `quant`）
- `--host`: 绑定主机地址（默认: `127.0.0.1`）
- `--port`: 绑定端口（默认: `5000`）
- `--debug`: 启用调试模式

### 访问控制台

启动后，在浏览器中访问：

```
http://localhost:5000
```

## 功能说明

### 1. 数据统计

首页顶部显示四个统计卡片：
- **总记录数**: 数据库中所有行业分类记录的总数
- **当前股票数**: 当前有效的股票数量（out_date 为 NULL 或未来日期）
- **行业数量**: 当前有效的三级行业数量
- **数据日期范围**: 数据的最早和最晚日期

### 2. 搜索股票

在搜索框中输入股票代码（如 `000001`）或股票名称（如 `平安银行`），点击搜索按钮。

**搜索选项**:
- **仅当前有效**: 只显示当前有效的行业分类（默认）
- **包含历史**: 显示所有历史记录

**搜索结果**:
- 显示股票代码、名称、一级/二级/三级行业、生效日期、失效日期
- 点击表格行可查看该股票的详细历史记录
- 支持分页浏览

### 3. 行业列表

切换到"行业列表"标签页，可以：
- 选择行业级别（一级/二级/三级）
- 选择是否仅显示当前有效的行业
- 查看每个行业的股票数量
- 点击"查看股票"按钮查看该行业下的所有股票

### 4. 股票详情

在搜索结果中点击股票行，或从行业列表中点击"查看详情"，可以：
- 查看该股票的所有历史行业分类记录
- 查看每次变更的生效日期和失效日期
- 编辑行业分类信息

### 5. 编辑行业分类

在股票详情页面点击"编辑"按钮，可以：
- 修改三级行业代码和名称
- 修改生效日期和失效日期
- 保存更改到数据库

**注意**: 
- 行业代码必须存在于数据库中（系统会自动查找对应的 L1/L2 信息）
- 如果 L3 代码不存在，会返回错误

## API 接口

控制台提供以下 REST API 接口：

### GET /api/stats
获取数据库统计信息

### GET /api/search?q={query}&page={page}&per_page={per_page}&current_only={true|false}
搜索股票

### GET /api/stock/{ts_code}
获取股票详细信息

### GET /api/industries?level={l1|l2|l3}&current_only={true|false}
获取行业列表

### GET /api/industry/{industry_code}/stocks?level={l1|l2|l3}&current_only={true|false}
获取行业下的股票列表

### POST /api/update
更新股票行业分类

请求体：
```json
{
  "ts_code": "000001.SZ",
  "l3_code": "801010",
  "l3_name": "银行",
  "in_date": "2024-01-01",
  "out_date": null
}
```

## 数据库表结构

控制台操作的是 `stock_industry_member` 表，主要字段：

- `ts_code`: 股票代码
- `stock_name`: 股票名称
- `l1_code`, `l1_name`: 一级行业代码和名称
- `l2_code`, `l2_name`: 二级行业代码和名称
- `l3_code`, `l3_name`: 三级行业代码和名称
- `in_date`: 生效日期
- `out_date`: 失效日期（NULL 表示当前有效）

## 注意事项

1. **数据准确性**: 编辑功能会直接修改数据库，请谨慎操作
2. **行业代码**: 编辑时输入的 L3 代码必须已存在于数据库中
3. **日期格式**: 日期格式为 `YYYY-MM-DD`
4. **权限**: 确保数据库用户有读写权限

## 故障排查

### 无法连接数据库
- 检查配置文件路径是否正确
- 检查数据库连接信息是否正确
- 检查数据库服务是否运行

### 搜索无结果
- 检查搜索关键词是否正确
- 尝试使用股票代码而不是名称
- 检查"仅当前有效"选项设置

### 编辑失败
- 检查 L3 代码是否存在于数据库中
- 检查必填字段是否都已填写
- 查看浏览器控制台的错误信息

## 技术栈

- **后端**: Flask (Python)
- **前端**: Bootstrap 5, Vanilla JavaScript
- **数据库**: PostgreSQL (通过 SQLAlchemy)


