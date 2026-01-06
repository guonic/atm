# Structure Expert Dashboard 使用指南

## 概述

Structure Expert Dashboard 是一个基于 Streamlit + Plotly 的交互式可视化 Dashboard，用于评估和可视化 Structure Expert GNN 模型。

## 架构

- **底座**: Streamlit - 快速搭建 Web 界面，原生支持 Python
- **交互核心**: Plotly - 实现点击、缩放、多图联动等交互功能
- **数据存储**: PostgreSQL / 文件存储 - 存储回测过程中的中间特征 (Embedding, Scores)

## 安装依赖

```bash
pip install streamlit plotly pandas numpy scikit-learn torch pyarrow
```

## 使用方法

### 1. 生成 Embeddings（回测时保存）

在运行回测时，添加 `--save_embeddings` 参数来保存 embeddings：

```bash
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2025-01-02 \
    --end_date 2025-01-27 \
    --save_embeddings \
    --embeddings_storage_dir storage/structure_expert_cache
```

这会在 `storage/structure_expert_cache/` 目录下保存每天的 embeddings 文件（Parquet 格式）。

### 2. 启动 Dashboard

```bash
streamlit run python/examples/structure_expert_dashboard.py
```

Dashboard 会在浏览器中自动打开（通常是 `http://localhost:8501`）。

### 3. Dashboard 功能

#### Tab 1: Visualization（可视化）
- 选择模型路径和日期
- 加载并显示 t-SNE 降维后的股票分布图
- 按行业分类着色
- 点的大小代表预测得分
- 支持交互式缩放、悬停查看详情

#### Tab 2: Data Management（数据管理）
- 查看已缓存的 embeddings 文件
- 手动生成特定日期的 embeddings（需要集成 GraphDataBuilder）

#### Tab 3: Statistics（统计信息）
- 查看缓存文件的统计信息
- 显示股票数量、行业分布等

## 数据存储

### 文件存储（默认）

Embeddings 以 Parquet 格式存储在 `storage/structure_expert_cache/` 目录下：
- 文件名格式：`embeddings_YYYYMMDD.parquet`
- 包含列：`symbol`, `score`, `industry`, `embedding_0`, `embedding_1`, ...

### PostgreSQL 存储（可选）

如需使用 PostgreSQL 存储，需要：
1. 配置数据库连接（`config/config.yaml`）
2. 在 Dashboard 侧边栏选择 "postgresql" 作为存储后端
3. 提供数据库配置路径

## 可视化函数说明

`visualize_structure_expert()` 函数：
- 使用 t-SNE 将高维 embeddings 降维到 2D
- 创建交互式散点图
- 按行业分类着色
- 点的大小表示预测得分绝对值

## 集成到回测流程

回测脚本已集成 embeddings 保存功能：
- 添加 `--save_embeddings` 参数即可自动保存
- Embeddings 与回测结果同步生成
- 支持批量日期处理

## 注意事项

1. **首次使用**：需要先运行回测生成 embeddings，Dashboard 才能显示可视化
2. **行业标签**：需要数据库中有行业分类数据（通过 Tushare 同步）
3. **性能**：t-SNE 计算可能需要一些时间，特别是股票数量较多时
4. **存储空间**：Parquet 格式相对高效，但大量日期仍会占用一定空间

## 示例工作流

```bash
# 1. 运行回测并保存 embeddings
python python/examples/backtest_structure_expert.py \
    --model_path models/structure_expert.pth \
    --start_date 2025-01-02 \
    --end_date 2025-01-27 \
    --save_embeddings

# 2. 启动 Dashboard
streamlit run python/examples/structure_expert_dashboard.py

# 3. 在浏览器中选择日期查看可视化
```

## 故障排除

- **模型文件未找到**：检查模型路径是否正确
- **无缓存数据**：先运行回测生成 embeddings
- **行业标签缺失**：检查数据库配置和行业分类数据是否同步
- **t-SNE 计算慢**：这是正常现象，可以等待或减少股票数量

