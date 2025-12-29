# Structure Expert GNN Training

本目录包含基于图神经网络（GNN）的股票预测模型训练脚本。

## 文件说明

### `structure_expert.py`
Structure Expert GNN 模型定义，包含：
- `StructureExpertGNN`: 基于 GAT 的图神经网络模型
- `GraphDataBuilder`: 将 Qlib 数据转换为图格式的工具
- `StructureTrainer`: 模型训练管理器

### `train_structure_expert.py`
训练脚本，集成系统数据：
- 从数据库加载行业映射数据
- 使用 Qlib Alpha158 特征
- 实现滚动训练（按日期循环）
- 保存模型检查点

### `visualize_structure_expert.py`
Streamlit 可视化应用：
- 提取模型嵌入（embeddings）
- 使用 t-SNE 进行降维可视化
- 交互式图表展示

## 使用方法

### 1. 训练模型

```bash
# 基本训练（使用默认参数）
python python/tools/qlib/train/train_structure_expert.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# 自定义参数
python python/tools/qlib/train/train_structure_expert.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --qlib-dir ~/.qlib/qlib_data/cn_data \
  --n-hidden 128 \
  --n-heads 8 \
  --lr 0.001 \
  --save-model models/structure_expert.pth
```

### 2. 可视化嵌入

```bash
# 启动 Streamlit 应用
streamlit run python/tools/qlib/train/visualize_structure_expert.py
```

在浏览器中打开应用后：
1. 在侧边栏配置参数（Qlib 目录、模型路径、目标日期等）
2. 点击 "Load Data and Model" 加载数据和模型
3. 点击 "Extract Embeddings" 提取嵌入
4. 点击 "Apply t-SNE" 进行降维可视化
5. 查看交互式图表

## 数据要求

### 1. 行业数据
需要先同步行业分类和成分数据：

```bash
# 同步行业分类
python tools/dataingestor/service/industry_classify_sync_service.py

# 同步行业成分
python tools/dataingestor/service/industry_member_sync_service.py
```

### 2. Qlib 数据
需要先导出 Qlib 格式数据：

```bash
# 导出日线数据
python python/tools/qlib/export_qlib.py --freq day
```

## 训练流程

训练脚本会：
1. **加载行业映射**：从数据库读取股票行业关系
2. **按日期循环**：对每个交易日：
   - 加载当日所有股票的 Alpha158 特征
   - 构建图结构（同行业股票连接）
   - 训练模型（如果提供标签）
3. **保存模型**：训练完成后保存模型权重

## 可视化流程

可视化应用会：
1. **加载模型**：从检查点加载训练好的模型
2. **提取嵌入**：对指定日期的股票提取嵌入向量
3. **降维可视化**：使用 t-SNE 将高维嵌入降维到 2D
4. **交互展示**：使用 Plotly 创建交互式散点图

## 参数说明

### 训练参数

- `--start-date`: 训练开始日期（YYYY-MM-DD）
- `--end-date`: 训练结束日期（YYYY-MM-DD）
- `--qlib-dir`: Qlib 数据目录（默认: ~/.qlib/qlib_data/cn_data）
- `--n-feat`: 特征维度（默认: 158，Alpha158）
- `--n-hidden`: 隐藏层维度（默认: 64）
- `--n-heads`: 注意力头数（默认: 4）
- `--lr`: 学习率（默认: 1e-3）
- `--device`: 设备（cuda/cpu，默认: cuda）
- `--save-model`: 模型保存路径（可选）

### 可视化参数

- `Config Path`: 配置文件路径
- `Database Schema`: 数据库 schema
- `Qlib Data Directory`: Qlib 数据目录
- `Model Path`: 模型检查点路径（可选）
- `Target Date`: 要可视化的日期
- `Perplexity`: t-SNE 困惑度参数（5-50）
- `Iterations`: t-SNE 迭代次数（250-2000）

## 模型架构

Structure Expert GNN 使用两层 GAT（Graph Attention Network）：
- **第一层**：捕捉个股与行业/概念邻居的交互
- **第二层**：特征融合
- **预测层**：输出个股评分

## 数据格式

### 输入格式
- Qlib 格式的 DataFrame，MultiIndex (datetime, instrument)
- Alpha158 特征（158 维）

### 输出格式
- 嵌入向量：高维结构嵌入
- 预测分数：个股评分

## 注意事项

1. **行业数据**：确保行业数据已同步，否则图结构可能为空
2. **数据完整性**：确保 Qlib 数据完整，缺失数据会影响训练
3. **内存使用**：训练大量股票时注意内存使用
4. **GPU 支持**：建议使用 GPU 加速训练

## 故障排除

### 训练失败：No edges found
- 检查行业数据是否已同步
- 检查股票代码格式是否正确

### 训练失败：No data for date
- 检查 Qlib 数据是否完整
- 检查日期是否为交易日

### 可视化失败：Model not found
- 检查模型路径是否正确
- 可以先使用随机权重模型进行测试

## 相关文档

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [Streamlit 文档](https://docs.streamlit.io/)


