# 卖出策略（Exit Strategy）学习总结

## 策略概述

这是一个基于**机器学习**的卖出模型，具备"枯竭性预测"与"持仓管理"双重能力。核心思想是：**在动能耗尽和利润回撤之前及时卖出，保护盈利并避免亏损扩大**。

## 核心问题解决

### 问题场景
- **高盈亏比但亏钱**：例如盈亏比 1.84 但仍然亏损
- **原因**：没有及时止盈，利润被回撤吞噬
- **解决方案**：通过机器学习模型预测"何时应该卖出"

## 四步实现流程

### 第一步：数据提取 (Data Extraction)

**目标**：从回测中提取每笔交易持仓期间的逐日快照

**数据结构**：
```
交易ID | 日期 | 标的代码 | 现价 | 买入均价 | 持仓天数 | 持仓期间最高价
T001   | 2025-02-05 | 000001.SZ | 10.5 | 10.0 | 0 | 10.5
T001   | 2025-02-06 | 000001.SZ | 10.8 | 10.0 | 1 | 10.8
T001   | 2025-02-07 | 000001.SZ | 10.3 | 10.0 | 2 | 10.8
```

**实现方式**：
- 在策略的 `on_bar` 或 `post_exe_step` 回调中记录持仓状态
- 回测结束后用 `pandas.to_csv` 保存

### 第二步：特征构造 (Feature Engineering)

这是策略的核心，分为三类特征：

#### 1. 动能枯竭指标 (Momentum Exhaustion)

**bias_5（乖离率）**
```python
ma5 = daily_df['close'].rolling(5).mean()
feat['bias_5'] = (daily_df['close'] - ma5) / ma5
```
- **含义**：价格偏离 5 日均线的程度
- **信号**：偏离过远（>0.1）通常预示回调
- **应用**：`bias_5 > 0.1` 且 `vol_ratio < 0.8` → 强卖出信号

**close_pos（价格重心位置）**
```python
feat['close_pos'] = (daily_df['close'] - daily_df['low']) / (daily_df['high'] - daily_df['low'] + 1e-6)
```
- **含义**：收盘价在当日波幅中的位置（0~1）
- **信号**：连续多日接近 0 → 收盘被"按在地上摩擦" → 阴跌信号

**vol_ratio（成交量衰减）**
```python
feat['vol_ratio'] = daily_df['volume'] / daily_df['volume'].rolling(5).mean()
```
- **含义**：今日成交量相对于过去 5 日均量的比例
- **信号**：`vol_ratio < 0.8` → 缩量上涨 → 动能衰竭

#### 2. 收益不对称性/持仓管理 (Risk Asymmetry)

**curr_ret（当前浮盈）**
```python
feat['curr_ret'] = (daily_df['close'] - daily_df['entry_price']) / daily_df['entry_price']
```
- **含义**：当前收益率
- **作用**：判断是否盈利，决定是否应该止盈

**drawdown（利润回撤）** ⭐ **最重要特征**
```python
feat['drawdown'] = (daily_df['highest_price_since_entry'] - daily_df['close']) / daily_df['highest_price_since_entry']
```
- **含义**：从持仓期间最高点跌下来的幅度
- **作用**：**保护盈利的核心指标**
- **逻辑**：当收益曾经超过 5% 但现在回落到 2% 时，立刻卖出
- **权重**：在模型中通常获得极高的正权重

**days_held（持仓时间）**
```python
feat['days_held'] = daily_df['days_held']
```
- **含义**：持仓天数
- **作用**：时效性特征，避免持仓过久

#### 3. 标签定义 (Labeling)

```python
feat['label'] = (daily_df['next_3d_max_loss'] < -0.03).astype(int)
```
- **含义**：未来 3 天最大跌幅超过 3% 或收益变负 → 标签为 1（应卖出）
- **注意**：需要 lookahead 到未来数据，避免未来信息泄露

### 第三步：模型训练

**模型选择**：LogisticRegression
- **原因**：能输出风险概率（0~1），便于设定阈值
- **优势**：可解释性强，能看到特征权重

**训练代码**：
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def train_exit_model(feature_df):
    X = feature_df.drop('label', axis=1)
    y = feature_df['label']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型（使用 class_weight 应对胜率低的问题）
    model = LogisticRegression(class_weight='balanced', C=0.1) 
    model.fit(X_scaled, y)
    
    # 打印特征重要性
    importance = pd.DataFrame({'feature': X.columns, 'weight': model.coef_[0]})
    print(importance.sort_values(by='weight', ascending=False))
    
    return model, scaler

# 保存模型
joblib.dump(model, 'models/sell_expert.pkl')
joblib.dump(scaler, 'models/sell_expert_scaler.pkl')
```

**关键参数**：
- `class_weight='balanced'`：应对卖出信号（正样本）较少的问题
- `C=0.1`：正则化强度，防止过拟合

### 第四步：实战集成

在 `RefinedTopKStrategy` 中集成：

```python
class RefinedTopKStrategy(BaseCustomStrategy):
    def __init__(self, ...):
        super().__init__(...)
        # 加载训练好的模型
        self.sell_model = joblib.load('models/sell_expert.pkl')
        self.sell_scaler = joblib.load('models/sell_expert_scaler.pkl')
        self.sell_threshold = 0.65  # 风险概率阈值
    
    def on_bar(self, bar):
        # 检查每个持仓
        for symbol, pos in self.active_positions.items():
            # 获取历史数据
            hist = self.get_history(symbol, 10)
            
            # 构造实时特征
            current_feat = {
                'bias_5': (bar.close - hist.ma5) / hist.ma5,
                'close_pos': (bar.close - bar.low) / (bar.high - bar.low + 1e-6),
                'vol_ratio': bar.volume / hist.ma_vol,
                'curr_ret': (bar.close - pos.entry_price) / pos.entry_price,
                'drawdown': (pos.max_price - bar.close) / pos.max_price,
                'days_held': (bar.date - pos.entry_date).days
            }
            
            # 模型预测风险概率
            feat_array = np.array([list(current_feat.values())])
            feat_scaled = self.sell_scaler.transform(feat_array)
            risk_prob = self.sell_model.predict_proba(feat_scaled)[0][1]
            
            # 如果风险概率超过阈值，卖出
            if risk_prob > self.sell_threshold:
                self.close_position(symbol, reason="Exhaustion_AI")
                logger.info(
                    f"Exit signal triggered: {symbol}, "
                    f"risk_prob={risk_prob:.3f}, "
                    f"curr_ret={current_feat['curr_ret']:.2%}, "
                    f"drawdown={current_feat['drawdown']:.2%}"
                )
```

## 关键洞察

### 1. 最强特征组合

**"枯竭"的第一个强特征**：
- `bias_5 > 0.1`（超涨）且 `vol_ratio < 0.8`（缩量）
- 在回归模型中通常会获得极高的正权重（卖出信号）

### 2. 解决高盈亏比但亏钱的问题

**drawdown 特征的重要性**：
- 模型会学会："当收益曾经超过 5% 但现在回落到 2% 时，立刻卖出"
- 这能保护盈利不被吞噬
- drawdown 权重大概率会非常高

### 3. 数据获取快捷方式

如果不想慢慢跑回测攒数据：
- 使用全市场历史数据
- 随机模拟"买入点"
- 跟踪买入后 10 天的表现
- 快速生成几十万条"模拟交易记录"来训练模型

## 与现有策略的对比

### 现有策略的卖出方式（规则-based）

查看现有策略代码，它们使用**规则-based**的卖出逻辑：

1. **CCIStrategyOptimized**：
   - CCI 下穿零轴 + 下跌趋势
   - CCI 下穿超买区（部分止盈）
   - 顶背离

2. **ImprovedMACDStrategy**：
   - 死叉（DIFF 下穿 DEA）
   - 顶背离
   - 成交量萎缩

3. **ATRStrategy**：
   - 价格跌破均线
   - ATR 收缩（趋势可能改变）

### 新策略的优势（ML-based）

1. **自适应**：模型会根据历史数据自动学习最优卖出时机
2. **多特征融合**：同时考虑动能、收益、持仓时间等多个维度
3. **概率输出**：输出风险概率，可以灵活调整阈值
4. **保护盈利**：通过 drawdown 特征主动保护已实现的盈利

## 实施建议

### 1. 数据提取脚本

创建一个脚本来提取回测数据：

```python
# python/tools/qlib/extract_exit_training_data.py
def extract_position_snapshots(strategy_instance, output_path):
    """
    从策略实例中提取持仓期间的逐日快照
    """
    snapshots = []
    
    # 遍历所有交易
    for trade_id, position_history in strategy_instance.position_history.items():
        for snapshot in position_history:
            snapshots.append({
                'trade_id': trade_id,
                'date': snapshot['date'],
                'symbol': snapshot['symbol'],
                'close': snapshot['close'],
                'high': snapshot['high'],
                'low': snapshot['low'],
                'volume': snapshot['volume'],
                'entry_price': snapshot['entry_price'],
                'highest_price_since_entry': snapshot['max_price'],
                'days_held': snapshot['days_held'],
                'next_3d_max_loss': snapshot['next_3d_max_loss'],  # 需要 lookahead
            })
    
    df = pd.DataFrame(snapshots)
    df.to_csv(output_path, index=False)
    return df
```

### 2. 特征工程模块

创建独立的特征工程模块：

```python
# python/nq/analysis/exit/feature_builder.py
class ExitFeatureBuilder:
    """卖出模型特征构建器"""
    
    def build_features(self, daily_df):
        # 实现特征构建逻辑
        pass
```

### 3. 模型训练脚本

创建训练脚本：

```python
# python/tools/qlib/train/train_exit_model.py
def main():
    # 1. 加载数据
    # 2. 构建特征
    # 3. 训练模型
    # 4. 评估模型
    # 5. 保存模型
    pass
```

### 4. 集成到现有策略

修改 `RefinedTopKStrategy` 或创建新的策略类：

```python
# python/examples/backtest_structure_expert.py
class MLExitStrategy(RefinedTopKStrategy):
    """带机器学习卖出模型的策略"""
    
    def __init__(self, ...):
        super().__init__(...)
        self._load_exit_model()
    
    def _load_exit_model(self):
        # 加载模型和 scaler
        pass
    
    def _check_exit_signal(self, symbol, position):
        # 构造特征并预测
        pass
```

## 注意事项

1. **避免未来信息泄露**：标签定义时使用 `next_3d_max_loss`，需要确保在训练时正确划分时间序列
2. **数据质量**：确保提取的数据完整，包括价格、成交量、持仓信息等
3. **模型更新**：定期用新数据重新训练模型，适应市场变化
4. **阈值调优**：风险概率阈值（如 0.65）需要根据回测结果调整
5. **特征稳定性**：确保特征在不同市场环境下都能稳定计算

## 总结

这个卖出策略的核心价值在于：

1. **主动保护盈利**：通过 drawdown 特征及时止盈
2. **预测动能衰竭**：通过技术指标组合预测趋势反转
3. **数据驱动**：用历史数据训练，而非人工规则
4. **灵活可调**：通过概率阈值灵活控制卖出时机

这是一个**从规则-based 到 ML-based 的升级**，能够更好地解决"高盈亏比但亏钱"的问题。
