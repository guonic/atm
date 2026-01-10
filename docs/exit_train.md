要实现一个具备“枯竭性预测”与“持仓管理”双重能力的卖出模型，我们需要分为三个步骤：数据提取、特征构造、模型训练。

第一步：获取交易数据 (Data Extraction)
由于你使用的是 backtest_structure_expert.py，你需要从回测框架中导出每笔交易持仓期间的逐日快照。

如果你的框架支持导出，你需要以下结构的 CSV：

交易ID	日期	标的代码	现价	买入均价	持仓天数
T001	2025-02-05	000001.SZ	10.5	10.0	0
T001	2025-02-06	000001.SZ	10.8	10.0	1
获取方式： 在策略的 on_bar 或 on_trade 回调中，将 self.active_positions 的状态写入一个 List，回测结束后用 pandas.to_csv 保存。

第二步：特征构造 (枯竭指标 + 时效指标)
这是卖出模型的核心。我们利用日线数据计算“动能耗尽”特征。


import pandas as pd
import numpy as np

def build_sell_features(daily_df):
    """
    daily_df: 包含 [close, high, low, volume, entry_price, highest_price_since_entry]
    """
    feat = pd.DataFrame()

    # --- 1. 动能枯竭指标 (Momentum Exhaustion) ---
    # 乖离率：偏离 5 日均线过远通常预示回调
    ma5 = daily_df['close'].rolling(5).mean()
    feat['bias_5'] = (daily_df['close'] - ma5) / ma5
    
    # 价格重心下移：收盘价在当日波幅的位置 (0~1)
    # 若连续多日接近 0，说明收盘被按在地上摩擦，属于阴跌信号
    feat['close_pos'] = (daily_df['close'] - daily_df['low']) / (daily_df['high'] - daily_df['low'] + 1e-6)
    
    # 成交量衰减：今日量比过去 5 日均量
    feat['vol_ratio'] = daily_df['volume'] / daily_df['volume'].rolling(5).mean()

    # --- 2. 收益不对称性/持仓管理 (Risk Asymmetry) ---
    # 当前浮盈
    feat['curr_ret'] = (daily_df['close'] - daily_df['entry_price']) / daily_df['entry_price']
    
    # 利润回撤：从持仓期间最高点跌下来了多少
    feat['drawdown'] = (daily_df['highest_price_since_entry'] - daily_df['close']) / daily_df['highest_price_since_entry']
    
    # 持仓时间：时效性特征
    feat['days_held'] = daily_df['days_held']

    # --- 3. 标签定义 (Labeling) ---
    # 定义未来 3 天如果最大跌幅超过 3% 或者收益变负，则设为 1 (应卖出)
    # 这里需要根据你的回测数据 lookahead 指向未来
    feat['label'] = (daily_df['next_3d_max_loss'] < -0.03).astype(int)
    
    return feat.dropna()
第三步：训练卖出回归模型
我们使用 LogisticRegression 来拟合，因为它能输出风险概率。


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import jobstate

def train_exit_model(feature_df):
    X = feature_df.drop('label', axis=1)
    y = feature_df['label']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型（使用 class_weight 应对胜率低的问题）
    model = LogisticRegression(class_weight='balanced', C=0.1) 
    model.fit(X_scaled, y)
    
    # 打印权重分布，看看谁才是“杀手”特征
    importance = pd.DataFrame({'feature': X.columns, 'weight': model.coef_[0]})
    print(importance.sort_values(by='weight', ascending=False))
    
    return model, scaler

# 保存模型供回测调用
# joblib.dump(model, 'models/sell_expert.pkl')
第四步：在现有框架中调用 (实战集成)
你需要修改你的 RefinedTopKStrategy 脚本：

加载模型：在 __init__ 中加载训练好的 .pkl 文件。
每根 K 线检查：

# 在 strategy 的每个 bar 循环中
def on_bar(self, bar):
    for symbol, pos in self.active_positions.items():
        # 获取该标的最近的数据
        hist = self.get_history(symbol, 10)
        
        # 实时构造特征特征
        current_feat = {
            'bias_5': (bar.close - hist.ma5) / hist.ma5,
            'close_pos': (bar.close - bar.low) / (bar.high - bar.low),
            'vol_ratio': bar.volume / hist.ma_vol,
            'curr_ret': (bar.close - pos.entry_price) / pos.entry_price,
            'drawdown': (pos.max_price - bar.close) / pos.max_price,
            'days_held': bar.date - pos.entry_date
        }
        
        # 模型预测风险概率
        risk_prob = self.sell_model.predict_proba([list(current_feat.values())])[0][1]
        
        # 设定一个阈值，比如 0.65
        if risk_prob > 0.65:
            self.close_position(symbol, reason="Exhaustion_AI")
💡 针对你日线数据的特别调试建议：
关于“枯竭”的第一个强特征：
关注 vol_ratio 和 bias_5 的交叉。如果 bias_5 > 0.1 (超涨) 且 vol_ratio < 0.8 (缩量)，这在回归模型中通常会获得极高的正权重（卖出信号）。
解决 1.84 盈亏比但亏钱的问题：
你的 drawdown 权重大概率会非常高。模型会学会：“当收益曾经超过 5% 但现在回落到 2% 时，立刻卖出”。这能保护你的盈利不被吞噬。
数据获取快捷方式：
如果你不想改代码去慢慢跑回测攒数据，你可以先用全市场历史数据，随机模拟“买入点”，然后跟踪买入后 10 天的表现，快速生成几十万条“模拟交易记录”来训练这个模型。
你可以先尝试提取一小部分交易数据（比如 100 笔），我帮你写一个脚本来分析这些特征与未来跌幅的相关性。