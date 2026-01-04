#!/usr/bin/env python3
"""
检查 bt_trades 表中的 exp_id 关联情况
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text, create_engine
from nq.config import load_config

def check_trades_exp_id():
    """检查 trades 表中的 exp_id 情况"""
    config = load_config()
    db_config = config.database
    
    # 构建数据库连接字符串
    conn_str = f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
    engine = create_engine(conn_str)
    
    with engine.connect() as conn:
        # 检查是否有 exp_id 为 NULL 的记录
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(exp_id) as trades_with_exp_id,
                COUNT(*) - COUNT(exp_id) as trades_without_exp_id
            FROM eidos.bt_trades
        """))
        row = result.fetchone()
        print(f"总交易数: {row[0]}")
        print(f"有 exp_id 的交易数: {row[1]}")
        print(f"无 exp_id 的交易数: {row[2]}")
        
        # 检查每个 exp_id 的交易数
        result = conn.execute(text("""
            SELECT 
                exp_id,
                COUNT(*) as trade_count
            FROM eidos.bt_trades
            GROUP BY exp_id
            ORDER BY trade_count DESC
            LIMIT 20
        """))
        print("\n各 exp_id 的交易数:")
        print("exp_id      | trade_count")
        print("-" * 30)
        for row in result:
            exp_id = row[0] if row[0] else "(NULL)"
            print(f"{exp_id:12} | {row[1]}")
        
        # 检查是否有 exp_id 为 NULL 的记录详情
        result = conn.execute(text("""
            SELECT 
                trade_id,
                exp_id,
                symbol,
                deal_time,
                side
            FROM eidos.bt_trades
            WHERE exp_id IS NULL
            LIMIT 10
        """))
        null_rows = result.fetchall()
        if null_rows:
            print("\n发现 exp_id 为 NULL 的记录:")
            for row in null_rows:
                print(f"  trade_id: {row[0]}, symbol: {row[2]}, deal_time: {row[3]}, side: {row[4]}")
        else:
            print("\n未发现 exp_id 为 NULL 的记录")
        
        # 检查 exp_id 是否在 bt_experiment 表中存在
        result = conn.execute(text("""
            SELECT 
                t.exp_id,
                COUNT(*) as trade_count,
                CASE WHEN e.exp_id IS NULL THEN 'NOT FOUND' ELSE 'EXISTS' END as exp_status
            FROM eidos.bt_trades t
            LEFT JOIN eidos.bt_experiment e ON t.exp_id = e.exp_id
            GROUP BY t.exp_id, e.exp_id
            ORDER BY trade_count DESC
            LIMIT 20
        """))
        print("\nexp_id 在 bt_experiment 表中的存在情况:")
        print("exp_id      | trade_count | status")
        print("-" * 40)
        for row in result:
            exp_id = row[0] if row[0] else "(NULL)"
            print(f"{exp_id:12} | {row[1]:11} | {row[2]}")

if __name__ == "__main__":
    check_trades_exp_id()

