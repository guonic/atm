#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速诊断 CorrelationOptimizer 问题的工具。

用法:
    python python/tools/diagnose_correlation_optimizer.py --return-matrix outputs/return_matrix.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def analyze_return_matrix(return_matrix_path: Path):
    """分析 return_matrix 文件."""
    print("=" * 80)
    print("Return Matrix Analysis")
    print("=" * 80)
    
    if not return_matrix_path.exists():
        print(f"❌ Return matrix file not found: {return_matrix_path}")
        return None
    
    try:
        return_matrix = pd.read_csv(return_matrix_path, index_col=[0, 1])
    except Exception as e:
        print(f"❌ Failed to load return matrix: {e}")
        return None
    
    print(f"✅ Loaded return matrix: {return_matrix.shape}")
    
    # 检查索引结构
    if not isinstance(return_matrix.index, pd.MultiIndex):
        print(f"❌ Return matrix index is not MultiIndex: {type(return_matrix.index)}")
        print("   Expected MultiIndex with (date, symbol) levels")
        return None
    
    # 提取日期和 symbol
    dates = set(return_matrix.index.get_level_values(0))
    symbols = set(return_matrix.index.get_level_values(1))
    
    print(f"✅ Dates: {len(dates)} unique dates")
    print(f"   Range: {min(dates)} to {max(dates)}")
    print(f"   Sample: {sorted(dates)[:5]}")
    
    print(f"✅ Symbols: {len(symbols)} unique symbols")
    print(f"   Sample: {sorted(symbols)[:10]}")
    
    # 检查每个日期的 symbol 数量
    date_symbol_counts = return_matrix.groupby(level=0).size()
    print(f"\n📊 Symbols per date:")
    print(f"   Min: {date_symbol_counts.min()}, Max: {date_symbol_counts.max()}, Mean: {date_symbol_counts.mean():.1f}")
    print(f"   Sample counts: {dict(date_symbol_counts.head(5))}")
    
    # 检查缺失值
    missing_counts = return_matrix.isna().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   {col}: {count} ({count/len(return_matrix)*100:.1f}%)")
    else:
        print(f"\n✅ No missing values")
    
    # 检查列
    print(f"\n📋 Columns: {list(return_matrix.columns)}")
    
    return return_matrix


def check_data_consistency(return_matrix: pd.DataFrame, daily_ranks_sample: dict = None):
    """检查数据一致性."""
    print("\n" + "=" * 80)
    print("Data Consistency Check")
    print("=" * 80)
    
    if return_matrix is None:
        print("❌ Cannot check consistency: return_matrix is None")
        return
    
    return_matrix_dates = set(return_matrix.index.get_level_values(0))
    return_matrix_symbols = set(return_matrix.index.get_level_values(1))
    
    print(f"Return matrix: {len(return_matrix_dates)} dates, {len(return_matrix_symbols)} symbols")
    
    if daily_ranks_sample:
        daily_ranks_dates = set(daily_ranks_sample.keys())
        all_daily_ranks_symbols = set()
        for date, ranks_df in daily_ranks_sample.items():
            all_daily_ranks_symbols.update(ranks_df.head(20)['symbol'].tolist())
        
        print(f"Daily ranks sample: {len(daily_ranks_dates)} dates, {len(all_daily_ranks_symbols)} symbols")
        
        # 检查日期重叠
        overlap_dates = daily_ranks_dates & return_matrix_dates
        print(f"\n📅 Date overlap: {len(overlap_dates)} dates")
        if len(overlap_dates) == 0:
            print("   ❌ No overlapping dates!")
            print(f"   Daily ranks dates: {sorted(daily_ranks_dates)[:5]}")
            print(f"   Return matrix dates: {sorted(return_matrix_dates)[:5]}")
        else:
            print(f"   ✅ Overlapping dates: {sorted(overlap_dates)[:5]}")
        
        # 检查 symbol 重叠
        overlap_symbols = all_daily_ranks_symbols & return_matrix_symbols
        print(f"\n📊 Symbol overlap: {len(overlap_symbols)} symbols ({len(overlap_symbols)/len(all_daily_ranks_symbols)*100:.1f}%)")
        if len(overlap_symbols) == 0:
            print("   ❌ No overlapping symbols!")
            print(f"   Daily ranks symbols: {sorted(all_daily_ranks_symbols)[:10]}")
            print(f"   Return matrix symbols: {sorted(return_matrix_symbols)[:10]}")
        else:
            print(f"   ✅ Overlapping symbols: {sorted(overlap_symbols)[:10]}")
            
            # 检查缺失的 symbol
            missing_symbols = all_daily_ranks_symbols - return_matrix_symbols
            if missing_symbols:
                print(f"   ⚠️  Missing symbols in return_matrix: {sorted(missing_symbols)[:10]}")


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="Diagnose CorrelationOptimizer issues")
    parser.add_argument(
        "--return-matrix",
        type=str,
        default="outputs/return_matrix.csv",
        help="Path to return_matrix.csv file",
    )
    parser.add_argument(
        "--check-index",
        action="store_true",
        help="Check return_matrix index structure in detail",
    )
    
    args = parser.parse_args()
    
    return_matrix_path = Path(args.return_matrix)
    
    # 分析 return_matrix
    return_matrix = analyze_return_matrix(return_matrix_path)
    
    # 检查索引结构
    if args.check_index and return_matrix is not None:
        print("\n" + "=" * 80)
        print("Index Structure Details")
        print("=" * 80)
        print(f"Index type: {type(return_matrix.index)}")
        print(f"Index names: {return_matrix.index.names}")
        print(f"Index levels: {return_matrix.index.nlevels}")
        if isinstance(return_matrix.index, pd.MultiIndex):
            print(f"Level 0 (dates) type: {type(return_matrix.index.get_level_values(0)[0])}")
            print(f"Level 1 (symbols) type: {type(return_matrix.index.get_level_values(1)[0])}")
            print(f"\nSample index tuples:")
            for i, idx in enumerate(return_matrix.index[:5]):
                print(f"   {idx}")
    
    # 检查数据一致性（如果有 daily_ranks 样本数据）
    # 这里可以扩展，从日志或其他来源加载 daily_ranks 样本
    
    print("\n" + "=" * 80)
    print("Diagnosis Complete")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("1. Check if dates and symbols match between daily_ranks and return_matrix")
    print("2. Verify that instruments list includes all needed symbols")
    print("3. Check logs for 'Symbol filtering' and 'Date overlap' messages")
    print("4. Run with --check-index to see detailed index structure")


if __name__ == '__main__':
    main()
