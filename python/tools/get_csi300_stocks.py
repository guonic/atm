#!/usr/bin/env python3
"""
获取 CSI300（沪深300）股票代码列表的工具。

支持从 Tushare API 或数据库获取 CSI300 成分股列表。
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Add python directory to path (where nq package is located)
_python_dir = Path(__file__).parent.parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from nq.config import DatabaseConfig, load_config
from nq.data.source import TushareSource, TushareSourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_csi300_from_tushare(tushare_token: str, trade_date: Optional[str] = None) -> List[str]:
    """
    从 Tushare API 获取 CSI300 成分股列表。

    Args:
        tushare_token: Tushare Pro API token.
        trade_date: 交易日期 (YYYYMMDD)，如果为空则获取最新数据。

    Returns:
        CSI300 股票代码列表（格式：000001.SZ）。
    """
    config = TushareSourceConfig(
        token=tushare_token,
        type="tushare",
    )
    source = TushareSource(config)

    try:
        # CSI300 指数代码：000300.SH
        index_code = "000300.SH"

        # 方法1: 使用 index_weight 接口获取指数成分股权重
        # 这个接口可以获取指定日期的指数成分股
        params = {
            "index_code": index_code,
        }
        if trade_date:
            params["trade_date"] = trade_date

        logger.info(f"Fetching CSI300 stocks from Tushare (index_code={index_code}, trade_date={trade_date or 'latest'})")

        stocks = []
        
        # 方法1: 使用 index_weight 接口获取指数成分股权重（推荐）
        # 这个接口返回指定日期的指数成分股及其权重
        try:
            logger.info("Trying index_weight API...")
            records = list(source.fetch(api_name="index_weight", **params))
            if records:
                # 提取股票代码（可能是 con_code 或 code 字段）
                stocks = []
                for record in records:
                    code = record.get("con_code") or record.get("code") or record.get("ts_code")
                    if code:
                        stocks.append(code)
                
                if stocks:
                    # 去重并排序
                    stocks = sorted(list(set(stocks)))
                    logger.info(f"✓ Fetched {len(stocks)} CSI300 stocks from index_weight API")
                    return stocks
        except Exception as e:
            logger.warning(f"index_weight API failed: {e}")

        # 方法2: 使用 index_dailybasic 接口（通常不包含成分股列表）
        # 这个方法通常只返回指数本身的数据，不包含成分股
        logger.warning("index_weight not available, trying alternative methods...")

        # 方法3: 如果没有指定日期，尝试获取最近的交易日
        if not trade_date:
            try:
                # 获取最近的交易日
                cal_records = list(source.fetch(api_name="trade_cal", exchange="SSE", is_open="1", end_date="20241231"))
                if cal_records:
                    # 获取最近的交易日
                    recent_dates = sorted([r.get("cal_date") for r in cal_records if r.get("cal_date")], reverse=True)
                    if recent_dates:
                        trade_date = recent_dates[0]
                        logger.info(f"Using recent trade date: {trade_date}")
                        params["trade_date"] = trade_date
                        
                        # 再次尝试 index_weight
                        records = list(source.fetch(api_name="index_weight", **params))
                        if records:
                            stocks = []
                            for record in records:
                                code = record.get("con_code") or record.get("code") or record.get("ts_code")
                                if code:
                                    stocks.append(code)
                            
                            if stocks:
                                stocks = sorted(list(set(stocks)))
                                logger.info(f"✓ Fetched {len(stocks)} CSI300 stocks using recent trade date")
                                return stocks
            except Exception as e:
                logger.warning(f"Failed to get recent trade date: {e}")

        if not stocks:
            logger.error("Failed to fetch CSI300 stocks from Tushare API")
            logger.info("Please check:")
            logger.info("1. Tushare token is valid and has access to index_weight API")
            logger.info("2. Index code is correct: 000300.SH")
            logger.info("3. Trade date format is correct: YYYYMMDD")
            return []

        return stocks

    except Exception as e:
        logger.error(f"Error fetching CSI300 stocks from Tushare: {e}", exc_info=True)
        return []


def get_csi300_from_database(db_config: DatabaseConfig) -> List[str]:
    """
    从数据库获取 CSI300 股票代码列表。

    注意：这需要数据库中已经有 CSI300 成分股数据。
    如果没有，需要先通过 Tushare API 同步。

    Args:
        db_config: 数据库配置。

    Returns:
        CSI300 股票代码列表。
    """
    from nq.repo.stock_repo import StockBasicRepo

    repo = StockBasicRepo(db_config)

    # 这里需要根据实际的数据结构来查询
    # 如果数据库中有指数成分股表，可以从那里查询
    # 否则可能需要从其他数据源获取

    logger.warning("Database method not fully implemented. Please use Tushare API method.")
    return []


def save_to_file(stocks: List[str], output_file: Path) -> None:
    """
    将股票代码列表保存到文件。

    Args:
        stocks: 股票代码列表。
        output_file: 输出文件路径。
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for stock in sorted(stocks):
            f.write(f"{stock}\n")

    logger.info(f"Saved {len(stocks)} stocks to {output_file}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="获取 CSI300（沪深300）股票代码列表")
    parser.add_argument(
        "--source",
        choices=["tushare", "database"],
        default="tushare",
        help="数据源：tushare 或 database（默认：tushare）",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("TUSHARE_TOKEN", ""),
        help="Tushare Pro API token（也可通过 TUSHARE_TOKEN 环境变量设置）",
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="交易日期 (YYYYMMDD)，如果为空则获取最新数据",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（如果不指定则打印到控制台）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（用于数据库方式）",
    )

    args = parser.parse_args()

    stocks: List[str] = []

    if args.source == "tushare":
        if not args.token:
            logger.error("Tushare token is required. Set TUSHARE_TOKEN environment variable or use --token option.")
            sys.exit(1)

        stocks = get_csi300_from_tushare(args.token, args.trade_date)

    elif args.source == "database":
        config = load_config(args.config) if args.config else load_config()
        stocks = get_csi300_from_database(config.database)

    if not stocks:
        logger.error("No stocks found. Please check your data source and parameters.")
        sys.exit(1)

    # 输出结果
    if args.output:
        output_file = Path(args.output)
        save_to_file(stocks, output_file)
        print(f"\n✓ Successfully saved {len(stocks)} CSI300 stocks to {output_file}")
    else:
        print(f"\nCSI300 股票代码列表（共 {len(stocks)} 只）:")
        print("=" * 60)
        for i, stock in enumerate(sorted(stocks), 1):
            print(f"{i:4d}. {stock}")
        print("=" * 60)
        print(f"\n总计: {len(stocks)} 只股票")


if __name__ == "__main__":
    main()

