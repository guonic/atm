#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industry Management Console

A web-based management console for viewing and editing stock industry classification data.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.stock_repo import StockIndustryMemberRepo
from nq.utils.data_normalize import normalize_stock_code
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for database connection
db_config: Optional[DatabaseConfig] = None
repo: Optional[StockIndustryMemberRepo] = None
schema: str = "quant"


def init_database(config_path: str, db_schema: str = "quant"):
    """Initialize database connection."""
    global db_config, repo, schema
    try:
        config = load_config(config_path)
        db_config = config.database
        schema = db_schema
        repo = StockIndustryMemberRepo(db_config, schema=schema)
        logger.info(f"Database initialized: {db_config.host}:{db_config.port}/{db_config.database}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@app.route("/")
def index():
    """Main page."""
    return render_template("industry_console.html")


@app.route("/api/stats")
def get_stats():
    """Get database statistics."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        # Total records
        sql_total = f"SELECT COUNT(*) FROM {table_name}"
        with engine.connect() as conn:
            result = conn.execute(text(sql_total))
            total_count = result.scalar()

        # Current records (out_date is NULL or future)
        sql_current = f"""
        SELECT COUNT(DISTINCT ts_code) 
        FROM {table_name}
        WHERE out_date IS NULL OR out_date > CURRENT_DATE
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql_current))
            current_count = result.scalar()

        # Unique industries (L3)
        sql_industries = f"""
        SELECT COUNT(DISTINCT l3_code)
        FROM {table_name}
        WHERE out_date IS NULL OR out_date > CURRENT_DATE
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql_industries))
            industry_count = result.scalar()

        # Date range
        sql_dates = f"""
        SELECT MIN(in_date) as min_date, MAX(in_date) as max_date
        FROM {table_name}
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql_dates))
            row = result.fetchone()
            min_date = row[0].isoformat() if row[0] else None
            max_date = row[1].isoformat() if row[1] else None

        return jsonify({
            "total_records": total_count,
            "current_stocks": current_count,
            "unique_industries": industry_count,
            "date_range": {"min": min_date, "max": max_date},
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def search_stocks():
    """Search stocks by code or name."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        query = request.args.get("q", "").strip()
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        current_only = request.args.get("current_only", "true").lower() == "true"
        target_date = request.args.get("target_date")

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        # Build WHERE clause
        where_clauses = []
        params = {}

        if current_only:
            if target_date:
                where_clauses.append("(out_date IS NULL OR out_date > :target_date)")
                where_clauses.append("in_date <= :target_date")
                params["target_date"] = target_date
            else:
                where_clauses.append("(out_date IS NULL OR out_date > CURRENT_DATE)")

        # Search by code or name
        search_clause = "(ts_code ILIKE :query OR stock_name ILIKE :query)"
        where_clauses.append(search_clause)
        params["query"] = f"%{query}%"

        where_sql = " AND ".join(where_clauses)

        # Get total count
        count_sql = f"""
        SELECT COUNT(DISTINCT ts_code)
        FROM {table_name}
        WHERE {where_sql}
        """
        with engine.connect() as conn:
            result = conn.execute(text(count_sql), params)
            total = result.scalar()

        # Get paginated results
        offset = (page - 1) * per_page
        search_sql = f"""
        SELECT DISTINCT ON (ts_code)
            ts_code,
            stock_name,
            l1_code,
            l1_name,
            l2_code,
            l2_name,
            l3_code,
            l3_name,
            in_date,
            out_date
        FROM {table_name}
        WHERE {where_sql}
        ORDER BY ts_code, in_date DESC
        LIMIT :limit OFFSET :offset
        """
        params["limit"] = per_page
        params["offset"] = offset

        with engine.connect() as conn:
            result = conn.execute(text(search_sql), params)
            rows = result.fetchall()

        stocks = []
        for row in rows:
            stocks.append({
                "ts_code": row[0],
                "stock_name": row[1] or "",
                "l1_code": row[2],
                "l1_name": row[3],
                "l2_code": row[4],
                "l2_name": row[5],
                "l3_code": row[6],
                "l3_name": row[7],
                "in_date": row[8].isoformat() if row[8] else None,
                "out_date": row[9].isoformat() if row[9] else None,
            })

        return jsonify({
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "stocks": stocks,
        })
    except Exception as e:
        logger.error(f"Error searching stocks: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ts_code>")
def get_stock_details(ts_code: str):
    """Get detailed history for a stock."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        sql = f"""
        SELECT 
            ts_code,
            stock_name,
            l1_code,
            l1_name,
            l2_code,
            l2_name,
            l3_code,
            l3_name,
            in_date,
            out_date,
            is_new,
            update_time
        FROM {table_name}
        WHERE ts_code = :ts_code
        ORDER BY in_date DESC
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql), {"ts_code": ts_code})
            rows = result.fetchall()

        history = []
        for row in rows:
            history.append({
                "ts_code": row[0],
                "stock_name": row[1] or "",
                "l1_code": row[2],
                "l1_name": row[3],
                "l2_code": row[4],
                "l2_name": row[5],
                "l3_code": row[6],
                "l3_name": row[7],
                "in_date": row[8].isoformat() if row[8] else None,
                "out_date": row[9].isoformat() if row[9] else None,
                "is_new": row[10],
                "update_time": row[11].isoformat() if row[11] else None,
            })

        return jsonify({
            "ts_code": ts_code,
            "history": history,
        })
    except Exception as e:
        logger.error(f"Error getting stock details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/industries")
def get_industries():
    """Get list of all industries."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        level = request.args.get("level", "l3")  # l1, l2, or l3
        current_only = request.args.get("current_only", "true").lower() == "true"
        target_date = request.args.get("target_date")

        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        where_clauses = []
        params = {}

        if current_only:
            if target_date:
                where_clauses.append("(out_date IS NULL OR out_date > :target_date)")
                where_clauses.append("in_date <= :target_date")
                params["target_date"] = target_date
            else:
                where_clauses.append("(out_date IS NULL OR out_date > CURRENT_DATE)")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        if level == "l1":
            sql = f"""
            SELECT DISTINCT l1_code, l1_name, COUNT(DISTINCT ts_code) as stock_count
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY l1_code, l1_name
            ORDER BY l1_name
            """
        elif level == "l2":
            sql = f"""
            SELECT DISTINCT l2_code, l2_name, COUNT(DISTINCT ts_code) as stock_count
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY l2_code, l2_name
            ORDER BY l2_name
            """
        else:  # l3
            sql = f"""
            SELECT DISTINCT l3_code, l3_name, COUNT(DISTINCT ts_code) as stock_count
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY l3_code, l3_name
            ORDER BY l3_name
            """

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchall()

        industries = []
        for row in rows:
            industries.append({
                "code": row[0],
                "name": row[1],
                "stock_count": row[2],
            })

        return jsonify({
            "level": level,
            "industries": industries,
        })
    except Exception as e:
        logger.error(f"Error getting industries: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/industry/<industry_code>/stocks")
def get_industry_stocks(industry_code: str):
    """Get stocks in a specific industry."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        level = request.args.get("level", "l3")  # l1, l2, or l3
        current_only = request.args.get("current_only", "true").lower() == "true"
        target_date = request.args.get("target_date")

        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        where_clauses = []
        params = {"industry_code": industry_code}

        if level == "l1":
            where_clauses.append("l1_code = :industry_code")
        elif level == "l2":
            where_clauses.append("l2_code = :industry_code")
        else:  # l3
            where_clauses.append("l3_code = :industry_code")

        if current_only:
            if target_date:
                where_clauses.append("(out_date IS NULL OR out_date > :target_date)")
                where_clauses.append("in_date <= :target_date")
                params["target_date"] = target_date
            else:
                where_clauses.append("(out_date IS NULL OR out_date > CURRENT_DATE)")

        where_sql = " AND ".join(where_clauses)

        sql = f"""
        SELECT DISTINCT ON (ts_code)
            ts_code,
            stock_name,
            l1_code,
            l1_name,
            l2_code,
            l2_name,
            l3_code,
            l3_name,
            in_date,
            out_date
        FROM {table_name}
        WHERE {where_sql}
        ORDER BY ts_code, in_date DESC
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchall()

        stocks = []
        for row in rows:
            stocks.append({
                "ts_code": row[0],
                "stock_name": row[1] or "",
                "l1_code": row[2],
                "l1_name": row[3],
                "l2_code": row[4],
                "l2_name": row[5],
                "l3_code": row[6],
                "l3_name": row[7],
                "in_date": row[8].isoformat() if row[8] else None,
                "out_date": row[9].isoformat() if row[9] else None,
            })

        return jsonify({
            "industry_code": industry_code,
            "level": level,
            "stocks": stocks,
        })
    except Exception as e:
        logger.error(f"Error getting industry stocks: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/update", methods=["POST"])
def update_stock_industry():
    """Update stock industry classification."""
    if repo is None:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        data = request.get_json()
        ts_code = data.get("ts_code")
        l3_code = data.get("l3_code")
        l3_name = data.get("l3_name")
        in_date = data.get("in_date")
        out_date = data.get("out_date")

        if not ts_code or not l3_code or not l3_name:
            return jsonify({"error": "ts_code, l3_code, and l3_name are required"}), 400

        # Get L1 and L2 from L3
        engine = repo._get_engine()
        table_name = repo._get_full_table_name()

        # Find existing L1/L2 for this L3 code
        sql_find = f"""
        SELECT l1_code, l1_name, l2_code, l2_name
        FROM {table_name}
        WHERE l3_code = :l3_code
        LIMIT 1
        """
        with engine.connect() as conn:
            result = conn.execute(text(sql_find), {"l3_code": l3_code})
            row = result.fetchone()
            if row:
                l1_code, l1_name, l2_code, l2_name = row
            else:
                return jsonify({"error": f"L3 code {l3_code} not found in database"}), 400

        # Insert or update record
        if in_date:
            in_date_obj = datetime.strptime(in_date, "%Y-%m-%d").date()
        else:
            in_date_obj = datetime.now().date()

        out_date_obj = None
        if out_date:
            out_date_obj = datetime.strptime(out_date, "%Y-%m-%d").date()

        update_data = {
            "ts_code": ts_code,
            "l1_code": l1_code,
            "l1_name": l1_name,
            "l2_code": l2_code,
            "l2_name": l2_name,
            "l3_code": l3_code,
            "l3_name": l3_name,
            "in_date": in_date_obj,
            "out_date": out_date_obj,
        }

        # Use save method which handles upsert
        success = repo.save(update_data)

        if success:
            return jsonify({"success": True, "message": "Stock industry updated successfully"})
        else:
            return jsonify({"error": "Failed to update stock industry"}), 500

    except Exception as e:
        logger.error(f"Error updating stock industry: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Industry Management Console")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="quant",
        help="Database schema (default: quant)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    # Initialize database
    try:
        init_database(args.config_path, args.schema)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1

    # Get template directory
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    # Create directories if they don't exist
    template_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    app.template_folder = str(template_dir)
    app.static_folder = str(static_dir)

    logger.info(f"Starting Industry Management Console on http://{args.host}:{args.port}")
    logger.info(f"Template directory: {template_dir}")
    logger.info(f"Static directory: {static_dir}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())

