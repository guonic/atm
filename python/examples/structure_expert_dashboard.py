#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Expert Dashboard

基于 Streamlit + Plotly 的交互式可视化 Dashboard
用于评估和可视化 Structure Expert GNN 模型

架构：
- 底座：Streamlit (Web 界面)
- 交互核心：Plotly (交互式图表)
- 数据存储：PostgreSQL / 文件存储
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sklearn.manifold import TSNE

# Streamlit
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Warning: streamlit not installed. Install with: pip install streamlit")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.stock_repo import StockIndustryMemberRepo
from sqlalchemy import text

# Import structure expert model
tools_train_dir = Path(__file__).parent.parent / "tools" / "qlib" / "train"
if str(tools_train_dir) not in sys.path:
    sys.path.insert(0, str(tools_train_dir))

from structure_expert import GraphDataBuilder, StructureExpertGNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Storage backend
STORAGE_BACKEND = "file"  # "postgresql" or "file"
STORAGE_DIR = Path("storage/structure_expert_cache")


def load_industry_label_map(
    db_config: Optional[DatabaseConfig] = None,
    target_date: Optional[datetime] = None,
    schema: str = "quant",
) -> Dict[str, str]:
    """
    Load industry label mapping from database.
    
    Returns:
        Dictionary mapping stock codes to industry names.
        Format: {stock_code: "行业名称"}
    """
    if db_config is None:
        # Try to load from default config
        try:
            config = load_config("config/config.yaml")
            db_config = config.database
        except Exception as e:
            logger.warning(f"Could not load database config: {e}")
            return {}
    
    repo = StockIndustryMemberRepo(db_config, schema=schema)
    
    if target_date is None:
        target_date = datetime.now()
    
    engine = repo._get_engine()
    table_name = repo._get_full_table_name()
    
    # Query stock codes with L3 industry names
    sql = f"""
    SELECT DISTINCT 
        sim.ts_code,
        sic.industry_name as l3_name
    FROM {table_name} sim
    LEFT JOIN quant.stock_industry_classify sic 
        ON sim.l3_code = sic.index_code 
        AND sic.level = 'L3'
        AND sic.src = 'SW2021'
    WHERE (sim.out_date IS NULL OR sim.out_date > :target_date)
      AND sim.in_date <= :target_date
    ORDER BY sim.ts_code
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(sql),
                {"target_date": target_date.date()},
            )
            rows = result.fetchall()
        
        # Convert to Qlib format
        label_map = {}
        for ts_code, l3_name in rows:
            qlib_code = convert_ts_code_to_qlib_format(ts_code)
            label_map[qlib_code] = l3_name if l3_name else "Unknown"
        
        logger.info(f"Loaded industry labels: {len(label_map)} stocks")
        return label_map
    except Exception as e:
        logger.error(f"Failed to load industry labels: {e}")
        return {}


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """Convert ts_code to Qlib format."""
    # Qlib uses same format: 000001.SZ -> 000001.SZ
    return ts_code


def load_model(
    model_path: str,
    n_feat: int = 158,
    n_hidden: int = 128,
    n_heads: int = 8,
    device: str = "cpu",
) -> StructureExpertGNN:
    """Load trained Structure Expert model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = StructureExpertGNN(n_feat=n_feat, n_hidden=n_hidden, n_heads=n_heads)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(device_obj)
    model.eval()
    
    logger.info(f"Model loaded on {device_obj}")
    return model


def extract_embeddings_and_scores(
    model: StructureExpertGNN,
    daily_graph,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings and scores from model.
    
    Returns:
        (embeddings, scores) tuple
    """
    model.eval()
    device_obj = torch.device(device)
    
    with torch.no_grad():
        # Move graph to device
        x = daily_graph.x.to(device_obj)
        edge_index = daily_graph.edge_index.to(device_obj)
        
        # Forward pass
        logits, embedding = model(x, edge_index)
        
        scores = logits.squeeze().cpu().numpy()
        emb_np = embedding.cpu().numpy()
    
    return emb_np, scores


def visualize_structure_expert(
    embeddings: np.ndarray,
    scores: np.ndarray,
    symbols: List[str],
    industry_label_map: Optional[Dict[str, str]] = None,
    title: str = "Structure Expert Visualization (t-SNE)",
) -> go.Figure:
    """
    Create interactive t-SNE visualization using Plotly.
    
    Args:
        embeddings: Node embeddings from model (N x D)
        scores: Prediction scores (N,)
        symbols: Stock symbols (N,)
        industry_label_map: Mapping from symbol to industry name
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Run t-SNE
    logger.info("Running t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    components = tsne.fit_transform(embeddings)
    
    # Prepare data
    df_vis = pd.DataFrame({
        'x': components[:, 0],
        'y': components[:, 1],
        'symbol': symbols,
        'score': scores,
        'industry': [industry_label_map.get(s, "Unknown") if industry_label_map else "Unknown" for s in symbols],
    })
    
    # Create interactive scatter plot
    fig = px.scatter(
        df_vis,
        x='x',
        y='y',
        color='industry',
        size=np.abs(df_vis['score']) + 0.1,
        hover_name='symbol',
        hover_data={'score': ':.4f', 'x': False, 'y': False},
        title=title,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    
    # Optimize layout
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    
    return fig


def save_embeddings_to_file(
    date: str,
    symbols: List[str],
    embeddings: np.ndarray,
    scores: np.ndarray,
    industry_labels: Optional[Dict[str, str]] = None,
) -> Path:
    """Save embeddings and scores to file."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'symbol': symbols,
        'score': scores,
        'industry': [industry_labels.get(s, "Unknown") if industry_labels else "Unknown" for s in symbols],
    })
    
    # Add embedding columns
    for i in range(embeddings.shape[1]):
        df[f'embedding_{i}'] = embeddings[:, i]
    
    # Save to parquet (efficient) or CSV
    output_path = STORAGE_DIR / f"embeddings_{date}.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved embeddings to {output_path}")
    return output_path


def load_embeddings_from_file(date: str) -> Optional[pd.DataFrame]:
    """Load embeddings from file."""
    file_path = STORAGE_DIR / f"embeddings_{date}.parquet"
    if file_path.exists():
        return pd.read_parquet(file_path)
    return None


def save_embeddings_to_postgresql(
    date: str,
    symbols: List[str],
    embeddings: np.ndarray,
    scores: np.ndarray,
    industry_labels: Optional[Dict[str, str]] = None,
    db_config: Optional[DatabaseConfig] = None,
) -> None:
    """Save embeddings to PostgreSQL database."""
    if db_config is None:
        logger.error("Database config required for PostgreSQL storage")
        return
    
    # TODO: Implement PostgreSQL storage
    # Create table if not exists
    # Insert embeddings
    logger.warning("PostgreSQL storage not yet implemented")


def main_streamlit():
    """Main Streamlit application."""
    if not HAS_STREAMLIT:
        st.error("Streamlit is not installed. Please install with: pip install streamlit")
        st.stop()
    
    st.set_page_config(
        page_title="Structure Expert Dashboard",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Structure Expert Model Visualization Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/structure_expert.pth",
        help="Path to trained model file (.pth)",
    )
    
    # Date selection
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.now().date(),
        help="Date for visualization",
    )
    
    # Storage backend selection
    storage_backend = st.sidebar.selectbox(
        "Storage Backend",
        options=["file", "postgresql"],
        index=0,
        help="Choose storage backend",
    )
    
    # Load database config if PostgreSQL selected
    db_config = None
    if storage_backend == "postgresql":
        config_path = st.sidebar.text_input(
            "Config Path",
            value="config/config.yaml",
            help="Path to database config file",
        )
        try:
            config = load_config(config_path)
            db_config = config.database
            st.sidebar.success("✓ Database config loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load config: {e}")
            storage_backend = "file"
            st.sidebar.warning("Falling back to file storage")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📈 Visualization", "💾 Data Management", "📊 Statistics"])
    
    with tab1:
        st.header("Interactive Visualization")
        
        # Check if model exists
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            st.info("Please provide a valid model path in the sidebar")
            st.stop()
        
        # Load model button
        if st.button("🔄 Load Model & Generate Visualization", type="primary"):
            with st.spinner("Loading model and generating visualization..."):
                try:
                    # Load model
                    model = load_model(model_path, device="cpu")
                    
                    # Load industry labels
                    industry_label_map = {}
                    if db_config:
                        industry_label_map = load_industry_label_map(db_config, target_date=selected_date)
                    
                    # Check if embeddings are cached
                    date_str = selected_date.strftime("%Y%m%d")
                    cached_data = None
                    if storage_backend == "file":
                        cached_data = load_embeddings_from_file(date_str)
                    
                    if cached_data is not None:
                        st.success(f"✓ Loaded cached embeddings for {selected_date}")
                        # Extract data from cache
                        symbols = cached_data['symbol'].tolist()
                        scores = cached_data['score'].values
                        embedding_cols = [col for col in cached_data.columns if col.startswith('embedding_')]
                        embeddings = cached_data[embedding_cols].values
                        
                        # Create visualization
                        fig = visualize_structure_expert(
                            embeddings=embeddings,
                            scores=scores,
                            symbols=symbols,
                            industry_label_map=industry_label_map,
                            title=f"Structure Expert Visualization - {selected_date}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("⚠️ No cached data found. Please run backtest first to generate embeddings.")
                        st.info("Use the 'Data Management' tab to generate and cache embeddings.")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Error in visualization")
    
    with tab2:
        st.header("Data Management")
        
        st.subheader("Generate Embeddings")
        st.markdown("Generate and cache embeddings from a specific date.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gen_date = st.date_input(
                "Date for Embedding Generation",
                value=datetime.now().date(),
                key="gen_date",
            )
        
        with col2:
            gen_device = st.selectbox(
                "Device",
                options=["cpu", "cuda"],
                index=0,
                help="Device to run model inference",
            )
        
        if st.button("🚀 Generate Embeddings", type="primary"):
            if not Path(model_path).exists():
                st.error(f"Model file not found: {model_path}")
            else:
                with st.spinner("Generating embeddings..."):
                    try:
                        # Load model
                        model = load_model(model_path, device=gen_device)
                        
                        # Load industry labels
                        industry_label_map = {}
                        if db_config:
                            industry_label_map = load_industry_label_map(db_config, target_date=gen_date)
                        
                        # TODO: Build graph for selected date
                        # This requires integrating with GraphDataBuilder and Qlib data
                        st.info("⚠️ Graph building integration needed. Please use backtest script to generate embeddings.")
                        st.code("""
# Example usage:
python python/examples/backtest_structure_expert.py \\
    --model_path models/structure_expert.pth \\
    --start_date 2025-01-02 \\
    --end_date 2025-01-02 \\
    --save_embeddings
                        """)
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.exception("Error generating embeddings")
        
        st.subheader("Cached Data")
        st.markdown("View cached embedding files.")
        
        if storage_backend == "file":
            cache_files = list(STORAGE_DIR.glob("embeddings_*.parquet"))
            if cache_files:
                st.success(f"Found {len(cache_files)} cached files")
                for cache_file in sorted(cache_files, reverse=True)[:10]:
                    st.text(str(cache_file.name))
            else:
                st.info("No cached files found")
    
    with tab3:
        st.header("Statistics")
        st.markdown("View statistics about cached embeddings.")
        
        if storage_backend == "file":
            cache_files = list(STORAGE_DIR.glob("embeddings_*.parquet"))
            if cache_files:
                st.metric("Total Cached Files", len(cache_files))
                
                # Load and display stats
                latest_file = sorted(cache_files, reverse=True)[0]
                df = pd.read_parquet(latest_file)
                
                st.subheader(f"Latest File: {latest_file.name}")
                st.dataframe(df.head(20))
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stocks", len(df))
                with col2:
                    st.metric("Unique Industries", df['industry'].nunique())
                with col3:
                    st.metric("Avg Score", f"{df['score'].mean():.4f}")
            else:
                st.info("No cached files found")


def main_cli():
    """CLI interface for embedding generation."""
    parser = argparse.ArgumentParser(description="Structure Expert Dashboard CLI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--storage", type=str, default="file", choices=["file", "postgresql"], help="Storage backend")
    parser.add_argument("--config_path", type=str, help="Database config path (for PostgreSQL)")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load industry labels
    db_config = None
    if args.storage == "postgresql":
        if args.config_path:
            config = load_config(args.config_path)
            db_config = config.database
        else:
            logger.error("--config_path required for PostgreSQL storage")
            return
    
    industry_label_map = load_industry_label_map(db_config, target_date=datetime.strptime(args.date, "%Y-%m-%d"))
    
    logger.info("CLI mode: Use Streamlit dashboard for interactive visualization")
    logger.info("Run: streamlit run python/examples/structure_expert_dashboard.py")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        if HAS_STREAMLIT:
            main_streamlit()
        else:
            print("Streamlit not installed. Install with: pip install streamlit")
            print("Or use CLI mode: python structure_expert_dashboard.py --help")

