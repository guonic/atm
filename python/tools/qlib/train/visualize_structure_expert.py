#!/usr/bin/env python3
"""
Streamlit visualization app for Structure Expert GNN embeddings.

This app visualizes stock embeddings extracted from the Structure Expert GNN model
using t-SNE for dimensionality reduction.

Usage:
    streamlit run python/tools/qlib/train/visualize_structure_expert.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qlib
import streamlit as st
import torch
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from sklearn.manifold import TSNE

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.utils.industry import load_industry_map

# Import structure expert model using standard package import
from tools.qlib.train.structure_expert import GraphDataBuilder, StructureExpertGNN, StructureTrainer


@st.cache_data
def load_model_and_data(
    model_path: Optional[str],
    qlib_dir: str,
    target_date: str,
    db_config: DatabaseConfig,
    schema: str = "quant",
) -> tuple:
    """
    Load model and prepare data for visualization.

    Returns:
        Tuple of (model, trainer, builder, daily_graph, stock_symbols)
    """
    # Initialize Qlib
    qlib.init(provider_uri=qlib_dir, region="cn")

    # Load industry mapping
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    industry_map = load_industry_map(db_config, target_date=target_dt, schema=schema)

    # Initialize model
    n_feat = 158  # Alpha158 features
    model = StructureExpertGNN(n_feat=n_feat)

    # Load model weights if provided
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success(f"Model loaded from {model_path}")
    else:
        st.warning("Using untrained model (random weights)")

    trainer = StructureTrainer(model, device="cpu")
    builder = GraphDataBuilder(industry_map)

    # Load data for target date
    instruments = D.instruments()
    if not isinstance(instruments, list):
        instruments = list(instruments)

    # Load Alpha158 features
    handler = Alpha158(
        start_time=target_date,
        end_time=target_date,
    )
    df_x = handler.fetch(col_set="feature", data_key="train")

    if df_x.empty:
        st.error(f"No data available for {target_date}")
        return None, None, None, None, None

    # Build graph
    daily_graph = builder.get_daily_graph(df_x, None)
    stock_symbols = daily_graph.symbols if hasattr(daily_graph, "symbols") else []

    return model, trainer, builder, daily_graph, stock_symbols


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Structure Expert GNN Visualization",
        page_icon="📊",
        layout="wide",
    )

    st.title("Structure Expert GNN Embedding Visualization")
    st.markdown(
        """
    This app visualizes stock embeddings extracted from the Structure Expert GNN model.
    The embeddings capture industry relationships and can be used for stock clustering and analysis.
    """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Load config
    config_path = st.sidebar.text_input(
        "Config Path", value="config/config.yaml", help="Path to config file"
    )
    schema = st.sidebar.text_input("Database Schema", value="quant")

    try:
        config = load_config(config_path)
        db_config = config.database
    except Exception as e:
        st.sidebar.error(f"Failed to load config: {e}")
        st.stop()

    # Qlib directory
    qlib_dir = st.sidebar.text_input(
        "Qlib Data Directory",
        value="~/.qlib/qlib_data/cn_data",
        help="Path to Qlib data directory",
    )
    qlib_dir = str(Path(qlib_dir).expanduser())

    # Model path (optional)
    model_path = st.sidebar.text_input(
        "Model Path (Optional)",
        value="",
        help="Path to trained model checkpoint (leave empty for random weights)",
    )

    # Target date
    target_date = st.sidebar.date_input(
        "Target Date",
        value=datetime.now().date(),
        help="Date to extract embeddings for",
    )

    # t-SNE parameters
    st.sidebar.header("t-SNE Parameters")
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30, help="t-SNE perplexity parameter")
    n_iter = st.sidebar.slider("Iterations", 250, 2000, 1000, step=250)

    # Load button
    if st.sidebar.button("Load Data and Model", type="primary"):
        with st.spinner("Loading model and data..."):
            result = load_model_and_data(
                model_path if model_path else None,
                qlib_dir,
                target_date.strftime("%Y-%m-%d"),
                db_config,
                schema,
            )
            if result[0] is not None:
                st.session_state["model"] = result[0]
                st.session_state["trainer"] = result[1]
                st.session_state["builder"] = result[2]
                st.session_state["daily_graph"] = result[3]
                st.session_state["stock_symbols"] = result[4]
                st.success("Data and model loaded successfully!")

    # Main content
    if "daily_graph" not in st.session_state:
        st.info("Please configure and load data in the sidebar.")
        st.stop()

    # Extract embeddings
    st.header("Extract Embeddings")
    if st.button("Extract Embeddings"):
        with st.spinner("Extracting embeddings..."):
            trainer = st.session_state["trainer"]
            daily_graph = st.session_state["daily_graph"]

            embeddings = trainer.get_embeddings(daily_graph)
            stock_symbols = st.session_state["stock_symbols"]

            st.session_state["embeddings"] = embeddings
            st.session_state["stock_symbols"] = stock_symbols

            st.success(f"Extracted embeddings for {len(stock_symbols)} stocks")
            st.write(f"Embedding shape: {embeddings.shape}")

    # Visualize embeddings
    if "embeddings" in st.session_state:
        st.header("Visualize Embeddings")

        embeddings = st.session_state["embeddings"]
        stock_symbols = st.session_state["stock_symbols"]

        # Apply t-SNE
        if st.button("Apply t-SNE"):
            with st.spinner("Applying t-SNE..."):
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings)

                st.session_state["embeddings_2d"] = embeddings_2d
                st.success("t-SNE applied successfully!")

        # Plot
        if "embeddings_2d" in st.session_state:
            embeddings_2d = st.session_state["embeddings_2d"]

            # Create DataFrame for plotting
            plot_df = pd.DataFrame(
                {
                    "x": embeddings_2d[:, 0],
                    "y": embeddings_2d[:, 1],
                    "stock": stock_symbols,
                }
            )

            # Create interactive plot
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                hover_data=["stock"],
                title="Stock Embeddings Visualization (t-SNE)",
                labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2"},
            )

            fig.update_traces(marker=dict(size=5, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stocks", len(stock_symbols))
            with col2:
                st.metric("Embedding Dimension", embeddings.shape[1])
            with col3:
                st.metric("2D Projection", "t-SNE")

            # Download data
            st.download_button(
                label="Download Embeddings CSV",
                data=plot_df.to_csv(index=False),
                file_name=f"embeddings_{target_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

