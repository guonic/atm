#!/usr/bin/env python3
"""
EDiOS Dashboard - Streamlit Web Application

Launch the EDiOS visualization dashboard using Streamlit.

Usage:
    streamlit run python/examples/edios_dashboard.py

Or directly:
    python python/examples/edios_dashboard.py
"""

import sys
from pathlib import Path

# Add project root and python directory to path
project_root = Path(__file__).parent.parent.parent
python_dir = project_root / "python"

# Add both paths to sys.path
sys.path.insert(0, str(python_dir))
sys.path.insert(0, str(project_root))

# Now import the module
from nq.analysis.edios.visualization import EdiosVisualization

if __name__ == "__main__":
    # Initialize visualization
    viz = EdiosVisualization()
    
    # Run Streamlit app
    viz.run_streamlit_app()

