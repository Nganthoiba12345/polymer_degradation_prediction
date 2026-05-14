"""
Visualization functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score


def plot_true_vs_pred(
    y_true,
    y_pred,
    title="True vs Predicted",
    target_name: str = "",
):
    """
    Scatter plot comparing true vs. predicted values.
    Automatically adds:
      - Best-fit diagonal line
      - R² score display
      - Units (% or days) based on target_name
    """

    # Determine unit based on target
    if "Percent" in target_name or "%" in target_name:
        unit = "%"
    elif "Day" in target_name or "day" in target_name:
        unit = "days"
    else:
        unit = ""

    # Compute R²
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = None

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')

    # Diagonal line
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)

    # Labels
    ax.set_title(title, fontsize=14, weight="bold")

    ax.set_xlabel(f"True {target_name} {unit}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_name} {unit}", fontsize=12)

    # Show R² text inside plot
    if r2 is not None:
        ax.text(
            0.05, 0.95,
            f"R² = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7)
        )

    st.pyplot(fig)
