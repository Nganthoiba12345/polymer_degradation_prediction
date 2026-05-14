"""
Prediction-related functions
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict


def create_prediction_inputs(feature_cols: list, df: pd.DataFrame = None) -> Dict:
    """
    Create input widgets for manual predictions.
    
    Args:
        feature_cols: List of feature column names (original feature names)
        df: Optional dataframe for default values (raw data preferred)
        
    Returns:
        Dict: Dictionary of input values
    """
    input_values = {}
    
    for col in feature_cols:
        # If original dataframe is provided and column exists, choose widget based on dtype
        if df is not None and col in df.columns:
            series = df[col]

            # CATEGORICAL: object or category dtype -> dropdown
            if series.dtype == "object" or str(series.dtype).startswith("category"):
                options = sorted(series.dropna().unique().tolist())
                if not options:
                    options = [""]
                default_idx = 0

                input_values[col] = st.selectbox(
                    f"{col}",
                    options=options,
                    index=default_idx,
                    key=f"input_{col}"
                )
            else:
                # NUMERIC: number input with mean as default
                try:
                    default = float(series.mean())
                except Exception:
                    default = 0.0

                input_values[col] = st.number_input(
                    f"{col}",
                    value=default,
                    format="%.4f",
                    key=f"input_{col}"
                )
        else:
            # Fallback: unknown column -> numeric input
            input_values[col] = st.number_input(
                f"{col}",
                value=0.0,
                format="%.4f",
                key=f"input_{col}"
            )
    
    return input_values


def get_prediction_summary(
    model,
    input_values: Dict,
    target_name: str
) -> Dict:
    """
    Get detailed prediction summary including component predictions.
    
    Args:
        model: Trained ensemble wrapper (e.g., EnsembleWrapper)
        input_values: Dictionary of input feature values
        target_name: Name of target variable
        
    Returns:
        Dict: Prediction summary
    """
    # Create input dataframe
    input_df = pd.DataFrame([input_values])
    
    # Final prediction from full stacking ensemble
    final_pred = model.predict(input_df)[0]
    
    # Component predictions (RF, SVM or LGBM, and stacking/meta)
    rf_pred, other_pred, meta_pred = model.predict_components(input_df)
    
    # Difference between RF and second base model
    diff_rf_other = abs(rf_pred[0] - other_pred[0])
    
    # In your stacking setup, the "meta" model is always used to produce final prediction,
    # so we mark that as the method used.
    summary = {
        "final_prediction": float(final_pred),
        "rf_prediction": float(rf_pred[0]),
        "other_model_prediction": float(other_pred[0]),
        "meta_prediction": float(meta_pred[0]),
        "diff_rf_other": float(diff_rf_other),
        # No real tolerance in this architecture; set to 0 for display consistency
        "tolerance": 0.0,
        "used_meta_model": True,
        "method_used": "Stacking Ensemble (meta model)",
    }
    
    return summary


def display_prediction_results(summary: Dict, target_name: str):
    """
    Display prediction results in an organized format.
    
    Args:
        summary: Prediction summary dictionary
        target_name: Name of target variable
    """
    st.markdown("### Prediction Results")
    
    # Decide units based on target_name
    if "Percent" in target_name or "%" in target_name:
        unit = "%"
    elif "Day" in target_name or "days" in target_name:
        unit = "days"
    else:
        unit = ""
    
    # Main prediction
    if unit:
        st.success(
            f"**Final predicted `{target_name}`:** "
            f"{summary['final_prediction']:.4f} {unit}"
        )
    else:
        st.success(
            f"**Final predicted `{target_name}`:** "
            f"{summary['final_prediction']:.4f}"
        )
    
    # Component predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Random Forest",
            f"{summary['rf_prediction']:.4f}" + (f" {unit}" if unit else ""),
            delta=None
        )
    
    with col2:
        st.metric(
            "Second Base Model (SVM / LGBM)",
            f"{summary['other_model_prediction']:.4f}" + (f" {unit}" if unit else ""),
            delta=None
        )
    
    with col3:
        st.metric(
            "Stacking Meta Model",
            f"{summary['meta_prediction']:.4f}" + (f" {unit}" if unit else ""),
            delta=None
        )
    
    # Decision information
    st.markdown("### Decision Process")
    
    decision_col1, decision_col2 = st.columns(2)
    
    with decision_col1:
        st.info(f"**Difference (RF - Second Model):** {summary['diff_rf_other']:.4f}")
    
    with decision_col2:
        st.info(f"**Agreement Tolerance (not used):** {summary['tolerance']:.4f}")
    
    # Final decision
    st.warning(
        "**Decision:** Final prediction is always taken from the stacking "
        "meta-model, which combines Random Forest and the second base model."
    )
    
    st.info(f"**Method Used:** {summary['method_used']}")
