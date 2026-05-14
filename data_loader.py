"""
Data loader + preprocessing (updated for degradation % + days prediction)
"""

import pandas as pd
import numpy as np


def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Prepares data for modeling:
    - Computes Degradation_Percent automatically (70% CO2 + 30% WaterAbsorb)
    - Removes 'Control' rows (soil-only, no polymer)
    - Splits into X (features) and y (target)
    - One-hot encodes categorical columns using pandas.get_dummies
    - Returns encoded X, y, and feature_names
    """

    df = df.copy()

    # -------------------------------------------------------------
    # 🔹 1) FIX WATER ABSORB BEFORE ANYTHING
    # -------------------------------------------------------------
    df["WaterAbsorb"] = pd.to_numeric(
        df["WaterAbsorb"].replace("-", np.nan),
        errors="coerce"
    ).fillna(0)

    # -------------------------------------------------------------
    # 🔹 2) COMPUTE DEGRADATION_PERCENT (Option B)
    # -------------------------------------------------------------
    if "Degradation_Percent" not in df.columns:
        max_co2 = df["CO2_ppm"].max()
        max_wa = df["WaterAbsorb"].max()
        # Avoid division by zero
        if max_co2 == 0:
            max_co2 = 1
        if max_wa == 0:
            max_wa = 1
        df["Degradation_Percent"] = (
            0.7 * (df["CO2_ppm"] / max_co2) * 100 +
            0.3 * (df["WaterAbsorb"] / max_wa) * 100
        ).round(2)

    # -------------------------------------------------------------
    # 🔹 3) REMOVE CONTROL ROWS — essential for polymer learning
    # -------------------------------------------------------------
    control_filters = ["Polymer_name", "Polymer", "Shape", "Degradation"]
    for col in control_filters:
        if col in df.columns:
            df = df[df[col] != "Control"]

    # -------------------------------------------------------------
    # 🔹 4) Ensure the target column exists
    # -------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # -------------------------------------------------------------
    # 🔹 5) REMOVE LEAKAGE FEATURES (used to compute Degradation_Percent)
    # -------------------------------------------------------------
    leakage_features = ['CO2_ppm', 'WaterAbsorb']
    X = X.drop(columns=[c for c in leakage_features if c in X.columns], errors='ignore')

    # -------------------------------------------------------------
    # 🔹 6) One-hot encode categorical features
    # -------------------------------------------------------------
    categorical_cols = ["Shape", "Polymer", "Polymer_name", "Degradation"]
    # Only encode columns that actually exist
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # Perform one-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names