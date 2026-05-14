"""
Configuration settings for the Polymer Degradation Ensemble Application
-----------------------------------------------------------------------

This file centralizes all parameters used for:
- Ensemble model training
- Random Forest / SVM / LightGBM settings
- Stacking meta-model settings
- Preprocessing behavior
- Degradation % computation weights (CO2 vs WaterAbsorb)
- Application display preferences
"""

# ======================================================================
# Ensemble Core Settings
# ======================================================================

ENSEMBLE_CONFIG = {
    # Train/test split ratio for model evaluation
    "test_size": 0.20,

    # Ensures reproducible results
    "random_state": 42,

    # If dataset >= this size → use LightGBM instead of SVM
    "large_data_threshold": 500,

    # Tolerance used in older meta-ensemble logic (not needed for stacking)
    # Keeping for compatibility with prediction UI
    "agreement_tolerance": 0.20,
}

# ======================================================================
# Random Forest Parameters
# ======================================================================

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,
}

# ======================================================================
# SVM Parameters (used when dataset < large_data_threshold)
# ======================================================================

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.1,
    "gamma": "scale",
}

# ======================================================================
# LightGBM Parameters (used when dataset >= large_data_threshold)
# ======================================================================

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_bin": 63, 
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# ======================================================================
# Meta-Model (Stacking Final Estimator)
# ======================================================================

META_MODEL_PARAMS = {
    "alpha": 1.0,        # Ridge regression regularization
}

# ======================================================================
# Preprocessing Behavior
# ======================================================================

PREPROCESSING_CONFIG = {
    # Strategy for numeric missing values
    "numeric_imputation": "median",

    # Strategy for categorical missing values
    "categorical_imputation": "most_frequent",

    # Scaling behavior for numeric features
    "scale_numeric": True,
}

# ======================================================================
# Degradation % Computation (used when column missing)
# Option B → 70% CO₂ and 30% WaterAbsorb
# ======================================================================

DEGRADATION_PERCENT_WEIGHTS = {
    "CO2_weight": 0.70,
    "WaterAbsorb_weight": 0.30,
}

# ======================================================================
# Degradation Curve Settings
# ======================================================================

DEGRADATION_CURVE = {
    "reference_days": 90.0,    # T_REF used in kinetic model
    "max_curve_points": 60,    # How smooth the curve looks
}

# ======================================================================
# Streamlit Application Settings
# ======================================================================

APP_CONFIG = {
    "max_table_rows": 50,
    "decimal_places": 4,
    "allow_download_csv": True,
    "show_confidence_scores": True,
}

