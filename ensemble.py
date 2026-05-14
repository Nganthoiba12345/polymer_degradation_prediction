import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

from config import ENSEMBLE_CONFIG, RF_PARAMS, SVM_PARAMS, LGBM_PARAMS, META_MODEL_PARAMS


def _build_preprocessor(X: pd.DataFrame):
    """
    Build preprocessing pipeline:
    - Numeric: median imputation + standard scaling
    - Categorical: most_frequent imputation + one-hot encoding
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []

    if numeric_features:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def train_ensemble_system(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = None,
    random_state: int = None,
    large_data_threshold: int = None,
):
    """
    Train an ensemble system:

    - Always train RandomForest.
    - If n_samples < large_data_threshold:
          train SVM (NO LightGBM).
    - If n_samples >= large_data_threshold:
          train LightGBM (NO SVM).
    - Stack them with a Ridge meta-model.

    Also compute 5-fold cross-validation R² using RandomForest
    (with the same preprocessing).
    """
    if test_size is None:
        test_size = ENSEMBLE_CONFIG["test_size"]
    if random_state is None:
        random_state = ENSEMBLE_CONFIG["random_state"]
    if large_data_threshold is None:
        large_data_threshold = ENSEMBLE_CONFIG["large_data_threshold"]

    # 1) Remove rows where target is NaN
    df = df.dropna(subset=[target_col]).copy()

    # Split X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    n_samples = X.shape[0]
    is_large = n_samples >= large_data_threshold

    # 2) 5-fold Cross-Validation with RF + preprocessing
    preprocessor_cv = _build_preprocessor(X)
    rf_cv = Pipeline(
        steps=[
            ("prep", preprocessor_cv),
            (
                "model",
                RandomForestRegressor(
                    **RF_PARAMS,
                ),
            ),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf_cv, X, y, cv=kf, scoring="r2")
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std = float(cv_scores.std())

    # 3) Train/Test split for stacking model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor(X)

    # RandomForest (always)
    rf_pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    **RF_PARAMS,
                ),
            ),
        ]
    )

    estimators = [("rf", rf_pipe)]
    used_models = ["RandomForest"]

    svm_pipe = None
    lgbm_pipe = None

    # Either SVM or LightGBM depending on size
    if not is_large:
        # SMALL DATASET: add SVM, no LightGBM
        svm_pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", SVR(**SVM_PARAMS)),
            ]
        )
        estimators.append(("svm", svm_pipe))
        used_models.append("SVM")
    else:
        # LARGE DATASET: add LightGBM, no SVM
        lgbm_pipe = Pipeline([
            ("prep", preprocessor),
            ("model", LGBMRegressor(**LGBM_PARAMS)),
                 ])
        estimators.append(("lgbm", lgbm_pipe))
        used_models.append("LightGBM")

    meta_model = Ridge(**META_MODEL_PARAMS)

    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        n_jobs=-1,
        passthrough=False,
    )

    # 4) Fit full stacking model
    stack_model.fit(X_train, y_train)

    fitted_base = stack_model.named_estimators_

    # RF R² (always exists)
    rf_r2 = r2_score(y_test, fitted_base["rf"].predict(X_test))

    # SVM / LGBM R²
    if "svm" in fitted_base:
        svm_r2 = r2_score(y_test, fitted_base["svm"].predict(X_test))
    else:
        svm_r2 = None

    if "lgbm" in fitted_base:
        lgbm_r2 = r2_score(y_test, fitted_base["lgbm"].predict(X_test))
    else:
        lgbm_r2 = None

    # Final ensemble metrics
    final_pred = stack_model.predict(X_test)
    final_r2 = r2_score(y_test, final_pred)
    final_mae = mean_absolute_error(y_test, final_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

    metrics = {
        "rf_r2": rf_r2,
        "svm_r2": svm_r2,
        "lgbm_r2": lgbm_r2,
        "final_r2": final_r2,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "used_models": used_models,
        "is_large_dataset": is_large,
        "n_samples": n_samples,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "cv_scores": cv_scores.tolist(),
    }

    return stack_model, X_test, y_test, metrics


def get_detailed_predictions(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Per-sample predictions from each model.
    """
    fitted_base = model.named_estimators_
    out = {}

    if "rf" in fitted_base:
        out["RF Prediction"] = fitted_base["rf"].predict(X_test)
    if "svm" in fitted_base:
        out["SVM Prediction"] = fitted_base["svm"].predict(X_test)
    if "lgbm" in fitted_base:
        out["LGBM Prediction"] = fitted_base["lgbm"].predict(X_test)

    out["Final Prediction"] = model.predict(X_test)
    return pd.DataFrame(out, index=X_test.index)
