"""
model.py

Defines the MetaEnsembleSystem used for prediction:
- Random Forest + LightGBM base models
- Optional meta-model that combines them
- Works for regression (e.g. Degradation_Percent, Total_Incubation_Days)
  and classification (if needed in future).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    # Fallback if lightgbm is not installed
    _LGBM_AVAILABLE = False


TaskType = Literal["regression", "classification"]


@dataclass
class MetaEnsembleSystem:
    """
    A small meta-ensemble that combines:
      - Random Forest
      - LightGBM
      - Optional meta-model on top of RF + LGBM

    Attributes
    ----------
    task_type : {"regression", "classification"}
        Determines which base/meta models are used.
    diff_tolerance : float
        Threshold to decide if RF & LGBM predictions "disagree".
        If |RF - LGBM| > diff_tolerance → use meta-model.
        Otherwise → use average of RF & LGBM.
    rf_model : fitted scikit-learn model
    lgbm_model : fitted LightGBM model (or None if unavailable)
    meta_model : fitted meta-model taking [rf_pred, lgbm_pred] as input
    """

    task_type: TaskType = "regression"
    diff_tolerance: float = 0.5

    rf_model: Optional[object] = None
    lgbm_model: Optional[object] = None
    meta_model: Optional[object] = None

    def _build_base_models(self):
        """Create base models according to task type."""
        if self.task_type == "regression":
            self.rf_model = RandomForestRegressor(
                n_estimators=300,
                random_state=0,
            )
            if _LGBM_AVAILABLE:
                self.lgbm_model = LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    random_state=0,
                )
            else:
                # Fallback: use another RF if LightGBM is missing
                self.lgbm_model = RandomForestRegressor(
                    n_estimators=300,
                    random_state=1,
                )
            self.meta_model = LinearRegression()
        else:  # classification
            self.rf_model = RandomForestClassifier(
                n_estimators=300,
                random_state=0,
            )
            if _LGBM_AVAILABLE:
                self.lgbm_model = LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    random_state=0,
                )
            else:
                self.lgbm_model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=1,
                )
            self.meta_model = LogisticRegression(max_iter=1000)

    # ------------------------------------------------------------------ #
    #  FITTING
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        Fit RF, LGBM and meta-model on the given data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self._build_base_models()

        # Fit base models
        self.rf_model.fit(X, y)
        self.lgbm_model.fit(X, y)

        # Build meta-training data = predictions of base models
        rf_pred, lgbm_pred, _ = self._predict_components_raw(X)
        meta_X = np.vstack([rf_pred, lgbm_pred]).T  # shape (n_samples, 2)

        # Fit meta-model
        self.meta_model.fit(meta_X, y)

        return self

    # ------------------------------------------------------------------ #
    #  INTERNAL PREDICTION HELPERS
    # ------------------------------------------------------------------ #
    def _predict_components_raw(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get RF, LGBM and meta raw predictions (without combining).

        Returns
        -------
        rf_pred : np.ndarray, shape (n_samples,)
        lgbm_pred : np.ndarray, shape (n_samples,)
        meta_pred : np.ndarray, shape (n_samples,)
        """
        if self.task_type == "regression":
            rf_pred = self.rf_model.predict(X)
            lgbm_pred = self.lgbm_model.predict(X)
        else:
            # Use class probabilities of the positive class (or class 1)
            rf_prob = self.rf_model.predict_proba(X)
            lgbm_prob = self.lgbm_model.predict_proba(X)
            # Use column 1 if binary classification
            rf_pred = rf_prob[:, 1] if rf_prob.shape[1] > 1 else rf_prob[:, 0]
            lgbm_pred = lgbm_prob[:, 1] if lgbm_prob.shape[1] > 1 else lgbm_prob[:, 0]

        meta_X = np.vstack([rf_pred, lgbm_pred]).T
        meta_pred = self.meta_model.predict(meta_X)

        return rf_pred, lgbm_pred, meta_pred

    # ------------------------------------------------------------------ #
    #  PUBLIC PREDICTION API (used by prediction.py)
    # ------------------------------------------------------------------ #
    def predict_components(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Public function returning component predictions:
        RF prediction, LGBM prediction, Meta prediction.

        This is what `get_prediction_summary` in prediction.py expects.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        (rf_pred, lgbm_pred, meta_pred) : tuple of np.ndarrays
        """
        return self._predict_components_raw(X)

    def predict(self, X) -> np.ndarray:
        """
        Final combined prediction using disagreement logic:

        - Compute RF and LGBM predictions.
        - If |RF - LGBM| <= diff_tolerance → average them.
        - If |RF - LGBM|  > diff_tolerance → use meta-model prediction.

        Parameters
        ----------
        X : array-like or DataFrame

        Returns
        -------
        preds : np.ndarray, shape (n_samples,)
        """
        rf_pred, lgbm_pred, meta_pred = self._predict_components_raw(X)

        # Disagreement magnitude
        diff = np.abs(rf_pred - lgbm_pred)

        # Where RF & LGBM agree → average
        use_avg = diff <= self.diff_tolerance

        final_pred = np.where(use_avg,
                              (rf_pred + lgbm_pred) / 2.0,
                              meta_pred)
        return final_pred

    # ------------------------------------------------------------------ #
    #  SAVE / LOAD HELPERS
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        """
        Save the entire ensemble to disk.
        """
        joblib.dump(
            {
                "task_type": self.task_type,
                "diff_tolerance": self.diff_tolerance,
                "rf_model": self.rf_model,
                "lgbm_model": self.lgbm_model,
                "meta_model": self.meta_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MetaEnsembleSystem":
        """
        Load an ensemble from disk.

        Returns
        -------
        MetaEnsembleSystem
        """
        data = joblib.load(path)
        obj = cls(task_type=data["task_type"],
                  diff_tolerance=data["diff_tolerance"])
        obj.rf_model = data["rf_model"]
        obj.lgbm_model = data["lgbm_model"]
        obj.meta_model = data["meta_model"]
        return obj
