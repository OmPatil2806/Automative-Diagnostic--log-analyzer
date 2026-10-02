"""Train and apply ML anomaly detection (Isolation Forest) on sensor features.

One detector per feature group (engine, speed, fuel, air, battery), trained
only on normal driving. Each group combines two scores, both scaled so that
1.0 is the edge of normal behaviour:

- Isolation Forest score: catches unusual *combinations* of features. Divided
  by a high quantile of its scores on normal data.
- Range score: how far each feature is beyond its normal band, measured in
  multiples of the distance from the median to that band's edge (with a
  minimum width per feature, see FEATURE_MIN_WIDTH). Isolation
  Forest cannot extrapolate (a value far beyond anything in training scores
  about the same as the most extreme training value), so this catches large
  but simple shifts such as a +20% fuel trim.

The group score is the higher of the two. The overall score is the highest
group score, and that group is reported as the suspect subsystem.

Single readings are noisy, so an alarm is raised only when most of the recent
readings in a trip are anomalous (`persistence`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from analyzer.features.engineering import (
    FEATURE_GROUPS,
    FEATURE_MIN_WIDTH,
    TRIP_KEYS,
    VehicleBaseline,
    build_features,
)

MIN_TRAINING_ROWS = 500


@dataclass
class AnomalyDetector:
    threshold_quantile: float = 0.995
    persistence_window: int = 30
    persistence_ratio: float = 0.7
    n_estimators: int = 200
    random_state: int = 42
    baseline: VehicleBaseline = field(default_factory=VehicleBaseline)
    models: dict[str, IsolationForest] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    ranges: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, normal_logs: pd.DataFrame) -> "AnomalyDetector":
        """Learn normal behaviour from fault-free logs."""
        self.baseline = VehicleBaseline().fit(normal_logs)
        features = build_features(normal_logs, self.baseline)
        for group, columns in FEATURE_GROUPS.items():
            X = features[columns].dropna()
            if len(X) < MIN_TRAINING_ROWS:
                continue
            model = IsolationForest(
                n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
            ).fit(X)
            self.models[group] = model
            self.thresholds[group] = float(np.quantile(-model.score_samples(X), self.threshold_quantile))
            tail = (1 - self.threshold_quantile) / 2
            median = X.median()
            min_width = pd.Series({c: FEATURE_MIN_WIDTH.get(c, 0.0) for c in columns})
            self.ranges[group] = pd.DataFrame({
                "low": np.minimum(X.quantile(tail), median - min_width),
                "median": median,
                "high": np.maximum(X.quantile(1 - tail), median + min_width),
            })
        if not self.models:
            raise ValueError("Not enough normal data to train any detector.")
        return self

    def score(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Score every row. Returns keys, per-group scores, overall score, suspect group, alarm."""
        features = build_features(logs, self.baseline)
        out = features[TRIP_KEYS + ["time_ms"]].copy()
        for group, model in self.models.items():
            columns = FEATURE_GROUPS[group]
            X = features[columns].dropna()
            scores = pd.Series(np.nan, index=features.index)
            if len(X):
                forest = -model.score_samples(X) / self.thresholds[group]
                scores[X.index] = np.maximum(forest, self._range_score(X, self.ranges[group]))
            out[f"score_{group}"] = scores

        score_cols = [f"score_{g}" for g in self.models]
        group_scores = out[score_cols]
        out["anomaly_score"] = group_scores.max(axis=1)
        has_score = group_scores.notna().any(axis=1)
        out["suspect_group"] = None
        out.loc[has_score, "suspect_group"] = (
            group_scores[has_score].idxmax(axis=1).str.removeprefix("score_")
        )
        out["is_anomalous"] = out["anomaly_score"] > 1.0

        recent = (
            out["is_anomalous"].astype(float)
            .groupby([out["vehicle_id"], out["trip_id"]])
            .rolling(self.persistence_window, min_periods=self.persistence_window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        out["alarm"] = recent.reindex(out.index).fillna(0) >= self.persistence_ratio
        return out

    @staticmethod
    def _range_score(X: pd.DataFrame, ranges: pd.DataFrame) -> np.ndarray:
        """Max over features of distance beyond the median, in units of (band edge - median).

        Values on the median score 0, values on a band edge score 1. A side
        with zero width is ignored (only possible if FEATURE_MIN_WIDTH is 0).
        """
        med, low, high = (ranges[c].to_numpy() for c in ("median", "low", "high"))
        values = X.to_numpy()
        up_width, down_width = high - med, med - low
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(up_width > 0, (values - med) / up_width, 0)
            down = np.where(down_width > 0, (med - values) / down_width, 0)
        return np.maximum(up, down).max(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "AnomalyDetector":
        return joblib.load(path)
