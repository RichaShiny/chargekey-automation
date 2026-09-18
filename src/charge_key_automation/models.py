from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def model_panel(random_state: int = 42) -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        "svm_linear": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=random_state), cv=3
        ),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def train_attribute_panel(
    training: pd.DataFrame,
    embeddings: np.ndarray,
    attribute_columns: list[str],
    min_class_count: int,
) -> dict[str, dict[str, object]]:
    panel: dict[str, dict[str, object]] = {}
    for column in attribute_columns:
        numeric = pd.to_numeric(training[column], errors="coerce")
        valid = numeric.notna()
        if not valid.any():
            continue
        y = numeric.loc[valid].astype(int).to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < min_class_count:
            continue
        X = embeddings[valid.to_numpy()]
        fitted: dict[str, object] = {}
        for name, classifier in model_panel().items():
            try:
                classifier.fit(X, y)
                fitted[name] = classifier
            except Exception:
                continue
        if fitted:
            panel[column] = fitted
    return panel


def positive_class_probability(classifier: object, rows: np.ndarray) -> np.ndarray:
    try:
        probability = classifier.predict_proba(rows)
        if probability.shape[1] == 2:
            return probability[:, 1].astype(float)
    except Exception:
        pass
    return np.asarray(classifier.predict(rows), dtype=float)
