from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from constants import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS

@dataclass(frozen=True)
class SingleModelReport:
    model_name: str
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    false_positive_count: int
    false_negative_count: int


@dataclass(frozen=True, order=True)
class ThresholdScore:
    fp_count: float
    fn_count: float
    neg_precision: float


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    selected_model_name: str
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    feature_importance: pd.DataFrame
    model_comparison: pd.DataFrame
    model_path: Path
    reports: Dict[str, SingleModelReport]


def _extract_feature_importance(model: object, model_name: str) -> pd.Series:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    return pd.Series(importances, index=FEATURE_COLUMNS, name=model_name)


def _select_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.5, 0.95, 19)
    best_threshold = 0.5
    best_score = ThresholdScore(fp_count=float("inf"), fn_count=float("inf"), neg_precision=float("inf"))
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = float(tp / max(tp + fp, 1))
        score = ThresholdScore(fp_count=float(fp), fn_count=float(fn), neg_precision=-precision)
        if score < best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _evaluate_model(
    model_name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> SingleModelReport:
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = _select_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }
    return SingleModelReport(
        model_name=model_name,
        threshold=threshold,
        metrics=metrics,
        confusion_matrix=cm,
        false_positive_count=int(fp),
        false_negative_count=int(fn),
    )


def train_fraud_model(
    dataset: pd.DataFrame,
    model_path: str | Path = "model.pkl",
    random_state: int = 42,
) -> ModelArtifacts:
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    honest_train_count = int((y_train == 0).sum())
    attacker_train_count = int((y_train == 1).sum())
    attacker_weight = float(max(honest_train_count / max(attacker_train_count, 1), 1.0))

    rf_model = RandomForestClassifier(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: attacker_weight},
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model = XGBClassifier(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        scale_pos_weight=attacker_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    reports = {
        "random_forest": _evaluate_model("random_forest", rf_model, X_test, y_test),
        "xgboost": _evaluate_model("xgboost", xgb_model, X_test, y_test),
    }

    comparison_rows = []
    for key, report in reports.items():
        row = {"model": key, "threshold": report.threshold}
        row.update(report.metrics)
        row["false_positive_count"] = report.false_positive_count
        row["false_negative_count"] = report.false_negative_count
        comparison_rows.append(row)
    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["false_positive_rate", "false_positive_count", "roc_auc", "recall"],
        ascending=[True, True, False, False],
    )

    selected_model_name = str(model_comparison.iloc[0]["model"])
    selected_model = rf_model if selected_model_name == "random_forest" else xgb_model
    selected_report = reports[selected_model_name]

    rf_importance = _extract_feature_importance(rf_model, "random_forest_importance")
    xgb_importance = _extract_feature_importance(xgb_model, "xgboost_importance")
    feature_importance = (
        pd.concat([rf_importance, xgb_importance], axis=1)
        .reset_index()
        .rename(columns={"index": "feature"})
        .assign(
            mean_importance=lambda df: (
                df["random_forest_importance"] + df["xgboost_importance"]
            )
            / 2.0
        )
        .sort_values("mean_importance", ascending=False)
    )
    selected_importance = _extract_feature_importance(selected_model, "selected_model_importance")
    base_feature_importance = {}
    for base_feature in BASE_FEATURE_COLUMNS:
        values = []
        for suffix in ("mean", "std", "last"):
            feature_name = f"{base_feature}_{suffix}"
            if feature_name in selected_importance.index:
                values.append(float(selected_importance[feature_name]))
        base_feature_importance[base_feature] = float(np.mean(values) if values else 0.0)

    model_path = Path(model_path)
    with model_path.open("wb") as fp:
        pickle.dump(
            {
                "model": selected_model,
                "model_name": selected_model_name,
                "rf_model": rf_model,
                "xgb_model": xgb_model,
                "threshold": selected_report.threshold,
                "feature_columns": FEATURE_COLUMNS,
                "feature_importance": selected_importance.to_dict(),
                "base_feature_importance": base_feature_importance,
                "uncertainty_mode": "ensemble_disagreement",
            },
            fp,
        )

    return ModelArtifacts(
        model=selected_model,
        selected_model_name=selected_model_name,
        threshold=selected_report.threshold,
        metrics=selected_report.metrics,
        confusion_matrix=selected_report.confusion_matrix,
        feature_importance=feature_importance,
        model_comparison=model_comparison,
        model_path=model_path,
        reports=reports,
    )
