from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True)
class AttackerDecision:
    action: str
    predicted_score: float
    utility: float
    stealth_cost: float


@dataclass(frozen=True)
class DefenderStepMetrics:
    false_positive_rate: float
    false_negative_rate: float
    attacker_recall: float
    threshold: float


class AttackerPolicy(ABC):
    @abstractmethod
    def decide_action(
        self,
        feature_row: pd.Series,
        feature_columns: Sequence[str],
        model_bundle: object,
        current_score: float,
        threshold: float,
        recent_profit: float,
    ) -> AttackerDecision:
        raise NotImplementedError


class DefenderPolicy(ABC):
    @abstractmethod
    def effective_threshold(self, node_id: int, region: str, uncertainty: float) -> float:
        raise NotImplementedError


def predict_fraud_and_uncertainty(model_bundle: object, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(model_bundle, dict) and "rf_model" in model_bundle and "xgb_model" in model_bundle:
        rf_model = model_bundle["rf_model"]
        xgb_model = model_bundle["xgb_model"]
        rf_prob = rf_model.predict_proba(X)[:, 1]
        xgb_prob = xgb_model.predict_proba(X)[:, 1]
        fraud = 0.5 * (rf_prob + xgb_prob)
        uncertainty = np.abs(rf_prob - xgb_prob)
        return fraud.astype(float), uncertainty.astype(float)

    model = model_bundle["model"] if isinstance(model_bundle, dict) and "model" in model_bundle else model_bundle
    fraud = model.predict_proba(X)[:, 1]
    uncertainty = np.zeros_like(fraud, dtype=float)
    return fraud.astype(float), uncertainty.astype(float)


class StrategicAttackerPolicy(AttackerPolicy):
    ACTION_COSTS = {
        "noop": 0.0,
        "reduce_clustering": 0.28,
        "add_noise": 0.24,
        "increase_connections": 0.26,
    }
    HIGH_PROFIT_UTILITY_BONUS = 0.05
    HIGH_SCORE_UTILITY_BONUS = 0.03
    HIGH_PROFIT_THRESHOLD = 2.2

    def _apply_action(self, row: pd.Series, action: str) -> pd.Series:
        out = row.copy()

        def scale(base: str, factor: float) -> None:
            for suffix in ("_mean", "_last"):
                col = f"{base}{suffix}"
                if col in out:
                    out[col] = float(out[col]) * factor

        def shift(base: str, delta: float) -> None:
            for suffix in ("_mean", "_last"):
                col = f"{base}{suffix}"
                if col in out:
                    out[col] = float(max(float(out[col]) + delta, 0.0))

        if action == "reduce_clustering":
            scale("clustering_coefficient_", 0.72)
            shift("reciprocity_score_", 0.06)
        elif action == "add_noise":
            shift("rtt_variance_", 0.15)
            scale("latency_inconsistency_score_", 0.88)
            scale("edge_asymmetry_", 0.90)
        elif action == "increase_connections":
            shift("unique_peers_", 1.8)
            shift("peer_geographic_diversity_", 0.9)
            scale("claimed_inferred_distance_mismatch_", 0.82)
            shift("neighbor_trust_score_", 0.08)

        return out

    def decide_action(
        self,
        feature_row: pd.Series,
        feature_columns: Sequence[str],
        model_bundle: object,
        current_score: float,
        threshold: float,
        recent_profit: float,
    ) -> AttackerDecision:
        best = AttackerDecision(action="noop", predicted_score=float(current_score), utility=0.0, stealth_cost=0.0)
        base_X = pd.DataFrame([feature_row[list(feature_columns)]], columns=list(feature_columns))
        baseline_pred, _ = predict_fraud_and_uncertainty(model_bundle, base_X)
        baseline_score = float(baseline_pred[0])

        for action, cost in self.ACTION_COSTS.items():
            if action == "noop":
                candidate_score = baseline_score
                utility = 0.0
            else:
                row_after = self._apply_action(feature_row, action)
                X_after = pd.DataFrame([row_after[list(feature_columns)]], columns=list(feature_columns))
                pred_after, _ = predict_fraud_and_uncertainty(model_bundle, X_after)
                candidate_score = float(pred_after[0])
                fraud_reduction = max(baseline_score - candidate_score, 0.0)
                utility = float(fraud_reduction / max(cost, 1e-9))
                if recent_profit > self.HIGH_PROFIT_THRESHOLD and action == "increase_connections":
                    utility += self.HIGH_PROFIT_UTILITY_BONUS
                if baseline_score >= threshold and action != "noop":
                    utility += self.HIGH_SCORE_UTILITY_BONUS

            decision = AttackerDecision(action=action, predicted_score=candidate_score, utility=utility, stealth_cost=float(cost))
            if decision.utility > best.utility:
                best = decision

        return best


class AdaptiveDefenderPolicy(DefenderPolicy):
    QUARANTINE_UNCERTAINTY_MAX = 0.7
    RF_PARAMS = {
        "n_estimators": 220,
        "max_depth": 11,
        "min_samples_leaf": 2,
    }
    XGB_PARAMS = {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
    }
    UNCERTAINTY_THRESHOLD_ADJUSTMENT = 0.08

    def __init__(
        self,
        base_threshold: float,
        retrain_interval: int,
        sliding_window_steps: int,
        dynamic_threshold: bool = True,
        quarantine_duration: int = 2,
    ) -> None:
        self.base_threshold = float(base_threshold)
        self.current_threshold = float(base_threshold)
        self.retrain_interval = max(int(retrain_interval), 1)
        self.sliding_window_steps = max(int(sliding_window_steps), 2)
        self.dynamic_threshold = dynamic_threshold
        self.quarantine_duration = max(int(quarantine_duration), 1)

        self.quarantined_until: Dict[int, int] = {}
        self.trust_scores: Dict[int, float] = {}
        self.reputation_scores: Dict[int, float] = {}
        self.region_tightening: Dict[str, float] = {}

        self.training_history: List[pd.DataFrame] = []
        self.degradation_over_time: List[float] = []
        self.performance_vs_adaptive_attackers: List[float] = []
        self.last_auc: float | None = None

    def register_nodes(self, node_ids: Iterable[int]) -> None:
        for node_id in node_ids:
            self.trust_scores.setdefault(int(node_id), 1.0)
            self.reputation_scores.setdefault(int(node_id), 1.0)

    def is_quarantined(self, node_id: int, timestep: int) -> bool:
        return int(self.quarantined_until.get(int(node_id), -1)) >= int(timestep)

    def effective_threshold(self, node_id: int, region: str, uncertainty: float) -> float:
        trust = float(self.trust_scores.get(int(node_id), 1.0))
        rep = float(self.reputation_scores.get(int(node_id), 1.0))
        tighten = float(self.region_tightening.get(region, 0.0))
        threshold = self.current_threshold - tighten
        threshold -= 0.10 * (1.0 - trust)
        threshold -= 0.12 * (1.0 - rep)
        threshold += self.UNCERTAINTY_THRESHOLD_ADJUSTMENT * float(np.clip(uncertainty, 0.0, 1.0))
        return float(np.clip(threshold, 0.20, 0.95))

    def assess_node(
        self,
        node_id: int,
        label: str,
        region: str,
        fraud_score: float,
        uncertainty: float,
        timestep: int,
    ) -> bool:
        eff_threshold = self.effective_threshold(node_id=node_id, region=region, uncertainty=uncertainty)
        flagged = float(fraud_score) >= eff_threshold

        trust = float(self.trust_scores.get(node_id, 1.0))
        rep = float(self.reputation_scores.get(node_id, 1.0))
        if flagged:
            trust = max(0.0, trust - 0.08)
            rep = max(0.0, rep - 0.06)
            self.region_tightening[region] = min(0.20, float(self.region_tightening.get(region, 0.0)) + 0.01)
            if uncertainty < self.QUARANTINE_UNCERTAINTY_MAX:
                self.quarantined_until[node_id] = int(timestep + self.quarantine_duration)
        else:
            trust = min(1.0, trust + 0.01)
            rep = min(1.0, rep + 0.005)
            self.region_tightening[region] = max(0.0, float(self.region_tightening.get(region, 0.0)) - 0.002)

        if label == "honest":
            rep = min(1.0, rep + 0.002)

        self.trust_scores[node_id] = trust
        self.reputation_scores[node_id] = rep
        return flagged

    def maybe_retrain(
        self,
        timestep: int,
        feature_columns: Sequence[str],
        random_state: int,
    ) -> object | None:
        if timestep <= 0 or timestep % self.retrain_interval != 0:
            return None
        if len(self.training_history) < self.sliding_window_steps:
            return None

        window = pd.concat(self.training_history[-self.sliding_window_steps :], ignore_index=True)
        if window.empty or int(window["label"].nunique()) < 2:
            return None

        X = window[list(feature_columns)]
        y = window["label"].astype(int)

        honest_count = int((y == 0).sum())
        attacker_count = int((y == 1).sum())
        attacker_weight = float(max(honest_count / max(attacker_count, 1), 1.0))

        rf_model = RandomForestClassifier(
            n_estimators=self.RF_PARAMS["n_estimators"],
            max_depth=self.RF_PARAMS["max_depth"],
            min_samples_leaf=self.RF_PARAMS["min_samples_leaf"],
            class_weight={0: 1.0, 1: attacker_weight},
            random_state=random_state + timestep,
            n_jobs=-1,
        )
        xgb_model = XGBClassifier(
            n_estimators=self.XGB_PARAMS["n_estimators"],
            max_depth=self.XGB_PARAMS["max_depth"],
            learning_rate=self.XGB_PARAMS["learning_rate"],
            subsample=self.XGB_PARAMS["subsample"],
            colsample_bytree=self.XGB_PARAMS["colsample_bytree"],
            scale_pos_weight=attacker_weight,
            eval_metric="logloss",
            random_state=random_state + timestep,
            n_jobs=-1,
        )

        rf_model.fit(X, y)
        xgb_model.fit(X, y)
        ensemble_probs = 0.5 * (rf_model.predict_proba(X)[:, 1] + xgb_model.predict_proba(X)[:, 1])
        auc = float(roc_auc_score(y, ensemble_probs)) if int(y.nunique()) > 1 else 0.5
        if self.last_auc is not None:
            self.degradation_over_time.append(float(max(self.last_auc - auc, 0.0)))
        self.last_auc = auc

        return {
            "rf_model": rf_model,
            "xgb_model": xgb_model,
            "model": rf_model,
            "threshold": self.current_threshold,
            "feature_columns": list(feature_columns),
        }

    def update_threshold(self, fp_rate: float, fn_rate: float) -> None:
        if not self.dynamic_threshold:
            return
        self.current_threshold += 0.03 * float(fp_rate - fn_rate)
        self.current_threshold = float(np.clip(self.current_threshold, 0.2, 0.95))

    def timestep_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> DefenderStepMetrics:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp_rate = float(fp / max(fp + tn, 1))
        fn_rate = float(fn / max(fn + tp, 1))
        attacker_recall = float(tp / max(tp + fn, 1))
        self.performance_vs_adaptive_attackers.append(attacker_recall)
        self.update_threshold(fp_rate=fp_rate, fn_rate=fn_rate)
        return DefenderStepMetrics(
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            attacker_recall=attacker_recall,
            threshold=float(threshold),
        )


def summarize_defender_metrics(policy: AdaptiveDefenderPolicy, current_timestep: int) -> Dict[str, float]:
    degradation = float(np.mean(policy.degradation_over_time)) if policy.degradation_over_time else 0.0
    perf = float(np.mean(policy.performance_vs_adaptive_attackers)) if policy.performance_vs_adaptive_attackers else 0.0
    avg_trust = float(np.mean(list(policy.trust_scores.values()))) if policy.trust_scores else 1.0
    avg_rep = float(np.mean(list(policy.reputation_scores.values()))) if policy.reputation_scores else 1.0
    quarantined = float(sum(1 for _, until in policy.quarantined_until.items() if until >= current_timestep))
    return {
        "model_degradation_over_time": degradation,
        "performance_vs_adaptive_attackers": perf,
        "avg_trust_score": avg_trust,
        "avg_reputation_score": avg_rep,
        "quarantined_node_count": quarantined,
        "dynamic_threshold": float(policy.current_threshold),
    }
