from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from constants import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from geo import haversine_km

DEFAULT_FRAUD_SCORE_THRESHOLD = 0.5
# API scoring uses a simpler online trust heuristic than offline feature extraction
# because request payloads may include only partial neighbor evidence.
API_TRUST_RECIPROCITY_WEIGHT = 0.5
API_TRUST_STABILITY_WEIGHT = 0.5
API_TRUST_STABILITY_SCALE = 150.0
RISK_TREND_THRESHOLD = 0.03
FULL_EVIDENCE_LATENCY_COUNT = 8.0
BASE_CONFIDENCE = 0.45
SEPARATION_CONFIDENCE_WEIGHT = 0.35
EVIDENCE_CONFIDENCE_WEIGHT = 0.20
MIN_THRESHOLD_EPSILON = 1e-6
UNCERTAINTY_THRESHOLD_ADJUSTMENT = 0.10
UNCERTAINTY_THRESHOLD_MIN = 0.20
UNCERTAINTY_THRESHOLD_MAX = 0.95
UNCERTAINTY_CONFIDENCE_PENALTY = 0.35


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class ScoreNodeRequest(BaseModel):
    latencies: List[float]
    peers: List[int]
    claimed_location: Location
    inferred_location: Optional[Location] = None
    reverse_latencies: Optional[List[float]] = None
    peer_claimed_locations: Optional[List[Location]] = None
    clustering_coefficient: float = Field(0.0, ge=0.0, le=1.0)
    reciprocity_score: float = Field(0.0, ge=0.0, le=1.0)
    past_fraud_scores: Optional[List[float]] = None
    history_latencies: Optional[List[List[float]]] = None


class ScoreNodeResponse(BaseModel):
    fraud_score: float
    uncertainty_score: float
    risk_label: str
    trend: str
    confidence_score: float
    reasons: List[str]


@lru_cache(maxsize=1)
def _load_model_bundle():
    model_path = Path(os.getenv("MODEL_PATH", "model.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with model_path.open("rb") as fp:
        return pickle.load(fp)


def _build_feature_vector(payload: ScoreNodeRequest) -> List[float]:
    latencies = np.array(payload.latencies, dtype=float)
    if latencies.size == 0:
        raise ValueError("latencies must contain at least one value")
    latencies = latencies[np.isfinite(latencies)]
    if latencies.size == 0:
        raise ValueError("latencies must contain at least one finite value")

    inferred = payload.inferred_location or payload.claimed_location
    mismatch = haversine_km(
        (payload.claimed_location.lat, payload.claimed_location.lon),
        (inferred.lat, inferred.lon),
    )
    skewness = 0.0
    kurtosis = 0.0
    if latencies.size >= 3:
        centered = latencies - float(np.mean(latencies))
        std = float(np.std(latencies))
        if std > 1e-9:
            normalized = centered / std
            skewness = float(np.mean(normalized**3))
            if latencies.size >= 4:
                kurtosis = float(np.mean(normalized**4) - 3.0)

    peer_spread = 0.0
    if payload.peer_claimed_locations:
        coords = np.array([[p.lat, p.lon] for p in payload.peer_claimed_locations], dtype=float)
        if coords.shape[0] >= 2:
            center = coords.mean(axis=0)
            peer_spread = float(
                np.median(
                    [
                        haversine_km((lat, lon), (float(center[0]), float(center[1])))
                        for lat, lon in coords
                    ]
                )
            )

    inconsistency = float(np.median(np.abs(latencies - np.median(latencies)) / np.maximum(np.median(latencies), 1.0)))

    edge_asymmetry = 0.0
    if payload.reverse_latencies:
        reverse = np.array(payload.reverse_latencies, dtype=float)
        usable = min(latencies.size, reverse.size)
        if usable > 0:
            reverse = reverse[:usable]
            forward = latencies[:usable]
            valid = np.isfinite(forward) & np.isfinite(reverse)
            if np.any(valid):
                f = forward[valid]
                r = reverse[valid]
                edge_asymmetry = float(np.mean(np.abs(f - r) / np.maximum((f + r) / 2.0, 1.0)))

    base_map = {
        "rtt_variance": float(np.var(latencies)),
        "avg_latency_to_peers": float(np.mean(latencies)),
        "claimed_inferred_distance_mismatch": float(mismatch),
        "unique_peers": float(len(set(payload.peers))),
        "clustering_coefficient": float(payload.clustering_coefficient),
        "reciprocity_score": float(payload.reciprocity_score),
        "latency_skewness": skewness,
        "latency_kurtosis": kurtosis,
        "peer_geographic_diversity": peer_spread,
        "latency_inconsistency_score": inconsistency,
        "neighbor_trust_score": float(
            API_TRUST_RECIPROCITY_WEIGHT * payload.reciprocity_score
            + API_TRUST_STABILITY_WEIGHT * (1.0 - min(np.std(latencies) / API_TRUST_STABILITY_SCALE, 1.0))
        ),
        "edge_asymmetry": edge_asymmetry,
    }

    history_means: List[float] = []
    for entry in payload.history_latencies or []:
        hist = np.array(entry, dtype=float)
        hist = hist[np.isfinite(hist)]
        if hist.size > 0:
            history_means.append(float(np.mean(hist)))
    latency_series = np.array(history_means + [base_map["avg_latency_to_peers"]], dtype=float)
    if latency_series.size >= 2:
        latency_diff = np.diff(latency_series)
        max_spike = float(np.max(np.abs(latency_diff)))
        sudden_change = float(np.mean(np.abs(latency_diff)))
        burst_threshold = float(np.mean(np.abs(latency_diff)) + np.std(np.abs(latency_diff)))
        burst_activity = float(np.mean(np.abs(latency_diff) > burst_threshold))
    else:
        max_spike = 0.0
        sudden_change = 0.0
        burst_activity = 0.0
    if latency_series.size >= 3:
        slope = float(np.polyfit(np.arange(latency_series.size, dtype=float), latency_series, deg=1)[0])
    else:
        slope = 0.0

    behavior_volatility = float(np.std(latency_series)) if latency_series.size > 0 else 0.0
    base_map["max_latency_spike"] = max_spike
    base_map["latency_trend_slope"] = slope
    base_map["behavior_volatility"] = behavior_volatility
    base_map["sudden_change_score"] = sudden_change
    base_map["burst_activity_score"] = burst_activity

    feature_values = []
    for feature in BASE_FEATURE_COLUMNS:
        value = float(base_map.get(feature, 0.0))
        # API requests are single-point observations (not a temporal window),
        # so std is set to 0 while mean/last use the same observed value.
        feature_values.extend([value, 0.0, value])  # mean, std, last
    return feature_values


def _feature_risk_signal(feature: str, value: float) -> float:
    if feature == "claimed_inferred_distance_mismatch":
        return float(min(value / 1500.0, 1.0))
    if feature == "clustering_coefficient":
        return float(min(value / 0.7, 1.0))
    if feature == "neighbor_trust_score":
        return float(min(max((0.55 - value) / 0.55, 0.0), 1.0))
    if feature == "latency_inconsistency_score":
        return float(min(value / 0.8, 1.0))
    if feature == "edge_asymmetry":
        return float(min(value / 0.7, 1.0))
    if feature == "rtt_variance":
        return float(min(max((20.0 - value) / 20.0, 0.0), 1.0))
    if feature == "reciprocity_score":
        return float(min(max((0.45 - value) / 0.45, 0.0), 1.0))
    if feature == "unique_peers":
        return float(min(max((5.0 - value) / 5.0, 0.0), 1.0))
    return float(min(max(value, 0.0), 1.0))


def _reason_text(feature: str) -> str:
    return {
        "claimed_inferred_distance_mismatch": "high claimed vs inferred location mismatch",
        "clustering_coefficient": "high clustering around suspicious peers",
        "neighbor_trust_score": "low neighbor trust score",
        "latency_inconsistency_score": "high latency inconsistency with claimed geography",
        "edge_asymmetry": "high edge latency asymmetry",
        "rtt_variance": "low latency variance pattern",
        "reciprocity_score": "low reciprocity with peers",
        "unique_peers": "unusually low peer diversity",
    }.get(feature, f"elevated risk in {feature.replace('_', ' ')}")


def _top_reasons(
    base_map: Dict[str, float],
    base_feature_importance: Dict[str, float],
    top_k: int = 3,
) -> List[str]:
    scored = []
    for feature, value in base_map.items():
        importance = float(base_feature_importance.get(feature, 0.0))
        scored.append((importance * _feature_risk_signal(feature, float(value)), feature))
    top = [feature for score, feature in sorted(scored, reverse=True)[:top_k] if score > 0.0]
    if not top:
        return ["insufficient anomalous evidence in current observation"]
    return [_reason_text(feature) for feature in top]


app = FastAPI(title="Node Fraud Scoring API", version="1.0.0")


@app.post("/score-node", response_model=ScoreNodeResponse)
def score_node(payload: ScoreNodeRequest) -> ScoreNodeResponse:
    try:
        model_bundle = _load_model_bundle()
        model = model_bundle["model"]
        feature_columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)
        vector = _build_feature_vector(payload)
        if len(vector) != len(feature_columns):
            raise ValueError("Model feature schema mismatch")
        features = pd.DataFrame([vector], columns=feature_columns)
        rf_model = model_bundle.get("rf_model")
        xgb_model = model_bundle.get("xgb_model")
        if rf_model is not None and xgb_model is not None:
            rf_score = float(rf_model.predict_proba(features)[0][1])
            xgb_score = float(xgb_model.predict_proba(features)[0][1])
            fraud_score = float((rf_score + xgb_score) / 2.0)
            uncertainty_score = float(abs(rf_score - xgb_score))
        else:
            fraud_score = float(model.predict_proba(features)[0][1])
            uncertainty_score = 0.0
        base_threshold = float(
            os.getenv("FRAUD_SCORE_THRESHOLD", model_bundle.get("threshold", DEFAULT_FRAUD_SCORE_THRESHOLD))
        )
        threshold = float(
            np.clip(
                base_threshold + UNCERTAINTY_THRESHOLD_ADJUSTMENT * uncertainty_score,
                UNCERTAINTY_THRESHOLD_MIN,
                UNCERTAINTY_THRESHOLD_MAX,
            )
        )
        base_map = {}
        for i, feature in enumerate(BASE_FEATURE_COLUMNS):
            base_map[feature] = float(vector[i * 3])
        base_feature_importance = model_bundle.get("base_feature_importance", {})
        reasons = _top_reasons(base_map=base_map, base_feature_importance=base_feature_importance, top_k=3)
        risk_label = "high_risk" if fraud_score >= threshold else "low_risk"
        score_history = [float(v) for v in (payload.past_fraud_scores or []) if np.isfinite(v)]
        score_history.append(fraud_score)
        if len(score_history) >= 2:
            delta = score_history[-1] - score_history[-2]
            if delta > RISK_TREND_THRESHOLD:
                trend = "increasing_risk"
            elif delta < -RISK_TREND_THRESHOLD:
                trend = "decreasing_risk"
            else:
                trend = "stable_risk"
        else:
            trend = "stable_risk"
        separation = min(abs(fraud_score - threshold) / max(abs(threshold), MIN_THRESHOLD_EPSILON), 1.0)
        evidence_strength = min(len(payload.latencies) / FULL_EVIDENCE_LATENCY_COUNT, 1.0)
        confidence_score = float(
            np.clip(
                BASE_CONFIDENCE
                + SEPARATION_CONFIDENCE_WEIGHT * separation
                + EVIDENCE_CONFIDENCE_WEIGHT * evidence_strength,
                0.0,
                1.0,
            )
        )
        confidence_score = float(
            np.clip(confidence_score * (1.0 - UNCERTAINTY_CONFIDENCE_PENALTY * uncertainty_score), 0.0, 1.0)
        )
        return ScoreNodeResponse(
            fraud_score=fraud_score,
            uncertainty_score=uncertainty_score,
            risk_label=risk_label,
            trend=trend,
            confidence_score=confidence_score,
            reasons=reasons,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
