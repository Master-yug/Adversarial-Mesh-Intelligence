import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import ScoreNodeRequest, _build_feature_vector, app
from constants import FEATURE_COLUMNS
from features import extract_node_features, extract_temporal_node_features
from modeling import train_fraud_model
from simulation import build_network_simulation


class AntiSpoofPipelineTests(unittest.TestCase):
    def test_simulation_outputs_expected_shapes(self):
        sim = build_network_simulation(
            total_nodes=200,
            honest_ratio=0.65,
            naive_attacker_ratio=0.2,
            smart_attacker_ratio=0.15,
            time_steps=12,
            seed=1,
        )
        self.assertEqual(len(sim.nodes), 200)
        self.assertEqual(sim.latency_matrix.shape, (200, 200))
        self.assertEqual(len(sim.time_steps), 12)
        labels = [node.label for node in sim.nodes]
        self.assertEqual(labels.count("honest"), 130)
        self.assertEqual(labels.count("naive_attacker"), 40)
        self.assertEqual(labels.count("smart_attacker"), 30)
        self.assertIn("scenario_event_count", sim.scenario_metrics)

    def test_feature_extraction_schema(self):
        sim = build_network_simulation(total_nodes=50, time_steps=8, seed=2)
        temporal_df = extract_temporal_node_features(sim)
        df = extract_node_features(sim)
        self.assertEqual(len(df), 50)
        self.assertEqual(len(temporal_df), 50 * 8)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(set(df.columns)))
        for temporal_feature in (
            "max_latency_spike",
            "latency_trend_slope",
            "behavior_volatility",
            "sudden_change_score",
            "burst_activity_score",
        ):
            self.assertIn(temporal_feature, temporal_df.columns)
        self.assertIn("label_name", df.columns)
        self.assertIn("label", df.columns)

    def test_model_train_and_api_score(self):
        sim = build_network_simulation(total_nodes=120, time_steps=10, seed=3)
        df = extract_node_features(sim)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.pkl")
            artifacts = train_fraud_model(df, model_path=model_path, random_state=3)

            self.assertTrue(os.path.exists(model_path))
            for metric in ("accuracy", "precision", "recall", "roc_auc"):
                self.assertGreaterEqual(artifacts.metrics[metric], 0.0)
                self.assertLessEqual(artifacts.metrics[metric], 1.0)
            self.assertIn(artifacts.selected_model_name, {"random_forest", "xgboost"})
            self.assertEqual(set(artifacts.feature_importance["feature"]), set(FEATURE_COLUMNS))
            self.assertEqual(tuple(artifacts.confusion_matrix.shape), (2, 2))

            with patch.dict(os.environ, {"MODEL_PATH": model_path}, clear=False):
                client = TestClient(app)
                payload = {
                    "latencies": [10.1, 22.5, 13.2],
                    "history_latencies": [[9.5, 10.2], [15.0, 28.0, 11.0]],
                    "past_fraud_scores": [0.18, 0.24, 0.37],
                    "peers": [1, 2, 3, 3],
                    "claimed_location": {"lat": 40.7128, "lon": -74.0060},
                    "inferred_location": {"lat": 34.0522, "lon": -118.2437},
                    "reverse_latencies": [11.4, 20.7, 12.5],
                    "peer_claimed_locations": [
                        {"lat": 51.5074, "lon": -0.1278},
                        {"lat": 35.6762, "lon": 139.6503},
                        {"lat": 43.6532, "lon": -79.3832},
                    ],
                    "clustering_coefficient": 0.2,
                    "reciprocity_score": 0.4,
                }
                response = client.post("/score-node", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("fraud_score", data)
                self.assertIn("risk_label", data)
                self.assertIn("trend", data)
                self.assertIn("confidence_score", data)
                self.assertIn("reasons", data)
                self.assertTrue(isinstance(data["reasons"], list))
                self.assertGreaterEqual(float(data["confidence_score"]), 0.0)
                self.assertLessEqual(float(data["confidence_score"]), 1.0)
                baseline_score = float(data["fraud_score"])

                increasing_payload = dict(payload)
                increasing_payload["past_fraud_scores"] = [max(0.0, baseline_score - 0.2)]
                increasing_resp = client.post("/score-node", json=increasing_payload)
                self.assertEqual(increasing_resp.status_code, 200)
                self.assertEqual(increasing_resp.json()["trend"], "increasing_risk")

                decreasing_payload = dict(payload)
                decreasing_payload["past_fraud_scores"] = [min(1.0, baseline_score + 0.2)]
                decreasing_resp = client.post("/score-node", json=decreasing_payload)
                self.assertEqual(decreasing_resp.status_code, 200)
                self.assertEqual(decreasing_resp.json()["trend"], "decreasing_risk")

                stable_payload = dict(payload)
                stable_payload["past_fraud_scores"] = [baseline_score]
                stable_resp = client.post("/score-node", json=stable_payload)
                self.assertEqual(stable_resp.status_code, 200)
                self.assertEqual(stable_resp.json()["trend"], "stable_risk")

                feature_vector = _build_feature_vector(ScoreNodeRequest(**payload))
                self.assertEqual(len(feature_vector), len(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
