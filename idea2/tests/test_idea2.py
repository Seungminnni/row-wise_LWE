from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_idea2 import CNN1DRowPoolConfig, CNN1DRowPoolModel, LWEAblationDataset, LWEDataConfig, LWEParams
from lwe_idea2.metrics import all_zero_baseline_metrics, binary_support_metrics


class Idea2Tests(unittest.TestCase):
    def _batch(self, input_mode: str):
        params = LWEParams(n=10, m=16, q=257)
        dataset = LWEAblationDataset(
            LWEDataConfig(num_samples=4, params=params, input_mode=input_mode, fixed_h=2)
        )
        items = [dataset[i] for i in range(2)]
        return {key: torch.stack([item[key] for item in items]) for key in items[0]}

    def test_relation_pair_forward(self) -> None:
        batch = self._batch("relation_pair")
        model = CNN1DRowPoolModel(CNN1DRowPoolConfig(n=10, m=16, q=257, input_mode="relation_pair", hidden_channels=16, depth=2))
        logits = model(batch["A"], batch["b_input"])
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_b_shuffle_changes_b_input(self) -> None:
        batch = self._batch("b_shuffle")
        self.assertFalse(torch.equal(batch["b"], batch["b_input"]))

    def test_b_once_forward(self) -> None:
        batch = self._batch("b_once")
        model = CNN1DRowPoolModel(CNN1DRowPoolConfig(n=10, m=16, q=257, input_mode="b_once", hidden_channels=16, depth=2))
        logits = model(batch["A"], batch["b_input"])
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_top_h_metrics_predicts_two_active(self) -> None:
        batch = self._batch("relation_pair")
        logits = torch.zeros(2, 10)
        logits[:, :2] = 10.0
        metrics = binary_support_metrics(logits, batch["target"], batch["A"], batch["b"], q=257, decode="top_h", h=2)
        self.assertIn("support_f1", metrics)
        self.assertEqual(metrics["coord_acc"] <= 1.0, True)

    def test_all_zero_baseline(self) -> None:
        metrics = all_zero_baseline_metrics(n=10, h=2)
        self.assertEqual(metrics["coord_acc"], 0.8)
        self.assertEqual(metrics["support_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()

