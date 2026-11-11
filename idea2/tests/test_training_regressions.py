import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("row_wise_cnn_trainer", ROOT / "scripts" / "train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainingRegressionTests(unittest.TestCase):
    def _argv(self, directory, run_name):
        return ["train.py", "--n", "4", "--m", "4", "--fixed-h", "1",
                "--num-train", "4", "--num-val", "2", "--num-test", "2", "--epochs", "2",
                "--batch-size", "2", "--hidden-channels", "4", "--depth", "1",
                "--num-workers", "0", "--device", "cpu", "--no-progress", "--seed", "7",
                "--output-dir", str(directory), "--run-name", run_name]

    def test_seed_reproduces_weights_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints, summaries = [], []
            for index in range(2):
                torch.manual_seed(100 + index)
                name = f"repeat_{index}"
                with patch.object(sys, "argv", self._argv(directory, name)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    trainer.main()
                run_dir = Path(directory) / name
                checkpoints.append(torch.load(run_dir / "best.pt", weights_only=True))
                summaries.append(json.loads((run_dir / "summary.json").read_text()))
            self.assertEqual(summaries[0], summaries[1])
            for name, value in checkpoints[0]["model_state"].items():
                self.assertTrue(torch.equal(value, checkpoints[1]["model_state"][name]), name)

    def test_cpu_test_evaluation_uses_best_epoch_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        epochs = iter((1, 2))
        evaluated_weights = []

        def run_epoch(args, model, loader, device, optimizer=None):
            if optimizer is not None:
                with torch.no_grad():
                    model.weight.fill_(next(epochs))
            else:
                evaluated_weights.append(model.weight.item())
            score = 1.0 if model.weight.item() == 1.0 else 0.0
            return 0.0, {"support_f1": score, "exact_match": score}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", self._argv(directory, "regression")), \
                    patch.object(trainer, "CNN1DRowPoolModel", return_value=model), \
                    patch.object(trainer, "run_epoch", side_effect=run_epoch), \
                    contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            checkpoint = torch.load(Path(directory) / "regression" / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["model_state"]["weight"].item(), 1.0)
            self.assertEqual(evaluated_weights, [1.0, 2.0, 1.0])

    def test_variable_h_all_zero_does_not_report_fixed_h_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = self._argv(directory, "baseline") + ["--input-mode", "all_zero", "--h-setting", "variable_h"]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "requires.*fixed_h"):
                trainer.main()
            self.assertFalse((Path(directory) / "baseline" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
