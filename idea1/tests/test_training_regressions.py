import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))
spec = importlib.util.spec_from_file_location("row_wise_vit_trainer", ROOT / "scripts" / "train_lwe_vit.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainingRegressionTests(unittest.TestCase):
    def test_cpu_test_evaluation_uses_best_epoch_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        test_weights = []

        def run_epoch(model, loader, optimizer, class_weights, args, device, *, epoch=None, phase="train"):
            if phase == "train":
                with torch.no_grad():
                    model.weight.fill_(epoch)
            if phase == "test":
                test_weights.append(model.weight.item())
            score = 1.0 if model.weight.item() == 1.0 else 0.0
            return 0.0, 0.0, 0.0, {"exact_match": score, "support_f1": score}

        with tempfile.TemporaryDirectory() as directory:
            argv = ["train_lwe_vit.py", "--model", "row_cnn", "--n", "4", "--m", "4",
                    "--h-setting", "fixed_h", "--fixed-h", "1", "--num-train", "2",
                    "--num-val", "2", "--num-test", "2", "--epochs", "2", "--batch-size", "2",
                    "--output-dir", directory, "--run-name", "regression", "--save-best", "--no-resume"]
            with patch.object(sys, "argv", argv):
                args = trainer.normalize_args(trainer.parse_args())
            with patch.object(trainer, "RowLocalCNNLWEModel", return_value=model), \
                    patch.object(trainer, "run_epoch", side_effect=run_epoch), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = trainer.train_single_seed(args, 0, torch.device("cpu"))
            checkpoint = torch.load(args.run_dir / "seed_0" / "latest.pt", weights_only=False)
            self.assertEqual(result["summary"]["best_epoch"], 1)
            self.assertEqual(test_weights, [1.0])
            self.assertEqual(checkpoint["best_state"]["weight"].item(), 1.0)
            self.assertEqual(checkpoint["model_state"]["weight"].item(), 2.0)

    def test_legacy_cnn_cpu_evaluation_uses_best_epoch_weights(self):
        legacy = importlib.import_module("idea1.CNN+TRANS.train")
        checks = importlib.import_module("idea1.CNN+TRANS.checks")
        model = torch.nn.Linear(1, 1, bias=False)
        epochs = iter((1, 2))
        evaluated_weights = []

        def train_step_epoch(**kwargs):
            with torch.no_grad():
                kwargs["model"].weight.fill_(next(epochs))
            return 0.0

        def run_epoch(**kwargs):
            value = kwargs["model"].weight.item()
            evaluated_weights.append(value)
            score = 1.0 if value == 1.0 else 0.0
            return 0.0, {"exact_match": score, "support_f1": score}

        with tempfile.TemporaryDirectory() as directory:
            args = checks.make_args(n=4, m=4, num_train=2, num_val=2, num_test=2,
                                    h_setting="fixed_h", fixed_h=1, epochs=2, batch_size=2,
                                    output_dir=Path(directory), save_best=True)
            with patch.object(legacy, "build_model", return_value=model), \
                    patch.object(legacy, "train_step_epoch", side_effect=train_step_epoch), \
                    patch.object(legacy, "run_epoch", side_effect=run_epoch), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = legacy.train_single_seed(args, 0, torch.device("cpu"))
            self.assertEqual(result["summary"]["best_epoch"], 1)
            self.assertEqual(evaluated_weights, [1.0, 1.0, 2.0, 2.0, 1.0])
            state = torch.load(args.run_dir / "seed_0" / "best.pt", weights_only=True)
            self.assertEqual(state["weight"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
