import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAS_PLOTS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "matplotlib", "sklearn"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_PLOTS, "Install idea2[plots] to test visualization exports")
class PlotExportTests(unittest.TestCase):
    def test_small_embedding_sets_have_valid_tsne_perplexity(self):
        import numpy as np

        features = np.random.default_rng(0).normal(size=(4, 5))
        scripts = ["plot_relation_pair_tsne", "plot_relation_pair_tsne_by_secret",
                   "plot_relation_pair_tsne_binary_secret", "plot_tsne_embeddings",
                   "plot_relation_pair_tsne_binary_secret_3d_html"]
        for name in scripts:
            with self.subTest(script=name):
                module = load_script(name)
                dimensions = 3 if name.endswith("3d_html") else 2
                points = getattr(module, f"tsne_{dimensions}d")(features, seed=0, perplexity=30.0, max_iter=250)
                self.assertEqual(points.shape, (4, dimensions))
                self.assertTrue(np.isfinite(points).all())

    def test_support_labels_round_trip_as_one_csv_field(self):
        import numpy as np

        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = ["0,1", "2,3"]
        for name, field in (("plot_relation_pair_tsne", "label"),
                            ("plot_relation_pair_tsne_by_secret", "secret_support")):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as directory:
                module = load_script(name)
                path = Path(directory) / "coordinates.csv"
                module.save_coordinates(path, points, points, labels)
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row[field] for row in rows], labels)
                self.assertTrue(all(None not in row for row in rows))
                self.assertEqual([float(row["after_y"]) for row in rows], [2.0, 4.0])

    def test_phase_coordinate_embeddings_preserve_all_input_channels(self):
        from lwe_idea2 import CNN1DRowPoolConfig, CNN1DRowPoolModel

        module = load_script("plot_relation_pair_tsne_binary_secret")
        config = dict(n=4, m=4, q=17, noise_width=1.0, noise_dist="discrete_gaussian",
                      h_setting="fixed_h", fixed_h=1, h_min=1, h_max=3,
                      residue_encoding="phase", fourier_k=2, hidden_channels=4, depth=1, dropout=0.0)
        model = CNN1DRowPoolModel(CNN1DRowPoolConfig(
            n=4, m=4, q=17, residue_encoding="phase", fourier_k=2,
            hidden_channels=4, depth=1, dropout=0.0))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text(json.dumps(config))
            torch.save({"model_state": model.state_dict()}, run_dir / "best.pt")
            args = SimpleNamespace(run_dir=run_dir, num_samples=2, batch_size=2, seed=123)
            before, after, labels, coords, shapes = module.collect_coordinate_embeddings(args, torch.device("cpu"))
        self.assertEqual(before.shape, (8, 4 * 10))
        self.assertEqual(after.shape, (8, 4 * 4))
        self.assertEqual(labels.shape, coords.shape)
        self.assertEqual(shapes["cnn_input_row_tensor"], (8, 10, 4))


if __name__ == "__main__":
    unittest.main()
