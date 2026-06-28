from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_idea2 import CNN1DRowPoolConfig, CNN1DRowPoolModel, LWEAblationDataset, LWEDataConfig, LWEParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exact relation_pair t-SNE before and after CNN.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "n16_h2_noise3_m64_relation_pair_cnn1d_10k_1k_1k_ep200_seed0",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "result")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def make_dataset(config: dict, num_samples: int, seed: int) -> LWEAblationDataset:
    params = LWEParams(
        n=int(config["n"]),
        m=int(config["m"]),
        q=int(config["q"]),
        noise_width=float(config["noise_width"]),
        noise_dist=str(config["noise_dist"]),
    )
    return LWEAblationDataset(
        LWEDataConfig(
            num_samples=num_samples,
            params=params,
            seed=seed,
            input_mode="relation_pair",
            h_setting=str(config["h_setting"]),
            fixed_h=int(config["fixed_h"]),
            h_min=int(config["h_min"]),
            h_max=int(config["h_max"]),
        )
    )


def make_model(config: dict, run_dir: Path, device: torch.device) -> CNN1DRowPoolModel:
    model = CNN1DRowPoolModel(
        CNN1DRowPoolConfig(
            n=int(config["n"]),
            m=int(config["m"]),
            q=int(config["q"]),
            input_mode="relation_pair",
            residue_encoding=str(config["residue_encoding"]),
            fourier_k=int(config["fourier_k"]),
            hidden_channels=int(config["hidden_channels"]),
            depth=int(config["depth"]),
            dropout=float(config["dropout"]),
        )
    )
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def support_labels(target: torch.Tensor) -> list[str]:
    labels: list[str] = []
    for row in target.cpu().to(torch.long):
        labels.append(",".join(str(i) for i in torch.nonzero(row, as_tuple=False).flatten().tolist()))
    return labels


@torch.no_grad()
def collect_embeddings(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, tuple[int, ...]]]:
    config = load_config(args.run_dir)
    dataset = make_dataset(config, args.num_samples, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = make_model(config, args.run_dir, device)

    before_rows: list[torch.Tensor] = []
    after_rows: list[torch.Tensor] = []
    labels: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}

    for batch in loader:
        A = batch["A"].to(device)
        b_input = batch["b_input"].to(device)
        batch_size = int(A.shape[0])

        # Exact tensor entering Conv1d: [B*m, 2, n].
        row_input = model._relation_pair_input(A, b_input)
        shapes["cnn_input_row_tensor"] = tuple(row_input.shape)
        before_rows.append(row_input.reshape(batch_size, -1).cpu())

        # Exact tensor after all Conv1d layers in stem+blocks: [B*m, hidden, n].
        row_features = model.blocks(model.stem(row_input))
        shapes["cnn_output_row_tensor"] = tuple(row_features.shape)
        after_rows.append(row_features.reshape(batch_size, -1).cpu())

        labels.extend(support_labels(batch["target"]))

    before = torch.cat(before_rows).numpy()
    after = torch.cat(after_rows).numpy()
    shapes["before_embedding_flat"] = tuple(before.shape)
    shapes["after_embedding_flat"] = tuple(after.shape)
    return before, after, labels, shapes


def tsne_2d(x: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(perplexity, max(5.0, (x.shape[0] - 1) / 3.0)),
        random_state=seed,
        max_iter=max_iter,
    ).fit_transform(x)


def colors_for(labels: list[str]) -> np.ndarray:
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([unique[label] for label in labels])


def save_coordinates(path: Path, before_2d: np.ndarray, after_2d: np.ndarray, labels: list[str]) -> None:
    with path.open("w") as handle:
        handle.write("sample,label,before_x,before_y,after_x,after_y\n")
        for idx, label in enumerate(labels):
            handle.write(
                f"{idx},{label},{before_2d[idx, 0]:.8f},{before_2d[idx, 1]:.8f},"
                f"{after_2d[idx, 0]:.8f},{after_2d[idx, 1]:.8f}\n"
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    before, after, labels, shapes = collect_embeddings(args, device)
    print("relation_pair tensor shapes:", json.dumps({k: list(v) for k, v in shapes.items()}, indent=2), flush=True)
    print(f"fitting before-CNN t-SNE: {before.shape}", flush=True)
    before_2d = tsne_2d(before, args.tsne_seed, args.perplexity, args.max_iter)
    print(f"fitting after-CNN t-SNE: {after.shape}", flush=True)
    after_2d = tsne_2d(after, args.tsne_seed, args.perplexity, args.max_iter)

    color = colors_for(labels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].scatter(before_2d[:, 0], before_2d[:, 1], c=color, cmap="tab20", s=16, alpha=0.78, linewidths=0)
    axes[0].set_title("relation_pair before CNN\nflatten([B*m, 2, n])")
    axes[1].scatter(after_2d[:, 0], after_2d[:, 1], c=color, cmap="tab20", s=16, alpha=0.78, linewidths=0)
    axes[1].set_title("relation_pair after CNN\nflatten([B*m, 64, n])")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Relation-pair t-SNE before and after Conv1d layers", fontsize=13)

    png_path = args.output_dir / "relation_pair_tsne_before_after_cnn.png"
    pdf_path = args.output_dir / "relation_pair_tsne_before_after_cnn.pdf"
    csv_path = args.output_dir / "relation_pair_tsne_before_after_cnn.csv"
    meta_path = args.output_dir / "relation_pair_tsne_before_after_cnn_meta.json"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    save_coordinates(csv_path, before_2d, after_2d, labels)
    meta_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "num_samples": args.num_samples,
                "seed": args.seed,
                "tsne_seed": args.tsne_seed,
                "perplexity": args.perplexity,
                "max_iter": args.max_iter,
                "shapes": {key: list(value) for key, value in shapes.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    print(f"saved {csv_path}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
