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


RUNS = {
    "relation_pair": "n16_h2_noise3_m64_relation_pair_cnn1d_10k_1k_1k_ep200_seed0",
    "b_once": "n16_h2_noise3_m64_b_once_cnn1d_10k_1k_1k_ep200_seed0",
    "b_shuffle": "n16_h2_noise3_m64_b_shuffle_cnn1d_10k_1k_1k_ep200_seed0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE embeddings before/after the CNN for idea2 input modes.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "plots" / "tsne")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def make_dataset(config: dict, input_mode: str, num_samples: int, seed: int) -> LWEAblationDataset:
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
            input_mode=input_mode if input_mode != "all_zero" else "relation_pair",
            h_setting=str(config["h_setting"]),
            fixed_h=int(config["fixed_h"]),
            h_min=int(config["h_min"]),
            h_max=int(config["h_max"]),
        )
    )


def make_model(config: dict, input_mode: str, run_dir: Path, device: torch.device) -> CNN1DRowPoolModel:
    model = CNN1DRowPoolModel(
        CNN1DRowPoolConfig(
            n=int(config["n"]),
            m=int(config["m"]),
            q=int(config["q"]),
            input_mode=input_mode,
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


@torch.no_grad()
def input_embedding(model: CNN1DRowPoolModel, A: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
    if model.config.input_mode == "b_once":
        row_input = model._b_once_input(A, b_input)
    else:
        row_input = model._relation_pair_input(A, b_input)
    batch = A.shape[0]
    return row_input.reshape(batch, -1).cpu()


@torch.no_grad()
def cnn_embedding(model: CNN1DRowPoolModel, A: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
    if model.config.input_mode == "b_once":
        row_input = model._b_once_input(A, b_input)
    else:
        row_input = model._relation_pair_input(A, b_input)
    row_features = model.blocks(model.stem(row_input))
    if model.config.input_mode == "b_once":
        row_features = row_features[:, :, : model.config.n]
    features = row_features.reshape(A.shape[0], model.config.m, model.config.hidden_channels, model.config.n)
    mean_features = features.mean(dim=1).permute(0, 2, 1)
    max_features = features.max(dim=1).values.permute(0, 2, 1)
    return torch.cat([mean_features, max_features], dim=-1).reshape(A.shape[0], -1).cpu()


def support_labels(target: torch.Tensor) -> list[str]:
    labels: list[str] = []
    for row in target.cpu().to(torch.long):
        labels.append(",".join(str(i) for i in torch.nonzero(row, as_tuple=False).flatten().tolist()))
    return labels


def collect_embeddings(args: argparse.Namespace, input_mode: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reference_config = load_config(args.runs_dir / RUNS["relation_pair"])
    dataset = make_dataset(reference_config, input_mode, args.num_samples, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if input_mode == "all_zero":
        before_rows, after_rows, labels = [], [], []
        n = int(reference_config["n"])
        for batch in loader:
            batch_size = int(batch["target"].shape[0])
            before_rows.append(torch.zeros(batch_size, n))
            after_rows.append(torch.zeros(batch_size, n))
            labels.extend(support_labels(batch["target"]))
        return torch.cat(before_rows).numpy(), torch.cat(after_rows).numpy(), labels

    run_dir = args.runs_dir / RUNS[input_mode]
    config = load_config(run_dir)
    model = make_model(config, input_mode, run_dir, device)
    before_rows, after_rows, labels = [], [], []
    for batch in loader:
        A = batch["A"].to(device)
        b_input = batch["b_input"].to(device)
        before_rows.append(input_embedding(model, A, b_input))
        after_rows.append(cnn_embedding(model, A, b_input))
        labels.extend(support_labels(batch["target"]))
    return torch.cat(before_rows).numpy(), torch.cat(after_rows).numpy(), labels


def tsne_2d(x: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    if np.allclose(x, x[0]):
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    x = StandardScaler().fit_transform(x)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(perplexity, max(1.0, (x.shape[0] - 1) / 3.0)),
        random_state=seed,
        max_iter=max_iter,
    ).fit_transform(x)


def colors_for(labels: list[str]) -> np.ndarray:
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([unique[label] for label in labels])


def plot_panel(ax, points: np.ndarray, labels: list[str], title: str, degenerate: bool) -> None:
    color = colors_for(labels)
    ax.scatter(points[:, 0], points[:, 1], c=color, s=12, cmap="tab20", alpha=0.78, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if degenerate:
        ax.text(0.5, 0.5, "no learned input/CNN\nall points collapse", transform=ax.transAxes, ha="center", va="center", fontsize=9)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    modes = ["relation_pair", "b_once", "b_shuffle", "all_zero"]
    fig, axes = plt.subplots(len(modes), 2, figsize=(9, 13), constrained_layout=True)
    for row, mode in enumerate(modes):
        print(f"collecting {mode}", flush=True)
        before, after, labels = collect_embeddings(args, mode, device)
        print(f"fitting t-SNE for {mode} before CNN: {before.shape}", flush=True)
        before_2d = tsne_2d(before, args.tsne_seed, args.perplexity, args.max_iter)
        print(f"fitting t-SNE for {mode} after CNN: {after.shape}", flush=True)
        after_2d = tsne_2d(after, args.tsne_seed, args.perplexity, args.max_iter)
        degenerate = mode == "all_zero"
        plot_panel(axes[row, 0], before_2d, labels, f"{mode}: before CNN", degenerate)
        plot_panel(axes[row, 1], after_2d, labels, f"{mode}: after CNN/pooling", degenerate)
    fig.suptitle("t-SNE of sample embeddings before and after CNN (n=16, h=2, noise=3)", fontsize=13)
    png_path = args.output_dir / "n16_h2_noise3_m64_tsne_before_after.png"
    pdf_path = args.output_dir / "n16_h2_noise3_m64_tsne_before_after.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
