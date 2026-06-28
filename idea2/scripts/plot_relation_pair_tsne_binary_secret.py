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
    parser = argparse.ArgumentParser(description="Plot relation_pair coordinate embeddings colored by secret bit 0/1.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "n16_h2_noise3_m64_relation_pair_cnn1d_10k_1k_1k_ep200_seed0",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "result")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-points-per-class", type=int, default=500)
    parser.add_argument("--sampling", choices=["balanced", "natural"], default="natural")
    parser.add_argument("--max-total-points", type=int, default=2400)
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


@torch.no_grad()
def collect_coordinate_embeddings(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    config = load_config(args.run_dir)
    dataset = make_dataset(config, args.num_samples, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = make_model(config, args.run_dir, device)

    before_rows: list[torch.Tensor] = []
    after_rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    coord_ids: list[torch.Tensor] = []
    shapes: dict[str, tuple[int, ...]] = {}

    for batch in loader:
        A = batch["A"].to(device)
        b_input = batch["b_input"].to(device)
        target = batch["target"].to(torch.long)
        batch_size = int(A.shape[0])
        n = int(model.config.n)
        m = int(model.config.m)

        row_input = model._relation_pair_input(A, b_input)
        shapes["cnn_input_row_tensor"] = tuple(row_input.shape)
        input_features = row_input.reshape(batch_size, m, 2, n).permute(0, 3, 1, 2).reshape(batch_size * n, m * 2)
        before_rows.append(input_features.cpu())

        row_features = model.blocks(model.stem(row_input))
        shapes["cnn_output_row_tensor"] = tuple(row_features.shape)
        conv_features = row_features.reshape(batch_size, m, model.config.hidden_channels, n)
        conv_features = conv_features.permute(0, 3, 1, 2).reshape(batch_size * n, m * model.config.hidden_channels)
        after_rows.append(conv_features.cpu())

        labels.append(target.reshape(-1).cpu())
        coord_ids.append(torch.arange(n).repeat(batch_size))

    before = torch.cat(before_rows).numpy()
    after = torch.cat(after_rows).numpy()
    y = torch.cat(labels).numpy()
    coords = torch.cat(coord_ids).numpy()
    shapes["before_coordinate_embedding"] = tuple(before.shape)
    shapes["after_coordinate_embedding"] = tuple(after.shape)
    return before, after, y, coords, shapes


def balanced_subset(y: np.ndarray, max_points_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_parts = []
    for value in [0, 1]:
        idx = np.flatnonzero(y == value)
        if len(idx) > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        keep_parts.append(idx)
    keep = np.concatenate(keep_parts)
    rng.shuffle(keep)
    return keep


def natural_subset(y: np.ndarray, max_total_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(y) <= max_total_points:
        keep = np.arange(len(y))
    else:
        keep = rng.choice(np.arange(len(y)), size=max_total_points, replace=False)
    rng.shuffle(keep)
    return keep


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


def save_coordinates(path: Path, before_2d: np.ndarray, after_2d: np.ndarray, y: np.ndarray, coords: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("point,secret_bit,coordinate,before_x,before_y,after_x,after_y\n")
        for idx in range(len(y)):
            handle.write(
                f"{idx},{int(y[idx])},{int(coords[idx])},{before_2d[idx, 0]:.8f},{before_2d[idx, 1]:.8f},"
                f"{after_2d[idx, 0]:.8f},{after_2d[idx, 1]:.8f}\n"
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    before, after, y, coords, shapes = collect_coordinate_embeddings(args, device)
    if args.sampling == "balanced":
        keep = balanced_subset(y, args.max_points_per_class, args.seed)
    else:
        keep = natural_subset(y, args.max_total_points, args.seed)
    before = before[keep]
    after = after[keep]
    y = y[keep]
    coords = coords[keep]

    print("relation_pair coordinate tensor shapes:", json.dumps({k: list(v) for k, v in shapes.items()}, indent=2), flush=True)
    print(f"plotted points: zeros={(y == 0).sum()}, ones={(y == 1).sum()}, sampling={args.sampling}", flush=True)
    print(f"fitting before-CNN t-SNE: {before.shape}", flush=True)
    before_2d = tsne_2d(before, args.tsne_seed, args.perplexity, args.max_iter)
    print(f"fitting after-CNN t-SNE: {after.shape}", flush=True)
    after_2d = tsne_2d(after, args.tsne_seed, args.perplexity, args.max_iter)

    colors = np.where(y == 1, 1, 0)
    cmap = plt.get_cmap("coolwarm", 2)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for ax, points, title in [
        (axes[0], before_2d, "Before CNN coordinate embedding\nlabel = secret bit s_j"),
        (axes[1], after_2d, "After CNN coordinate embedding\nlabel = secret bit s_j"),
    ]:
        ax.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap, s=16, alpha=0.78, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0), label="0", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(1), label="1", markersize=8),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=True, title="secret bit")
    fig.suptitle("Relation-pair t-SNE colored by binary secret coordinate", fontsize=13)

    png_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn.png"
    pdf_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn.pdf"
    csv_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn.csv"
    meta_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn_meta.json"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    save_coordinates(csv_path, before_2d, after_2d, y, coords)
    meta_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "num_samples": args.num_samples,
                "seed": args.seed,
                "tsne_seed": args.tsne_seed,
                "perplexity": args.perplexity,
                "max_iter": args.max_iter,
                "max_points_per_class": args.max_points_per_class,
                "max_total_points": args.max_total_points,
                "sampling": args.sampling,
                "class_counts_after_balance": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
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
