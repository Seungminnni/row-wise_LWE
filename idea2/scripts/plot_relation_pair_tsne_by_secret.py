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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_idea2 import CNN1DRowPoolConfig, CNN1DRowPoolModel, LWEParams  # noqa: E402
from lwe_idea2.lwe import sample_noise  # noqa: E402


DEFAULT_SUPPORTS = ["0,1", "0,5", "1,7", "2,9", "3,11", "4,13", "6,14", "10,15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot relation_pair t-SNE before/after CNN, colored by fixed secret support.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "n16_h2_noise3_m64_relation_pair_cnn1d_10k_1k_1k_ep200_seed0",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "result")
    parser.add_argument("--supports", nargs="+", default=DEFAULT_SUPPORTS)
    parser.add_argument("--samples-per-secret", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def parse_support(text: str, n: int) -> torch.Tensor:
    indices = [int(part) for part in text.split(",") if part != ""]
    s = torch.zeros(n, dtype=torch.long)
    for idx in indices:
        if idx < 0 or idx >= n:
            raise ValueError(f"support index {idx} is outside n={n}")
        s[idx] = 1
    return s


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


def make_fixed_secret_batch(config: dict, supports: list[str], samples_per_secret: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    n = int(config["n"])
    m = int(config["m"])
    q = int(config["q"])
    params = LWEParams(n=n, m=m, q=q, noise_width=float(config["noise_width"]), noise_dist=str(config["noise_dist"]))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    A_rows: list[torch.Tensor] = []
    b_rows: list[torch.Tensor] = []
    s_rows: list[torch.Tensor] = []
    labels: list[str] = []
    for support in supports:
        s = parse_support(support, n)
        for _ in range(samples_per_secret):
            A = torch.randint(q, (m, n), generator=gen, dtype=torch.long)
            e = sample_noise(params, gen)
            b = torch.remainder((A * s.unsqueeze(0)).sum(dim=-1) + e, q).to(torch.long)
            A_rows.append(A)
            b_rows.append(b)
            s_rows.append(s)
            labels.append(support)
    return torch.stack(A_rows), torch.stack(b_rows), torch.stack(s_rows), labels


@torch.no_grad()
def collect_embeddings(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, tuple[int, ...]]]:
    config = load_config(args.run_dir)
    model = make_model(config, args.run_dir, device)
    A, b, _target, labels = make_fixed_secret_batch(config, args.supports, args.samples_per_secret, args.seed)
    A = A.to(device)
    b = b.to(device)

    row_input = model._relation_pair_input(A, b)
    row_features = model.blocks(model.stem(row_input))
    batch_size = int(A.shape[0])
    before = row_input.reshape(batch_size, -1).cpu().numpy()
    after = row_features.reshape(batch_size, -1).cpu().numpy()
    shapes = {
        "A": tuple(A.shape),
        "b": tuple(b.shape),
        "cnn_input_row_tensor": tuple(row_input.shape),
        "cnn_output_row_tensor": tuple(row_features.shape),
        "before_embedding_flat": tuple(before.shape),
        "after_embedding_flat": tuple(after.shape),
    }
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


def save_coordinates(path: Path, before_2d: np.ndarray, after_2d: np.ndarray, labels: list[str]) -> None:
    with path.open("w") as handle:
        handle.write("sample,secret_support,before_x,before_y,after_x,after_y\n")
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
    print("relation_pair fixed-secret tensor shapes:", json.dumps({k: list(v) for k, v in shapes.items()}, indent=2), flush=True)
    print(f"fitting before-CNN t-SNE: {before.shape}", flush=True)
    before_2d = tsne_2d(before, args.tsne_seed, args.perplexity, args.max_iter)
    print(f"fitting after-CNN t-SNE: {after.shape}", flush=True)
    after_2d = tsne_2d(after, args.tsne_seed, args.perplexity, args.max_iter)

    supports = list(dict.fromkeys(labels))
    color_map = {support: idx for idx, support in enumerate(supports)}
    colors = np.array([color_map[label] for label in labels])
    cmap = plt.get_cmap("tab10", len(supports))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, points, title in [
        (axes[0], before_2d, "Before CNN: flatten([B*m, 2, n])"),
        (axes[1], after_2d, "After CNN: flatten([B*m, 64, n])"),
    ]:
        scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap, s=18, alpha=0.82, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(idx), label=support, markersize=7)
        for support, idx in color_map.items()
    ]
    fig.legend(handles=handles, title="secret support", loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Relation-pair t-SNE colored by fixed sparse secret support", fontsize=13)

    png_path = args.output_dir / "relation_pair_tsne_by_secret_before_after_cnn.png"
    pdf_path = args.output_dir / "relation_pair_tsne_by_secret_before_after_cnn.pdf"
    csv_path = args.output_dir / "relation_pair_tsne_by_secret_before_after_cnn.csv"
    meta_path = args.output_dir / "relation_pair_tsne_by_secret_before_after_cnn_meta.json"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    save_coordinates(csv_path, before_2d, after_2d, labels)
    meta_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "supports": args.supports,
                "samples_per_secret": args.samples_per_secret,
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
