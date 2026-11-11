from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import shlex
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_idea2 import (  # noqa: E402
    CNN1DRowPoolConfig,
    CNN1DRowPoolModel,
    LWEAblationDataset,
    LWEDataConfig,
    LWEParams,
    all_zero_baseline_metrics,
    binary_support_statistics,
    finalize_statistics,
    merge_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LWE 1D CNN input ablation trainer.")
    parser.add_argument("--input-mode", dest="input_mode", choices=["relation_pair", "b_shuffle", "b_once", "all_zero"], default="relation_pair")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--q", type=int, default=257)
    parser.add_argument("--noise-width", dest="noise_width", type=float, default=1.0)
    parser.add_argument("--noise-dist", dest="noise_dist", choices=["zero", "uniform_small", "discrete_gaussian"], default="discrete_gaussian")
    parser.add_argument("--h-setting", dest="h_setting", choices=["fixed_h", "variable_h"], default="fixed_h")
    parser.add_argument("--fixed-h", dest="fixed_h", type=int, default=2)
    parser.add_argument("--h-min", dest="h_min", type=int, default=1)
    parser.add_argument("--h-max", dest="h_max", type=int, default=3)
    parser.add_argument("--residue-encoding", dest="residue_encoding", choices=["raw", "centered", "phase"], default="raw")
    parser.add_argument("--fourier-k", dest="fourier_k", type=int, default=1)
    parser.add_argument("--num-train", dest="num_train", type=int, default=10_000)
    parser.add_argument("--num-val", dest="num_val", type=int, default=1_000)
    parser.add_argument("--num-test", dest="num_test", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    parser.add_argument("--hidden-channels", dest="hidden_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--decode", choices=["top_h", "threshold"], default="top_h")
    parser.add_argument("--residual-success-factor", dest="residual_success_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def compact_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def default_run_name(args: argparse.Namespace) -> str:
    return (
        f"n{args.n}_h{args.fixed_h}_{args.input_mode}_cnn1d_"
        f"{compact_count(args.num_train)}_{compact_count(args.num_val)}_{compact_count(args.num_test)}_"
        f"ep{args.epochs}_seed{args.seed}"
    )


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def make_dataset(args: argparse.Namespace, num_samples: int, seed_offset: int) -> LWEAblationDataset:
    params = LWEParams(n=args.n, m=args.m, q=args.q, noise_width=args.noise_width, noise_dist=args.noise_dist)
    config = LWEDataConfig(
        num_samples=num_samples,
        params=params,
        seed=args.seed * 1000 + seed_offset,
        input_mode=args.input_mode,
        h_setting=args.h_setting,
        fixed_h=args.fixed_h,
        h_min=args.h_min,
        h_max=args.h_max,
    )
    return LWEAblationDataset(config)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(args, model, loader, device, optimizer=None) -> tuple[float, dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    sample_count = 0
    stat_rows: list[dict[str, float]] = []
    iterator = tqdm(loader, desc="train", leave=False) if train and args.progress else loader
    for batch in iterator:
        batch = to_device(batch, device)
        batch_size = int(batch["target"].shape[0])
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(batch["A"], batch["b_input"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["target"].to(torch.float32))
            if train:
                loss.backward()
                optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        loss_sum += loss_value * batch_size
        sample_count += batch_size
        stat_rows.append(
            binary_support_statistics(
                logits=logits,
                target=batch["target"],
                A=batch["A"],
                b=batch["b"],
                q=args.q,
                decode=args.decode,
                h=args.fixed_h,
                noise_width=args.noise_width,
                residual_success_factor=args.residual_success_factor,
            )
        )
        if train and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss_value:.4f}")
    return loss_sum / max(sample_count, 1), finalize_statistics(merge_statistics(stat_rows))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def run_all_zero(args: argparse.Namespace) -> None:
    if args.h_setting != "fixed_h":
        raise ValueError("The analytic all_zero baseline requires --h-setting fixed_h.")
    metrics = all_zero_baseline_metrics(args.n, args.fixed_h)
    run_name = args.run_name or f"n{args.n}_h{args.fixed_h}_all_zero"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"input_mode": "all_zero", **metrics}
    write_json(run_dir / "summary.json", summary)
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.input_mode == "all_zero":
        run_all_zero(args)
        return
    if args.h_setting != "fixed_h" and args.decode == "top_h":
        raise ValueError("top_h decoding currently requires fixed_h.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.run_name is None:
        args.run_name = default_run_name(args)

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n")
    with (run_dir / "train.log").open("a") as log_handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
            train_model(args, run_dir)


def train_model(args: argparse.Namespace, run_dir: Path) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader = DataLoader(make_dataset(args, args.num_train, 11), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(make_dataset(args, args.num_val, 23), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(make_dataset(args, args.num_test, 37), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = CNN1DRowPoolModel(
        CNN1DRowPoolConfig(
            n=args.n,
            m=args.m,
            q=args.q,
            input_mode=args.input_mode,
            residue_encoding=args.residue_encoding,
            fourier_k=args.fourier_k,
            hidden_channels=args.hidden_channels,
            depth=args.depth,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["num_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    write_json(run_dir / "config.json", config)
    print(json.dumps(config, indent=2, sort_keys=True))

    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_metrics = run_epoch(args, model, train_loader, device, optimizer)
        val_loss, val_metrics = run_epoch(args, model, val_loader, device)
        epoch_time = time.perf_counter() - start
        score = val_metrics["support_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row: dict[str, object] = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_time_sec": epoch_time}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        save_csv(run_dir / "history.csv", history)
        write_json(run_dir / "history.json", history)
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, run_dir / "latest.pt")
        print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_metrics['support_f1']:.4f} val_exact={val_metrics['exact_match']:.4f} time={epoch_time:.2f}s")
        print("  train", train_metrics)
        print("  val  ", val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"epoch": best_epoch, "best_score": best_score, "model_state": best_state}, run_dir / "best.pt")

    test_loss, test_metrics = run_epoch(args, model, test_loader, device)
    summary = {"best_epoch": best_epoch, "best_val_support_f1": best_score, "test_loss": test_loss, **{f"test_{key}": value for key, value in test_metrics.items()}}
    write_json(run_dir / "summary.json", summary)
    print("=== Test Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
