from __future__ import annotations

import torch

from .lwe import residual_from_binary_support


def decode_logits(logits: torch.Tensor, mode: str, h: int) -> torch.Tensor:
    if mode == "top_h":
        if h < 0 or h > logits.shape[1]:
            raise ValueError("h must be in [0,n].")
        pred = torch.zeros_like(logits, dtype=torch.long)
        if h == 0:
            return pred
        indices = torch.topk(logits, k=h, dim=1).indices
        pred.scatter_(1, indices, 1)
        return pred
    if mode == "threshold":
        return torch.sigmoid(logits).ge(0.5).to(torch.long)
    raise ValueError("decode mode must be top_h or threshold.")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


@torch.no_grad()
def binary_support_statistics(
    logits: torch.Tensor,
    target: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    decode: str,
    h: int,
    noise_width: float = 1.0,
    residual_success_factor: float = 2.0,
) -> dict[str, float]:
    pred = decode_logits(logits, decode, h).to(torch.long)
    target = target.to(torch.long)
    correct = pred.eq(target)
    exact = correct.all(dim=1)

    pred_active = pred.ne(0)
    true_active = target.ne(0)
    tp = (pred_active & true_active).sum().item()
    fp = (pred_active & ~true_active).sum().item()
    fn = (~pred_active & true_active).sum().item()
    tn = (~pred_active & ~true_active).sum().item()

    sample_tp = (pred_active & true_active).sum(dim=1).to(torch.float32)
    sample_fp = (pred_active & ~true_active).sum(dim=1).to(torch.float32)
    sample_fn = (~pred_active & true_active).sum(dim=1).to(torch.float32)
    sample_precision = sample_tp / (sample_tp + sample_fp).clamp(min=1.0)
    sample_recall = sample_tp / (sample_tp + sample_fn).clamp(min=1.0)
    sample_f1_den = sample_precision + sample_recall
    sample_empty_match = (pred_active.sum(dim=1) == 0) & (true_active.sum(dim=1) == 0)
    sample_f1 = torch.where(
        sample_f1_den > 0,
        2.0 * sample_precision * sample_recall / sample_f1_den,
        sample_empty_match.to(torch.float32),
    )

    residual = residual_from_binary_support(A, b, pred, q).to(torch.float32)
    residual_std = residual.std(dim=1, unbiased=False)
    if noise_width == 0.0:
        residual_success = residual.eq(0).all(dim=1)
    else:
        residual_success = residual_std.le(residual_success_factor * noise_width)

    return {
        "sample_count": float(target.shape[0]),
        "coord_count": float(target.numel()),
        "coord_correct": float(correct.sum().item()),
        "exact_correct": float(exact.sum().item()),
        "support_tp": float(tp),
        "support_fp": float(fp),
        "support_fn": float(fn),
        "support_tn": float(tn),
        "support_exact_count": float((pred_active == true_active).all(dim=1).sum().item()),
        "sample_support_f1_sum": float(sample_f1.sum().item()),
        "pred_support_count": float(pred_active.sum().item()),
        "true_support_count": float(true_active.sum().item()),
        "pred_residual_std_sum": float(residual_std.sum().item()),
        "pred_residual_std_gap_sum": float((residual_std - noise_width).abs().sum().item()),
        "residual_success_count": float(residual_success.sum().item()),
    }


def merge_statistics(stat_list: list[dict[str, float]]) -> dict[str, float]:
    if not stat_list:
        raise ValueError("stat_list must not be empty.")
    keys = set().union(*(stats.keys() for stats in stat_list))
    merged = {key: 0.0 for key in keys}
    for stats in stat_list:
        for key, value in stats.items():
            merged[key] += value
    return merged


def finalize_statistics(stats: dict[str, float]) -> dict[str, float]:
    precision = _safe_div(stats["support_tp"], stats["support_tp"] + stats["support_fp"])
    recall = _safe_div(stats["support_tp"], stats["support_tp"] + stats["support_fn"])
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    coord_acc = _safe_div(stats["coord_correct"], stats["coord_count"])
    exact_match = _safe_div(stats["exact_correct"], stats["sample_count"])
    return {
        "exact_match": exact_match,
        "exact_recovery_rate": exact_match,
        "support_f1": f1,
        "support_f1_micro": f1,
        "support_f1_sample_mean": _safe_div(stats["sample_support_f1_sum"], stats["sample_count"]),
        "support_precision": precision,
        "support_recall": recall,
        "active_recall": recall,
        "coord_acc": coord_acc,
        "coordinate_accuracy": coord_acc,
        "normalized_hamming_error": 1.0 - coord_acc,
        "support_exact_match": _safe_div(stats["support_exact_count"], stats["sample_count"]),
        "support_tp": stats["support_tp"],
        "support_fp": stats["support_fp"],
        "support_fn": stats["support_fn"],
        "support_tn": stats["support_tn"],
        "support_pred_rate": _safe_div(stats["pred_support_count"], stats["coord_count"]),
        "support_true_rate": _safe_div(stats["true_support_count"], stats["coord_count"]),
        "pred_residual_std_mean": _safe_div(stats["pred_residual_std_sum"], stats["sample_count"]),
        "pred_residual_std_gap_mean": _safe_div(stats["pred_residual_std_gap_sum"], stats["sample_count"]),
        "residual_success_rate": _safe_div(stats["residual_success_count"], stats["sample_count"]),
    }


@torch.no_grad()
def binary_support_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    decode: str,
    h: int,
    noise_width: float = 1.0,
    residual_success_factor: float = 2.0,
) -> dict[str, float]:
    stats = binary_support_statistics(
        logits=logits,
        target=target,
        A=A,
        b=b,
        q=q,
        decode=decode,
        h=h,
        noise_width=noise_width,
        residual_success_factor=residual_success_factor,
    )
    return finalize_statistics(stats)


def all_zero_baseline_metrics(n: int, h: int) -> dict[str, float]:
    if n <= 0 or not 0 <= h <= n:
        raise ValueError("n must be positive and h must be in [0,n].")
    coord_acc = float(n - h) / float(n)
    exact_match = 1.0 if h == 0 else 0.0
    return {
        "exact_match": exact_match,
        "exact_recovery_rate": exact_match,
        "support_f1": 0.0,
        "support_f1_micro": 0.0,
        "support_f1_sample_mean": exact_match,
        "support_precision": 0.0,
        "support_recall": 0.0,
        "active_recall": 0.0,
        "coord_acc": coord_acc,
        "coordinate_accuracy": coord_acc,
        "normalized_hamming_error": 1.0 - coord_acc,
        "support_exact_match": exact_match,
        "support_pred_rate": 0.0,
        "support_true_rate": float(h) / float(n),
    }
