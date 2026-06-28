from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LWEParams:
    n: int = 10
    m: int = 256
    q: int = 257
    noise_width: float = 1.0
    noise_dist: str = "discrete_gaussian"

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive.")
        if self.q <= 2:
            raise ValueError("q must be greater than 2.")
        if self.noise_width < 0:
            raise ValueError("noise_width must be non-negative.")
        if self.noise_dist not in {"zero", "uniform_small", "discrete_gaussian"}:
            raise ValueError("unsupported noise_dist.")


def centered_mod(x: torch.Tensor, q: int) -> torch.Tensor:
    return torch.remainder(x.to(torch.long) + q // 2, q) - q // 2


def sample_noise(params: LWEParams, gen: torch.Generator) -> torch.Tensor:
    shape = (params.m,)
    if params.noise_dist == "zero":
        return torch.zeros(shape, dtype=torch.long)
    if params.noise_dist == "uniform_small":
        radius = int(round(params.noise_width))
        return torch.randint(-radius, radius + 1, shape, generator=gen, dtype=torch.long)
    return torch.round(torch.randn(shape, generator=gen) * float(params.noise_width)).to(torch.long)


def sample_fixed_h_binary_secret(n: int, h: int, gen: torch.Generator) -> torch.Tensor:
    if h < 0 or h > n:
        raise ValueError("h must be in [0,n].")
    scores = torch.rand((n,), generator=gen)
    rank = scores.argsort(descending=True).argsort()
    return rank.lt(h).to(torch.long)


def sample_variable_h_binary_secret(n: int, h_min: int, h_max: int, gen: torch.Generator) -> torch.Tensor:
    if h_min < 0 or h_max < h_min or h_max > n:
        raise ValueError("expected 0 <= h_min <= h_max <= n.")
    h = int(torch.randint(h_min, h_max + 1, (1,), generator=gen).item())
    return sample_fixed_h_binary_secret(n, h, gen)


def residual_from_binary_support(A: torch.Tensor, b: torch.Tensor, support: torch.Tensor, q: int) -> torch.Tensor:
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if support.dim() == 1:
        support = support.unsqueeze(0)
    pred = (A.to(torch.long) * support.to(torch.long).unsqueeze(1)).sum(dim=-1)
    return centered_mod(b.to(torch.long) - pred, q)

