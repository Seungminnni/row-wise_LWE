from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .lwe import LWEParams, sample_fixed_h_binary_secret, sample_noise, sample_variable_h_binary_secret


@dataclass(frozen=True)
class LWEDataConfig:
    num_samples: int
    params: LWEParams
    seed: int = 0
    input_mode: str = "relation_pair"
    h_setting: str = "fixed_h"
    fixed_h: int = 2
    h_min: int = 1
    h_max: int = 3

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.input_mode not in {"relation_pair", "b_shuffle", "b_once"}:
            raise ValueError("input_mode must be relation_pair, b_shuffle, or b_once.")
        if self.h_setting not in {"fixed_h", "variable_h"}:
            raise ValueError("h_setting must be fixed_h or variable_h.")
        if self.h_setting == "fixed_h" and not 0 <= self.fixed_h <= self.params.n:
            raise ValueError("fixed_h must be in [0,n].")
        if self.h_setting == "variable_h" and not 0 <= self.h_min <= self.h_max <= self.params.n:
            raise ValueError("expected 0 <= h_min <= h_max <= n.")


class LWEAblationDataset(Dataset):
    """On-the-fly dataset for input representation ablations.

    b_shuffle is applied inside each instance by permuting the m response rows.
    The original b and secret label are kept for residual/metric computation.
    """

    sample_stride = 1_000_003

    def __init__(self, config: LWEDataConfig) -> None:
        self.config = config
        self.params = config.params

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.config.num_samples:
            raise IndexError(idx)
        gen = self._generator(idx)
        A = torch.randint(self.params.q, (self.params.m, self.params.n), generator=gen, dtype=torch.long)
        s = self._sample_secret(gen)
        e = sample_noise(self.params, gen)
        b = torch.remainder((A * s.unsqueeze(0)).sum(dim=-1) + e, self.params.q).to(torch.long)

        if self.config.input_mode == "b_shuffle":
            perm = torch.randperm(self.params.m, generator=gen)
            b_input = b[perm]
        else:
            b_input = b

        return {
            "A": A,
            "b": b,
            "b_input": b_input,
            "target": s,
            "noise": e,
            "h": s.sum().to(torch.long),
        }

    def _generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator(device="cpu")
        gen.manual_seed((self.config.seed + idx * self.sample_stride) % (2**63 - 1))
        return gen

    def _sample_secret(self, gen: torch.Generator) -> torch.Tensor:
        if self.config.h_setting == "fixed_h":
            return sample_fixed_h_binary_secret(self.params.n, self.config.fixed_h, gen)
        return sample_variable_h_binary_secret(self.params.n, self.config.h_min, self.config.h_max, gen)

