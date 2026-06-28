from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .lwe import centered_mod


@dataclass(frozen=True)
class CNN1DRowPoolConfig:
    n: int = 10
    m: int = 256
    q: int = 257
    input_mode: str = "relation_pair"
    residue_encoding: str = "raw"
    fourier_k: int = 1
    hidden_channels: int = 64
    depth: int = 3
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0 or self.q <= 2:
            raise ValueError("n/m must be positive and q must be greater than 2.")
        if self.input_mode not in {"relation_pair", "b_shuffle", "b_once"}:
            raise ValueError("input_mode must be relation_pair, b_shuffle, or b_once.")
        if self.residue_encoding not in {"raw", "centered", "phase"}:
            raise ValueError("unsupported residue_encoding.")
        if self.fourier_k < 1:
            raise ValueError("fourier_k must be at least 1.")
        if self.hidden_channels <= 0 or self.depth <= 0:
            raise ValueError("hidden_channels and depth must be positive.")


def residue_dim(residue_encoding: str, fourier_k: int) -> int:
    if residue_encoding in {"raw", "centered"}:
        return 1
    return 1 + 2 * fourier_k


def encode_residue(x: torch.Tensor, q: int, residue_encoding: str, fourier_k: int) -> torch.Tensor:
    x = torch.remainder(x.to(torch.long), q)
    if residue_encoding == "raw":
        return (x.to(torch.float32) / float(q - 1)).unsqueeze(-1)

    centered = centered_mod(x, q).to(torch.float32) / (float(q) / 2.0)
    if residue_encoding == "centered":
        return centered.unsqueeze(-1)

    feats = [centered]
    x_float = x.to(torch.float32)
    for k in range(1, fourier_k + 1):
        angle = x_float * (2.0 * math.pi * float(k) / float(q))
        feats.append(torch.sin(angle))
        feats.append(torch.cos(angle))
    return torch.stack(feats, dim=-1)


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _norm_groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNN1DRowPoolModel(nn.Module):
    """1D CNN over coordinate axis with row mean+max pooling."""

    def __init__(self, config: CNN1DRowPoolConfig) -> None:
        super().__init__()
        self.config = config
        self.base_dim = residue_dim(config.residue_encoding, config.fourier_k)
        self.in_channels = self.base_dim if config.input_mode == "b_once" else 2 * self.base_dim
        groups = _norm_groups(config.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, config.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvBlock(config.hidden_channels, config.dropout) for _ in range(config.depth - 1)])
        self.head = nn.Linear(2 * config.hidden_channels, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return encode_residue(x, self.config.q, self.config.residue_encoding, self.config.fourier_k)

    def _relation_pair_input(self, A: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
        batch, m, n = A.shape
        A_enc = self._encode(A)
        b_enc = self._encode(b_input).unsqueeze(2).expand(-1, -1, n, -1)
        row_pixels = torch.cat([A_enc, b_enc], dim=-1)
        return row_pixels.reshape(batch * m, n, self.in_channels).permute(0, 2, 1).contiguous()

    def _b_once_input(self, A: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
        batch, m, n = A.shape
        A_enc = self._encode(A)
        b_enc = self._encode(b_input).unsqueeze(2)
        row_sequence = torch.cat([A_enc, b_enc], dim=2)
        return row_sequence.reshape(batch * m, n + 1, self.in_channels).permute(0, 2, 1).contiguous()

    def forward(self, A: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if b_input.dim() == 1:
            b_input = b_input.unsqueeze(0)
        batch, m, n = A.shape
        if (m, n) != (self.config.m, self.config.n):
            raise ValueError(f"expected A shape [B,{self.config.m},{self.config.n}], got {tuple(A.shape)}")
        if b_input.shape != (batch, m):
            raise ValueError(f"expected b_input shape [{batch},{m}], got {tuple(b_input.shape)}")

        if self.config.input_mode == "b_once":
            row_input = self._b_once_input(A, b_input)
        else:
            row_input = self._relation_pair_input(A, b_input)

        row_features = self.blocks(self.stem(row_input))
        if self.config.input_mode == "b_once":
            row_features = row_features[:, :, :n]
        features = row_features.reshape(batch, m, self.config.hidden_channels, n)
        mean_features = features.mean(dim=1).permute(0, 2, 1)
        max_features = features.max(dim=1).values.permute(0, 2, 1)
        pooled = torch.cat([mean_features, max_features], dim=-1)
        return self.head(pooled).squeeze(-1)

