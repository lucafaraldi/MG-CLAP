"""Training utilities: model, loss, gap metric."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """A single linear projection — the minimum trainable structure.

    Mirrors what CLIP / CLAP put as the last layer of their encoders before the
    L2 normalization. Initialized with a controllable bias so we can plant a
    specific INITIAL gap and observe how it evolves under InfoNCE training.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        init_shift: Optional[torch.Tensor] = None,
        init_shift_strength: float = 0.0,
        weight_scale: float = 0.02,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.normal_(self.linear.weight, std=weight_scale)
        # Plant an initial bias along init_shift (a unit vector) with strength
        # init_shift_strength. Used to seed an audio head with +Δ/2 and a text
        # head with -Δ/2.
        with torch.no_grad():
            if init_shift is not None:
                assert init_shift.shape == (out_dim,)
                self.linear.bias.copy_(init_shift_strength * init_shift)
            else:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DualEncoderHead(nn.Module):
    """Two projection heads (audio + text). Embeddings come in pre-extracted from
    frozen CLAP backbones; we only train these heads.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        out_dim: Optional[int] = None,
        init_gap: float = 0.0,
        learn_temperature: bool = True,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        out_dim = out_dim or audio_dim
        # Plant ±init_gap/2 along the first basis vector — symmetric, just like
        # the post-hoc shift_features uses Δ.
        e = torch.zeros(out_dim); e[0] = 1.0
        self.audio_head = ProjectionHead(audio_dim, out_dim, init_shift=e, init_shift_strength=+0.5 * init_gap)
        self.text_head = ProjectionHead(text_dim, out_dim, init_shift=e, init_shift_strength=-0.5 * init_gap)
        # log-τ so τ stays positive under gradient updates
        log_tau = torch.tensor(np.log(init_temperature), dtype=torch.float32)
        if learn_temperature:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)
        self.learn_temperature = learn_temperature

    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, audio: torch.Tensor, text: torch.Tensor):
        a = F.normalize(self.audio_head(audio), dim=-1)
        t = F.normalize(self.text_head(text), dim=-1)
        return a, t


def info_nce_torch(audio_emb: torch.Tensor, text_emb: torch.Tensor, temperature) -> torch.Tensor:
    """Symmetric InfoNCE — same as `lib.gap_utils.info_nce_loss` but kept here
    so the training module is self-contained and stays on-device."""
    n = audio_emb.shape[0]
    tau = temperature if isinstance(temperature, torch.Tensor) else torch.tensor(temperature, device=audio_emb.device)
    logits_at = (audio_emb @ text_emb.T) / tau
    target = torch.arange(n, device=audio_emb.device)
    l_at = F.cross_entropy(logits_at, target)
    l_ta = F.cross_entropy(logits_at.T, target)
    return 0.5 * (l_at + l_ta)


@torch.no_grad()
def measure_gap(model: DualEncoderHead, audio: torch.Tensor, text: torch.Tensor) -> float:
    """‖Δ‖₂ on the projected, L2-normalized embeddings."""
    a, t = model(audio, text)
    return float(torch.linalg.vector_norm(a.mean(0) - t.mean(0)).item())
