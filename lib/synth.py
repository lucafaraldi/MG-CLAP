"""Synthetic embedding generators that mimic CLAP-style modality gap.

Used by every experiment as a fallback when no real CLAP cache is available
(handy for unit-testing the analysis pipeline). The synthetic distribution is
a pair of cones on the unit hypersphere, with a controllable gap and per-pair
correlation between audio and text.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .gap_utils import l2_normalize


def synth_paired_embeddings(
    n: int = 500,
    d: int = 512,
    gap: float = 0.6,
    pair_corr: float = 0.7,
    cone_kappa: float = 2000.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (audio, text) embedding pairs with a controllable post-normalization gap.

    Sanity-test-only helper — produces CLIP/CLAP-like geometry on the unit sphere:
    two cones offset along ±e_gap, with per-pair shared concepts driving positive
    aligned-pair cosine similarity.

    Defaults aim for a CLIP-like profile (gap ≈ 0.82 → pair_cos ≈ 0.3, cone ≈ 0.17).
    The exact gap is enforced via binary search on the anchor magnitude.

    Args:
        n: number of pairs.
        d: embedding dimensionality.
        gap: target post-normalization ‖Δ‖₂.
        pair_corr: per-pair concept correlation magnitude (positive pair_cos driver).
        cone_kappa: cone-tightness parameter (higher → tighter cones).
        seed: RNG seed.

    Returns:
        (audio_emb, text_emb) — both L2-normalized, shape (n, d).
    """
    rng = np.random.default_rng(seed)
    e_gap = np.zeros(d); e_gap[0] = 1.0

    # Per-pair shared "concept" direction — drives positive pair cosine sim.
    # Sampled in the subspace orthogonal to the gap axis so it doesn't interfere.
    shared = rng.normal(size=(n, d))
    shared[:, 0] = 0.0
    shared /= np.linalg.norm(shared, axis=-1, keepdims=True)

    # Independent intra-modal noise (controls cone width).
    audio_noise = rng.normal(size=(n, d)) / np.sqrt(cone_kappa)
    text_noise = rng.normal(size=(n, d)) / np.sqrt(cone_kappa)

    base_audio = pair_corr * shared + audio_noise
    base_text = pair_corr * shared + text_noise

    # Binary search the modality anchor magnitude along ±e_gap.
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        audio = l2_normalize(base_audio + mid * e_gap)
        text = l2_normalize(base_text - mid * e_gap)
        g = float(np.linalg.norm(audio.mean(0) - text.mean(0)))
        if g < gap:
            lo = mid
        else:
            hi = mid
    audio = l2_normalize(base_audio + 0.5 * (lo + hi) * e_gap)
    text = l2_normalize(base_text - 0.5 * (lo + hi) * e_gap)
    return audio, text


def synth_class_prompts(
    n_classes: int = 50,
    n_per_class: int = 40,
    d: int = 512,
    gap: float = 0.6,
    class_separation: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic ESC-50-style data: class-conditional audio embeddings + class-prompt
    text embeddings. Returns (audio_emb [N,d], text_emb [n_classes,d], labels [N]).
    """
    rng = np.random.default_rng(seed)
    e0 = np.zeros(d); e0[0] = 1.0
    a_center = +0.5 * gap * e0
    t_center = -0.5 * gap * e0

    # One class direction per class in the subspace orthogonal to the gap axis.
    class_dirs = rng.normal(size=(n_classes, d))
    class_dirs[:, 0] = 0.0
    class_dirs /= np.linalg.norm(class_dirs, axis=-1, keepdims=True)

    text_emb = t_center + class_separation * class_dirs
    text_emb = l2_normalize(text_emb)

    n = n_classes * n_per_class
    labels = np.repeat(np.arange(n_classes), n_per_class)
    audio = a_center + class_separation * class_dirs[labels] + rng.normal(scale=0.4, size=(n, d))
    audio = l2_normalize(audio)

    return audio, text_emb, labels
