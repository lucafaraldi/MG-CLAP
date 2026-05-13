"""Modality-gap math — exactly the operations from Mind the Gap (Liang et al., 2022).

All functions accept either NumPy arrays or torch.Tensors. Most experiments operate on
already-extracted embeddings, so NumPy is the default. The InfoNCE helper sticks with
torch so we can sweep gradients later if we want.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # numpy-only fallback for very lean envs
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization. Operates in-place safety: returns a new array."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


# ---------------------------------------------------------------------------
# Modality centroids and gap (Sec. 4.2 of the paper)
# ---------------------------------------------------------------------------
def modality_centroids(
    audio_emb: np.ndarray, text_emb: np.ndarray, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of L2-normalized embeddings per modality.

    Following the paper: embeddings are L2-normalized BEFORE averaging
    (otherwise the gap measurement is dominated by raw-norm differences).
    """
    if normalize:
        audio_emb = l2_normalize(audio_emb)
        text_emb = l2_normalize(text_emb)
    return audio_emb.mean(axis=0), text_emb.mean(axis=0)


def gap_vector(
    audio_emb: np.ndarray, text_emb: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Δ = mean(audio_normed) - mean(text_normed).

    Note the sign convention: positive λ in `shift_features` moves audio TOWARDS
    text (closes the gap), matching the paper's Eq. for x_shift_i = norm(x_i - λ Δ).
    """
    a_c, t_c = modality_centroids(audio_emb, text_emb, normalize=normalize)
    return a_c - t_c


def gap_distance(
    audio_emb: np.ndarray, text_emb: np.ndarray, normalize: bool = True
) -> float:
    """‖Δ‖₂ — the modality gap distance. CLIP on COCO ≈ 0.82."""
    return float(np.linalg.norm(gap_vector(audio_emb, text_emb, normalize=normalize)))


# ---------------------------------------------------------------------------
# Shift intervention (Eq. in Sec. 4.2 / Table 1)
# ---------------------------------------------------------------------------
def shift_features(
    audio_emb: np.ndarray,
    text_emb: np.ndarray,
    lam: float,
    delta: np.ndarray | None = None,
    renormalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric gap shift used everywhere in Mind the Gap.

    x'_i = normalize(x_i - λ Δ)         (audio)
    y'_i = normalize(y_i + λ Δ)         (text)

    `delta` defaults to the gap vector computed from the inputs. Pass an explicit
    `delta` (e.g. computed once on AudioCaps and re-used on ESC-50) when shifting
    embeddings outside the population that defined the gap, as the paper does for
    Table 1 (gap computed on COCO, applied on EuroSAT/CIFAR/etc.).
    """
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    text_emb = np.asarray(text_emb, dtype=np.float64)
    if delta is None:
        delta = gap_vector(audio_emb, text_emb, normalize=True)
    audio_shift = audio_emb - lam * delta
    text_shift = text_emb + lam * delta
    if renormalize:
        audio_shift = l2_normalize(audio_shift)
        text_shift = l2_normalize(text_shift)
    return audio_shift, text_shift


# ---------------------------------------------------------------------------
# Cosine similarity / cone-effect helpers (Sec. 3, Fig. 2)
# ---------------------------------------------------------------------------
def cosine_sim_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine similarity. If `b` is None, computes a vs. a."""
    a = l2_normalize(a)
    b = a if b is None else l2_normalize(b)
    return a @ b.T


def intra_modal_avg_cosine(
    emb: np.ndarray, exclude_diag: bool = True
) -> float:
    """Average pairwise cosine similarity within a modality.

    This is the cone-effect statistic from Fig 2 — the higher this is, the
    narrower the cone the encoder maps everything into.
    """
    sim = cosine_sim_matrix(emb)
    if exclude_diag:
        n = sim.shape[0]
        sim = sim.copy()
        np.fill_diagonal(sim, np.nan)
        return float(np.nanmean(sim))
    return float(sim.mean())


# ---------------------------------------------------------------------------
# Symmetric InfoNCE — used by Figure 3's loss-landscape sweep
# ---------------------------------------------------------------------------
def info_nce_loss(
    audio_emb,
    text_emb,
    temperature: float = 0.01,
    return_components: bool = False,
):
    """Symmetric NT-Xent / InfoNCE loss as used by CLIP and CLAP.

    audio_emb, text_emb: (N, d) tensors or arrays. They are L2-normalized inside.
    temperature: τ. CLIP uses ~0.01; LAION-CLAP also learns it.

    Loss = ½ (ℓ_{a→t} + ℓ_{t→a}).
    """
    if _HAS_TORCH and isinstance(audio_emb, torch.Tensor):
        a = F.normalize(audio_emb, dim=-1)
        t = F.normalize(text_emb, dim=-1)
        logits_at = (a @ t.T) / temperature
        logits_ta = logits_at.T
        n = a.shape[0]
        target = torch.arange(n, device=a.device)
        l_at = F.cross_entropy(logits_at, target)
        l_ta = F.cross_entropy(logits_ta, target)
        loss = 0.5 * (l_at + l_ta)
        if return_components:
            return loss, l_at.item(), l_ta.item()
        return loss

    # NumPy path
    a = l2_normalize(audio_emb)
    t = l2_normalize(text_emb)
    logits_at = (a @ t.T) / temperature
    n = a.shape[0]

    def _xent(logits: np.ndarray) -> float:
        # log-sum-exp trick
        m = logits.max(axis=1, keepdims=True)
        lse = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))
        diag = np.diag(logits)
        return float((-diag + lse).mean())

    l_at = _xent(logits_at)
    l_ta = _xent(logits_at.T)
    loss = 0.5 * (l_at + l_ta)
    if return_components:
        return loss, l_at, l_ta
    return loss


# ---------------------------------------------------------------------------
# Sanity / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 256, 64
    # Two clearly separated modalities on the unit sphere.
    audio = l2_normalize(rng.normal(size=(n, d)) + np.array([1.0] + [0.0] * (d - 1)))
    text = l2_normalize(rng.normal(size=(n, d)) - np.array([1.0] + [0.0] * (d - 1)))

    print(f"gap distance: {gap_distance(audio, text):.4f}")
    print(f"intra-audio avg cos: {intra_modal_avg_cosine(audio):.4f}")
    print(f"intra-text  avg cos: {intra_modal_avg_cosine(text):.4f}")

    # Each modality moves by λ·Δ in opposite directions, so the gap is multiplied
    # by approximately (1 − 2λ) before renormalization.
    a_s, t_s = shift_features(audio, text, lam=0.5)
    print(f"gap @ λ=0.5  : {gap_distance(a_s, t_s):.4f}  (expected ≈ 0)")
    a_s, t_s = shift_features(audio, text, lam=1.0)
    print(f"gap @ λ=1.0  : {gap_distance(a_s, t_s):.4f}  (modalities swap; ≈ |Δ|)")
    a_s, t_s = shift_features(audio, text, lam=-0.5)
    print(f"gap @ λ=-0.5 : {gap_distance(a_s, t_s):.4f}  (gap widened ~2×)")

    print(f"InfoNCE @ τ=0.07: {info_nce_loss(audio, text, temperature=0.07):.4f}")
    print(f"InfoNCE @ τ=0.01: {info_nce_loss(audio, text, temperature=0.01):.4f}")
