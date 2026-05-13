"""Visualization helpers shared across all experiments.

Two-panel UMAP/PCA plots match the figure style of the original Mind-the-Gap repo:
audio embeddings in one color, text embeddings in another, with optional
modality-centroid markers.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Tab10 from Mind-the-Gap palette
AUDIO_COLOR = "#3b82f6"   # blue
TEXT_COLOR = "#ef4444"    # red
GRID_COLOR = "#e5e7eb"


def pca_2d(*embeddings: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Joint PCA of any number of embedding sets (concatenated, then split back)."""
    from sklearn.decomposition import PCA

    sizes = [e.shape[0] for e in embeddings]
    stacked = np.concatenate(embeddings, axis=0)
    proj = PCA(n_components=2, random_state=0).fit_transform(stacked)
    out = []
    s = 0
    for n in sizes:
        out.append(proj[s : s + n])
        s += n
    return tuple(out)


def umap_2d(
    *embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 0,
) -> Tuple[np.ndarray, ...]:
    """Joint UMAP of any number of embedding sets."""
    import umap  # type: ignore

    sizes = [e.shape[0] for e in embeddings]
    stacked = np.concatenate(embeddings, axis=0)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    proj = reducer.fit_transform(stacked)
    out = []
    s = 0
    for n in sizes:
        out.append(proj[s : s + n])
        s += n
    return tuple(out)


def plot_modality_2d(
    audio_xy: np.ndarray,
    text_xy: np.ndarray,
    title: str = "",
    *,
    audio_label: str = "audio",
    text_label: str = "text",
    show_centroids: bool = True,
    show_pairs: int = 0,
    pairs: Optional[Sequence[int]] = None,
    ax=None,
):
    """Standard 2D scatter: audio vs. text embeddings, optional centroids and pair-lines.

    `pairs`: iterable of indices i — for each i, draws a thin grey line between
    audio_xy[i] and text_xy[i]. Useful for showing aligned pairs.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        audio_xy[:, 0],
        audio_xy[:, 1],
        s=8,
        alpha=0.6,
        c=AUDIO_COLOR,
        label=audio_label,
        edgecolor="none",
    )
    ax.scatter(
        text_xy[:, 0],
        text_xy[:, 1],
        s=8,
        alpha=0.6,
        c=TEXT_COLOR,
        label=text_label,
        edgecolor="none",
    )

    if pairs is None and show_pairs > 0:
        rng = np.random.default_rng(0)
        n = min(audio_xy.shape[0], text_xy.shape[0])
        pairs = rng.choice(n, size=min(show_pairs, n), replace=False)
    if pairs is not None:
        for i in pairs:
            ax.plot(
                [audio_xy[i, 0], text_xy[i, 0]],
                [audio_xy[i, 1], text_xy[i, 1]],
                color="#9ca3af",
                lw=0.4,
                alpha=0.4,
            )

    if show_centroids:
        a_c = audio_xy.mean(axis=0)
        t_c = text_xy.mean(axis=0)
        ax.scatter(*a_c, s=200, marker="X", c=AUDIO_COLOR, edgecolor="black", linewidth=1.2, zorder=5)
        ax.scatter(*t_c, s=200, marker="X", c=TEXT_COLOR, edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID_COLOR)
    return ax


def plot_cosine_hist(
    sims: np.ndarray,
    title: str = "",
    *,
    bins: int = 60,
    color: str = AUDIO_COLOR,
    avg_line: bool = True,
    ax=None,
):
    """Histogram of pairwise cosine similarities — used for the cone-effect figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    flat = sims.ravel()
    flat = flat[~np.isnan(flat)]
    ax.hist(flat, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(title)
    if avg_line:
        m = float(np.nanmean(flat))
        ax.axvline(m, color="black", lw=1.0, linestyle="--", label=f"avg={m:.3f}")
        ax.legend(frameon=False)
    return ax
