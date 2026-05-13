"""Shared library for the Mind-the-Gap CLAP port."""
from .gap_utils import (
    l2_normalize,
    modality_centroids,
    gap_vector,
    gap_distance,
    shift_features,
    cosine_sim_matrix,
    intra_modal_avg_cosine,
    info_nce_loss,
)

__all__ = [
    "l2_normalize",
    "modality_centroids",
    "gap_vector",
    "gap_distance",
    "shift_features",
    "cosine_sim_matrix",
    "intra_modal_avg_cosine",
    "info_nce_loss",
]
