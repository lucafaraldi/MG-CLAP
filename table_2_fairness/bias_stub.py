#!/usr/bin/env python3
"""Stub for an audio-domain bias study analogous to the CLIP×FairFace experiment.

This file is INTENTIONALLY not wired into the main run pipeline. It exists to
sketch the API for an Option-A speaker-demographic-bias study (see README.md
in this folder). Fill in the dataset loader and prompt set, then re-use the
shift utilities from `lib.gap_utils`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.gap_utils import (  # noqa: E402
    gap_vector, l2_normalize, shift_features,
)


# ---------------------------------------------------------------------------
# Step 1 — load demographic-tagged audio embeddings + prompt embeddings.
# Replace the stub returns below with a real Common Voice / VoxCeleb loader.
# ---------------------------------------------------------------------------
def load_demographic_audio_embeddings(backbone: str = "laion"):
    raise NotImplementedError(
        "Plug in your demographic-tagged audio embedding cache here.\n"
        "Expected return: (audio_emb [N, d], group_labels [N])."
    )


def load_bias_prompts(backbone: str = "laion"):
    """Two-track prompt set:
    * `neutral_prompts` — innocuous category labels ("a person speaking", ...)
    * `distractor_prompts` — stereotyped / denigrating prompts whose argmax we
       want to be RARE for any demographic group.

    Returns (text_emb [P, d], prompt_kind [P]) where prompt_kind ∈ {neutral, distractor}.
    """
    raise NotImplementedError(
        "Define your prompt set here. Each prompt should be paired with a "
        "kind in {'neutral', 'distractor'}."
    )


# ---------------------------------------------------------------------------
# Step 2 — measure denigration rate per group, before and after shift.
# ---------------------------------------------------------------------------
def denigration_rate(
    audio_emb: np.ndarray,
    text_emb: np.ndarray,
    group_labels: np.ndarray,
    prompt_kind: np.ndarray,
) -> dict[str, float]:
    """Fraction of audio clips whose top-1 prompt is a distractor, per group."""
    sim = l2_normalize(audio_emb) @ l2_normalize(text_emb).T
    pred = sim.argmax(axis=1)
    is_distractor = (prompt_kind == "distractor")[pred]
    out: dict[str, float] = {}
    for g in np.unique(group_labels):
        m = group_labels == g
        out[str(g)] = float(is_distractor[m].mean())
    return out


# ---------------------------------------------------------------------------
# Step 3 — sweep λ (computed on AudioCaps, applied here) and report curves.
# ---------------------------------------------------------------------------
def main():
    print("Stub: see README.md in this folder.")
    # Pseudo-code:
    #   audio_ac, text_ac = load_audiocaps(backbone)
    #   delta = gap_vector(audio_ac, text_ac)
    #   audio_demo, group = load_demographic_audio_embeddings()
    #   text_prompts, kind = load_bias_prompts()
    #   for lam in np.linspace(-1, 1, 21):
    #       a_s, t_s = shift_features(audio_demo, text_prompts, lam, delta=delta)
    #       rates = denigration_rate(a_s, t_s, group, kind)
    #       print(lam, rates)


if __name__ == "__main__":
    main()
