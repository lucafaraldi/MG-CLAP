"""Dual-encoder dataset for the lightweight CLAP training experiment.

Counterpart of the original repo's `Table_1.../training/datasets.py`. We work
on PRE-EXTRACTED CLAP embeddings rather than raw audio + tokenized text — the
backbones stay frozen and we only train tiny projection heads on top. This is
exactly enough to reproduce the paper's finding that temperature τ controls
the optimal gap distance, without the multi-day cost of training CLAP from
scratch.

Use:

    from table_1_zero_shot.training.datasets import EmbeddingPairs
    ds = EmbeddingPairs.from_cache("embeddings/laion/audiocaps__val.npz")
    loader = ds.dataloader(batch_size=64)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingPairs(Dataset):
    """Wraps a pre-extracted .npz with `audio` and `text` arrays as a torch Dataset.

    Returns tuples of (audio_emb, text_emb) — both float32 torch tensors. Not
    normalized; downstream code should L2-normalize after the projection head.
    """

    def __init__(self, audio: np.ndarray, text: np.ndarray):
        assert audio.shape[0] == text.shape[0], "audio and text must align pair-wise"
        self.audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        self.text = torch.from_numpy(np.asarray(text, dtype=np.float32))

    @classmethod
    def from_cache(cls, path: str | Path) -> "EmbeddingPairs":
        data = np.load(str(path), allow_pickle=True)
        return cls(audio=data["audio"], text=data["text"])

    def __len__(self) -> int:
        return self.audio.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.audio[idx], self.text[idx]

    def dataloader(
        self, batch_size: int = 64, shuffle: bool = True, drop_last: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
