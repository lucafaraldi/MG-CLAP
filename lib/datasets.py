"""Dataset loaders for AudioCaps and ESC-50.

We cover two roles, mirroring the original CLIP paper experiments:

* **AudioCaps** (audio + free-form caption pairs) → role of MS-COCO.
  Used in: Fig 1 (gap visualization), Fig 2 (real-data cone effect),
  Fig 3 (gap stats), and as the population that defines Δ for Table 1.

* **ESC-50** (50 environmental-sound classes, 5 cross-validation folds)
  → role of CIFAR / EuroSAT.  Used in: Table 1 (zero-shot under shift).

Both loaders return lightweight `dataclass` objects that can be iterated or
sliced. The actual audio is loaded lazily.

If you've pre-downloaded these datasets, set the env vars
`AUDIOCAPS_ROOT` / `ESC50_ROOT` to skip auto-discovery.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# AudioCaps
# ---------------------------------------------------------------------------
@dataclass
class AudioCapsItem:
    audiocap_id: str
    youtube_id: str
    start_time: float
    caption: str
    audio_path: Path  # local path on disk

    def exists(self) -> bool:
        return self.audio_path.exists()


class AudioCaps:
    """AudioCaps loader.

    Layout expected on disk (created by `scripts/00_setup_data.py`):

        $AUDIOCAPS_ROOT/
            csv/{train,val,test}.csv     # original AudioCaps CSVs
            audio/{train,val,test}/<youtube_id>_<start>.wav

    Audio is 10s clips at the original sample rate; we resample on demand.
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: Optional[os.PathLike] = None,
        split: str = "val",
        require_exists: bool = True,
    ):
        if root is None:
            root = os.environ.get("AUDIOCAPS_ROOT", "data/audiocaps")
        self.root = Path(root)
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}")
        self.split = split

        csv_path = self.root / "csv" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"AudioCaps CSV not found: {csv_path}. Run "
                f"`python scripts/00_setup_data.py --audiocaps`."
            )

        self.items: List[AudioCapsItem] = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yid = row.get("youtube_id") or row.get("ytid") or row.get("youtubeId")
                start = float(row.get("start_time", 0.0))
                cap = row.get("caption", "").strip()
                acid = row.get("audiocap_id") or row.get("id") or yid
                wav = self.root / "audio" / split / f"{yid}_{int(start)}.wav"
                item = AudioCapsItem(
                    audiocap_id=acid,
                    youtube_id=yid,
                    start_time=start,
                    caption=cap,
                    audio_path=wav,
                )
                if require_exists and not item.exists():
                    continue
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[AudioCapsItem]:
        return iter(self.items)

    def __getitem__(self, idx: int) -> AudioCapsItem:
        return self.items[idx]

    def sample(self, n: int, seed: int = 0) -> List[AudioCapsItem]:
        rng = np.random.default_rng(seed)
        if n >= len(self):
            return list(self.items)
        idx = rng.choice(len(self), size=n, replace=False)
        return [self.items[int(i)] for i in idx]

    def captions(self) -> List[str]:
        return [it.caption for it in self.items]

    def audio_paths(self) -> List[Path]:
        return [it.audio_path for it in self.items]


# ---------------------------------------------------------------------------
# ESC-50
# ---------------------------------------------------------------------------
@dataclass
class ESC50Item:
    filename: str
    fold: int
    target: int
    category: str
    esc10: bool
    audio_path: Path


class ESC50:
    """ESC-50 loader (Piczak, 2015). 2,000 environmental-sound clips, 50 classes,
    5 cross-validation folds.

    Layout expected:
        $ESC50_ROOT/
            meta/esc50.csv
            audio/<filename>.wav

    Built-in 5-fold split: pre-defined in the meta CSV.
    """

    def __init__(self, root: Optional[os.PathLike] = None):
        if root is None:
            root = os.environ.get("ESC50_ROOT", "data/esc50")
        self.root = Path(root)
        meta = self.root / "meta" / "esc50.csv"
        if not meta.exists():
            raise FileNotFoundError(
                f"ESC-50 meta not found: {meta}. Run "
                f"`python scripts/00_setup_data.py --esc50`."
            )

        self.items: List[ESC50Item] = []
        with meta.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["filename"]
                self.items.append(
                    ESC50Item(
                        filename=fn,
                        fold=int(row["fold"]),
                        target=int(row["target"]),
                        category=row["category"].replace("_", " "),
                        esc10=row.get("esc10", "False") in ("True", "true", "1"),
                        audio_path=self.root / "audio" / fn,
                    )
                )

        # Class names sorted by target id (stable order matches ESC-50 convention).
        seen = {}
        for it in self.items:
            seen[it.target] = it.category
        self.class_names: List[str] = [seen[i] for i in sorted(seen)]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ESC50Item]:
        return iter(self.items)

    def __getitem__(self, idx: int) -> ESC50Item:
        return self.items[idx]

    def fold_split(self, fold: int) -> Tuple[List[ESC50Item], List[ESC50Item]]:
        """Return (train, eval) item lists for a given test fold (1..5)."""
        if fold not in (1, 2, 3, 4, 5):
            raise ValueError("fold must be in 1..5")
        train = [it for it in self.items if it.fold != fold]
        evalu = [it for it in self.items if it.fold == fold]
        return train, evalu

    def class_prompts(self, template: str = "this is the sound of {}") -> List[str]:
        """Generate one prompt per class. Template can include `{}` for the class name."""
        return [template.format(c) for c in self.class_names]


# ---------------------------------------------------------------------------
# Audio I/O helper
# ---------------------------------------------------------------------------
def load_audio(
    path: os.PathLike, target_sr: int = 48_000, mono: bool = True
) -> np.ndarray:
    """Load audio at target sample rate, return float32 mono."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1 and mono:
        data = data.mean(axis=-1)
    if sr != target_sr:
        try:
            import resampy
            data = resampy.resample(data, sr, target_sr)
        except ImportError:
            import librosa
            data = librosa.resample(y=data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for cls, kwargs in [(AudioCaps, dict(split="val")), (ESC50, {})]:
        try:
            d = cls(**kwargs)
            print(f"{cls.__name__}: {len(d)} items")
            if len(d):
                print(f"  example: {d[0]}")
        except FileNotFoundError as e:
            print(f"{cls.__name__}: not yet downloaded ({e})")
