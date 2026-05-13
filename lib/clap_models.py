"""Unified interface to LAION-CLAP and Microsoft CLAP.

Each backbone is wrapped in a `CLAPBackbone` class with two methods:

    encode_audio(paths_or_array, sr=48000) -> np.ndarray   shape (N, d)
    encode_text(strings)                  -> np.ndarray   shape (N, d)

Embeddings are returned **without** L2 normalization — most experiments need raw
features so that gap math, normalization, and shift can be applied consistently
through `lib.gap_utils`. Pass `normalize=True` if you want unit-norm vectors.

Both backbones are heavyweight: each loads ≥500 MB. We lazy-import their packages
so that other parts of the codebase (gap math, dataset I/O, viz) work in lean envs.

Usage:
    from lib.clap_models import load_backbone
    laion = load_backbone("laion")
    audio_emb = laion.encode_audio(["clip.wav", ...])   # (N, 512)
    text_emb  = laion.encode_text(["a dog barking", ...])

Both LAION-CLAP and MS-CLAP produce 1024-dim raw projections that we project /
slice down to the published embedding dimensionality (LAION-CLAP: 512, MS-CLAP-2023: 1024).
We don't change the dimensionality — we just expose `.embed_dim` so downstream code
can sanity-check shapes.
"""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

PathOrArray = Union[str, os.PathLike, np.ndarray]


# ---------------------------------------------------------------------------
# Abstract backbone
# ---------------------------------------------------------------------------
@dataclass
class BackboneInfo:
    name: str
    embed_dim: int
    sample_rate: int
    notes: str = ""


class CLAPBackbone(ABC):
    info: BackboneInfo

    @abstractmethod
    def encode_audio(
        self,
        paths: Sequence[PathOrArray],
        batch_size: int = 8,
        normalize: bool = False,
    ) -> np.ndarray: ...

    @abstractmethod
    def encode_text(
        self,
        strings: Sequence[str],
        batch_size: int = 32,
        normalize: bool = False,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# LAION-CLAP wrapper
# ---------------------------------------------------------------------------
class LaionCLAP(CLAPBackbone):
    """LAION-CLAP (Wu et al., ICASSP 2023). HTSAT audio encoder + RoBERTa text encoder.

    Default checkpoint: 630k-audioset-best.pt (general-purpose, no fusion).
    Embedding dim: 512.  Audio sample rate: 48 kHz.
    """

    # Default checkpoint URL on Hugging Face (mirror of laion_clap's bundled URL).
    DEFAULT_CKPT_URL = (
        "https://huggingface.co/lukewys/laion_clap/resolve/main/"
        "630k-audioset-best.pt"
    )

    def __init__(
        self,
        ckpt: Optional[str] = None,
        enable_fusion: bool = False,
        amodel: str = "HTSAT-tiny",
        device: Optional[str] = None,
    ):
        try:
            import laion_clap  # type: ignore
            import torch
        except ImportError as e:
            raise ImportError(
                "LAION-CLAP requires `pip install laion-clap`."
            ) from e

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device

        self._torch = torch
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, amodel=amodel, device=device
        )

        # laion_clap.hook.load_ckpt() uses `wget` → urllib → fails on macOS python.org
        # due to missing system root CAs. Pre-download with `requests` (certifi-backed)
        # to a stable cache location and pass the local path in.
        if ckpt is None:
            ckpt = self._ensure_default_ckpt()
        self.model.load_ckpt(ckpt)
        self.model.eval()

        self.info = BackboneInfo(
            name="laion-clap",
            embed_dim=512,
            sample_rate=48_000,
            notes=f"checkpoint={Path(ckpt).name}, fusion={enable_fusion}",
        )

    def _ensure_default_ckpt(self) -> str:
        """Download `630k-audioset-best.pt` once into ~/.cache/laion_clap/.

        Uses requests (certifi-backed) so the certificate-verify failure that
        affects macOS python.org installs is sidestepped.
        """
        import requests

        cache_dir = Path.home() / ".cache" / "laion_clap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "630k-audioset-best.pt"
        if target.exists() and target.stat().st_size > 100_000_000:
            return str(target)
        print(f"[LaionCLAP] downloading checkpoint → {target}")
        with requests.get(self.DEFAULT_CKPT_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            print(f"  {done/1e6:6.1f} / {total/1e6:6.1f} MB ({pct:5.1f}%)", end="\r")
        print()
        return str(target)

    # --- audio ---------------------------------------------------------
    def encode_audio(
        self,
        paths: Sequence[PathOrArray],
        batch_size: int = 8,
        normalize: bool = False,
    ) -> np.ndarray:
        """Encode a list of audio file paths. Arrays are also supported but are routed
        through `get_audio_embedding_from_data` (expects 48 kHz mono, shape (B, T))."""
        out = []
        # File-path mode
        if all(isinstance(p, (str, os.PathLike)) for p in paths):
            paths = [str(p) for p in paths]
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                emb = self.model.get_audio_embedding_from_filelist(
                    x=chunk, use_tensor=False
                )
                out.append(np.asarray(emb))
        # Array mode (already-loaded waveforms at 48 kHz)
        else:
            arrs = [np.asarray(a, dtype=np.float32) for a in paths]
            for i in range(0, len(arrs), batch_size):
                chunk = arrs[i : i + batch_size]
                # LAION-CLAP expects (B, T) tensor at 48 kHz
                t = self._torch.tensor(np.stack(chunk), device=self.device)
                with self._torch.no_grad():
                    emb = self.model.get_audio_embedding_from_data(
                        x=t, use_tensor=True
                    )
                out.append(emb.detach().cpu().numpy())
        emb = np.concatenate(out, axis=0)
        return _maybe_normalize(emb, normalize)

    # --- text ----------------------------------------------------------
    def encode_text(
        self,
        strings: Sequence[str],
        batch_size: int = 32,
        normalize: bool = False,
    ) -> np.ndarray:
        out = []
        for i in range(0, len(strings), batch_size):
            chunk = list(strings[i : i + batch_size])
            emb = self.model.get_text_embedding(chunk, use_tensor=False)
            out.append(np.asarray(emb))
        return _maybe_normalize(np.concatenate(out, axis=0), normalize)


# ---------------------------------------------------------------------------
# Microsoft CLAP wrapper
# ---------------------------------------------------------------------------
class MicrosoftCLAP(CLAPBackbone):
    """Microsoft CLAP (Elizalde et al., 2023). HTSAT + GPT-2 / RoBERTa text head.

    Default version: 2023.  Embedding dim: 1024.  Audio sample rate: 44.1 kHz.
    """

    def __init__(self, version: str = "2023", use_cuda: Optional[bool] = None):
        try:
            from msclap import CLAP  # type: ignore
            import torch
        except ImportError as e:
            raise ImportError(
                "Microsoft CLAP requires `pip install msclap`."
            ) from e

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self._torch = torch
        self.model = CLAP(version=version, use_cuda=use_cuda)

        self.info = BackboneInfo(
            name=f"msclap-{version}",
            embed_dim=1024,
            sample_rate=44_100,
            notes=f"version={version}",
        )

    def encode_audio(
        self,
        paths: Sequence[PathOrArray],
        batch_size: int = 8,
        normalize: bool = False,
    ) -> np.ndarray:
        # MS-CLAP exposes only path-based inference; convert arrays via tempfile if needed.
        if not all(isinstance(p, (str, os.PathLike)) for p in paths):
            raise NotImplementedError(
                "MS-CLAP wrapper currently expects file paths; pre-write arrays to disk."
            )
        paths = [str(p) for p in paths]
        out = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            emb = self.model.get_audio_embeddings(chunk)
            # msclap returns a torch.Tensor
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            out.append(np.asarray(emb))
        return _maybe_normalize(np.concatenate(out, axis=0), normalize)

    def encode_text(
        self,
        strings: Sequence[str],
        batch_size: int = 32,
        normalize: bool = False,
    ) -> np.ndarray:
        out = []
        for i in range(0, len(strings), batch_size):
            chunk = list(strings[i : i + batch_size])
            emb = self.model.get_text_embeddings(chunk)
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            out.append(np.asarray(emb))
        return _maybe_normalize(np.concatenate(out, axis=0), normalize)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKBONES = {
    "laion": LaionCLAP,
    "laion-clap": LaionCLAP,
    "msclap": MicrosoftCLAP,
    "ms": MicrosoftCLAP,
    "ms-clap": MicrosoftCLAP,
}


def load_backbone(name: str, **kwargs) -> CLAPBackbone:
    """Factory: load a CLAP backbone by short name.

    Available names: 'laion', 'msclap'.
    """
    name = name.lower().strip()
    if name not in _BACKBONES:
        raise KeyError(
            f"Unknown CLAP backbone {name!r}. Available: {sorted(set(_BACKBONES))}"
        )
    return _BACKBONES[name](**kwargs)


def _maybe_normalize(x: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return x
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Smoke test (only runs if both packages are installed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Probing available CLAP backbones...")
    for short in ("laion", "msclap"):
        try:
            bb = load_backbone(short)
            te = bb.encode_text(["a dog barking", "piano music"])
            print(f"  {short:10s} OK | info={bb.info} | text_emb shape={te.shape}")
        except Exception as e:  # pragma: no cover
            print(f"  {short:10s} unavailable: {type(e).__name__}: {e}")
    print("Done.")
    sys.exit(0)
