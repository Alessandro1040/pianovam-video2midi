"""Dataset e aumentazioni.

Le due cose che rendono il modello robusto a "come e' messo il pianoforte":

* l'invarianza geometrica e' gia' risolta dal warp (src/geometry.py);
* restano da coprire gli errori di calibrazione e lo specchio sinistra/destra.
  Il flip orizzontale della striscia **inverte anche le etichette dei tasti**
  (tasto i <-> tasto 87-i): e' un'aumentazione esatta, perche' il pattern dei
  tasti neri esiste in entrambe le versioni, e insegna davvero la simmetria
  invece di far memorizzare un lato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from . import keys, targets, video

N_HEADS = 4


@dataclass
class AugmentConfig:
    """Aumentazioni attive in training."""

    flip_prob: float = 0.5
    brightness: float = 0.20          # +/- sul guadagno e sull'offset
    contrast: float = 0.20
    gamma: float = 0.25
    noise_std: float = 0.02
    blur_prob: float = 0.25
    shift_frames: int = 2             # jitter temporale: insegna la tolleranza di 50 ms
    extra_margin: int = 1             # dilatazione onset nei bersagli


@dataclass
class DatasetConfig:
    """Configurazione del dataset di training."""

    cache_root: str = "data/cache"
    raw_root: str = "data/raw"
    clip_frames: int = 64
    samples_per_epoch: int = 750
    temporal_radius: int = 15         # frame di contesto per la "linea di base" (0 = disattivato)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    enabled: bool = True

    @property
    def in_channels(self) -> int:
        return 2 if self.temporal_radius > 0 else 1

    def read_frames(self, augment: bool) -> int:
        """Quanti frame servono dalla cache per ottenere `clip_frames` di input."""
        shift = self.augment.shift_frames if augment else 0
        return self.clip_frames + 2 * shift + 1


def _normalise(strip: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in scala ~[-1, 1]."""
    return strip.astype(np.float32) / 127.5 - 1.0


CONTRAST_SCALE = 32.0      # quanti livelli di grigio vale "1" nel canale di differenza


def temporal_features(window: np.ndarray, radius: int = 15) -> np.ndarray:
    """(T, H, W) uint8 -> (T, C, H, W) float32.

    Canale 0: il frame normalizzato (com'e' adesso).
    Canale 1: **la differenza dalla linea di base** del tasto, cioe' il frame meno
    la media mobile centrata su +-`radius` frame.

    Il secondo canale e' il motivo per cui il modello puo' funzionare su un
    pianoforte e una stanza mai visti: non guarda "quanto e' chiaro questo tasto"
    (che cambia da pianoforte a pianoforte, da luce a luce), ma "quanto questo
    tasto e' diverso da come era un attimo prima", che e' esattamente cio' che
    succede quando viene premuto. E' la stessa idea del rilevamento del movimento
    nei video di sorveglianza, applicata alla tastiera.
    """
    if radius <= 0:
        return _normalise(window)[:, None]

    gray = window.astype(np.float32)
    padded = np.pad(gray, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float32)
    cumulative = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
    size = 2 * radius + 1
    baseline = (cumulative[size:] - cumulative[:-size]) / size
    difference = gray - baseline
    return np.stack([_normalise(window), np.clip(difference / CONTRAST_SCALE, -1.0, 1.0)], axis=1)


def augment_strip(
    strips: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Aumentazioni fotometriche su uint8 (luminosita', contrasto, gamma, blur, rumore).

    Si applicano ai frame *grezzi*, prima del calcolo del canale di differenza:
    cosi' anche quest'ultimo vede la stessa variazione di esposizione e il modello
    non puo' usare la luminosita' assoluta come scorciatoia.
    """
    out = strips.astype(np.float32).copy()
    if config.brightness > 0:
        out = out * (1.0 + rng.uniform(-config.brightness, config.brightness))
        out = out + rng.uniform(-1, 1) * config.brightness * 120.0
    if config.contrast > 0:
        mean = out.mean(axis=(1, 2), keepdims=True)
        out = (out - mean) * (1.0 + rng.uniform(-config.contrast, config.contrast)) + mean
    if config.gamma > 0:
        gamma = float(np.exp(rng.uniform(-config.gamma, config.gamma)))
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    if config.blur_prob > 0 and rng.random() < config.blur_prob:
        sigma = float(rng.uniform(0.4, 1.4))
        out = np.stack([gaussian_filter(f, sigma=sigma) for f in out])
    if config.noise_std > 0:
        out = out + rng.normal(0.0, config.noise_std * 255.0, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_example(
    window: np.ndarray,
    events: targets.Events,
    target_start: int,
    config: DatasetConfig,
    rng: np.random.Generator,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Da una finestra grezza a (input normalizzato, bersagli densi, frame iniziale).

    Sta qui e non nel Dataset perche' la usano sia il Dataset sia il campionatore
    dei batch: un solo posto dove sono definite le aumentazioni.
    """
    clip = config.clip_frames
    shift = config.augment.shift_frames if augment else 0
    available = 2 * shift + 1
    if window.shape[0] < clip + available - 1:
        raise ValueError(f"finestra troppo corta: {window.shape[0]} < {clip + available - 1}")

    offset = int(rng.integers(0, available)) if shift else 0
    start = target_start + offset

    flipped = bool(rng.random() < config.augment.flip_prob) if augment else False
    raw = window[:, :, ::-1].copy() if flipped else window
    if augment:
        raw = augment_strip(raw, rng, config.augment)

    features = temporal_features(raw, config.temporal_radius)[offset:offset + clip]
    dense = targets.dense_targets(
        events, start, clip, onset_dilate=config.augment.extra_margin if augment else 0
    )
    if flipped:
        dense = dense[:, :, ::-1].copy()

    return np.ascontiguousarray(features), np.ascontiguousarray(dense), start


class ClipDataset(Dataset):
    """Finestre temporali estratte a caso dalla cache su disco."""

    def __init__(self, records: list[str], config: DatasetConfig, seed: int = 0) -> None:
        self.records = list(records)
        self.config = config
        self.seed = seed
        self.cache_root = Path(config.cache_root)
        self.raw_root = Path(config.raw_root)
        self.events: dict[str, targets.Events] = {}
        self.lengths: dict[str, int] = {}
        for record in self.records:
            cache_dir = self.cache_root / record
            meta = video.read_meta(cache_dir)
            self.lengths[record] = int(meta["n_frames"])
            self.events[record] = video.load_events(
                self.raw_root / "TSV" / f"{record}.tsv", int(meta["n_frames"])
            )

    @property
    def total_frames(self) -> int:
        return sum(self.lengths.values())

    def __len__(self) -> int:
        return self.config.samples_per_epoch

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng((self.seed, index))
        record = self.records[int(rng.integers(len(self.records)))]
        return self.sample(record, rng)

    def sample(self, record: str, rng: np.random.Generator | None = None, augment: bool | None = None) -> dict:
        """Una finestra dalla registrazione indicata (usata anche per la validazione)."""
        rng = rng or np.random.default_rng(0)
        augment = self.config.enabled if augment is None else augment
        clip = self.config.clip_frames
        shift = self.config.augment.shift_frames if augment else 0
        need = clip + 2 * shift + 1

        total = self.lengths[record]
        start = int(rng.integers(0, max(1, total - need)))
        window = video.read_strips(self.cache_root / record, start, need)
        if window.shape[0] < need:
            raise RuntimeError(f"finestra fuori dalla cache per {record} ({start})")

        strips, dense, target_start = make_example(
            window, self.events[record], start, self.config, rng, augment
        )
        return {
            "strips": torch.from_numpy(strips),
            "targets": torch.from_numpy(dense),
            "record": record,
            "start_frame": target_start,
        }

