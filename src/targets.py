"""Dal TSV di PianoVAM ai bersagli che il modello deve predire.

Cosa predice il modello, per ogni frame e per ognuno degli 88 tasti:

* `onset`  : il tasto scende (inizio della nota)  <- colonna `onset`
* `frame`  : il tasto e' tenuto premuto          <- da `onset` a `key_offset`
* `offset` : il dito lascia fisicamente il tasto <- colonna `key_offset`
* `velocity`: forza stimata (solo nel frame di onset)

Nota importante sul dataset: si usa di proposito `key_offset` e **non**
`frame_offset`. Il secondo include il pedale di risonanza, che in un'inquadratura
dall'alto sui tasti *non e' visibile*: addestrarlo significherebbe chiedere al
modello di predire qualcosa che non puo' vedere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import keys

VIDEO_FPS = 60.0


@dataclass(frozen=True)
class Events:
    """Eventi di nota allineati ai frame del video."""

    onset_frame: np.ndarray   # (N,) int32
    off_frame: np.ndarray     # (N,) int32
    key: np.ndarray           # (N,) int8   indice di colonna 0..87
    velocity: np.ndarray      # (N,) uint8  1..127

    def __len__(self) -> int:
        return int(self.onset_frame.size)

    def subset(self, lo: int, hi: int) -> "Events":
        """Eventi il cui onset cade in [lo, hi)."""
        sel = (self.onset_frame >= lo) & (self.onset_frame < hi)
        return Events(
            self.onset_frame[sel],
            self.off_frame[sel],
            self.key[sel],
            self.velocity[sel],
        )


def read_tsv(path: str | Path) -> Events:
    """Legge un TSV di PianoVAM (l'header e' commentato con `#`)."""
    raw = np.loadtxt(path, dtype=np.float64, comments="#", delimiter="\t", ndmin=2)
    if raw.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return Events(empty, empty, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.uint8))

    onset = np.rint(raw[:, 0] * VIDEO_FPS).astype(np.int32)
    off = np.rint(raw[:, 1] * VIDEO_FPS).astype(np.int32)
    key = np.rint(raw[:, 3]).astype(np.int32) - keys.MIDI_MIN
    vel = np.clip(np.rint(raw[:, 4]), 1, 127).astype(np.uint8)

    valid = (key >= 0) & (key < keys.N_KEYS) & (off >= onset)
    return Events(onset[valid], np.maximum(off[valid], onset[valid]), key[valid].astype(np.int8), vel[valid])


def dense_targets(events: Events, start: int, n_frames: int, onset_dilate: int = 1) -> np.ndarray:
    """Bersagli densi per una finestra temporale: (4, n_frames, 88) float32.

    L'onset viene dilatato di +/- `onset_dilate` frame: se il modello sbaglia di
    un frame non viene punito, perche' a 60 fps un frame sono 16.7 ms e la
    metrica di riferimento tollera 50 ms.
    """
    out = np.zeros((4, n_frames, keys.N_KEYS), dtype=np.float32)
    stop = start + n_frames

    onsets = events.onset_frame
    keep = (onsets >= start - onset_dilate) & (onsets < stop + onset_dilate)
    for t, k, v in zip(onsets[keep], events.key[keep], events.velocity[keep]):
        lo = max(start, int(t) - onset_dilate)
        hi = min(stop, int(t) + onset_dilate + 1)
        out[0, lo - start:hi - start, int(k)] = 1.0
        # la velocita' si predice solo nel frame esatto dell'onset
        idx = int(t) - start
        if 0 <= idx < n_frames:
            out[3, idx, int(k)] = float(v) / 127.0

    offs = events.off_frame
    keep_off = (offs >= start) & (offs < stop)
    for t, k in zip(offs[keep_off], events.key[keep_off]):
        out[2, int(t) - start, int(k)] = 1.0

    # stato del tasto: 1 tra onset e off (incluso l'offset, escluso l'onset,
    # perche' il frame di onset e' gia' coperto dalla testa dedicata)
    for lo, hi, k in zip(onsets[keep], offs[keep], events.key[keep]):
        a = max(start, int(lo) + 1)
        b = min(stop, int(hi) + 1)
        if b > a:
            out[1, a - start:b - start, int(k)] = 1.0

    return out


def events_to_midi_notes(events: Events, fps: float = VIDEO_FPS) -> list[tuple[float, float, int, int]]:
    """(start_s, end_s, note, velocity) — comodo per i test e per il confronto con i MIDI del dataset."""
    return [
        (float(o) / fps, float(f) / fps, keys.index_to_note(int(k)), int(v))
        for o, f, k, v in zip(events.onset_frame, events.off_frame, events.key, events.velocity)
    ]
