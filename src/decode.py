"""Dalle probabilita' alle note, e dalle note al MIDI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import keys

FPS = 60.0


@dataclass(frozen=True)
class Note:
    start_frame: int
    end_frame: int
    key: int
    velocity: int

    @property
    def start(self) -> float:
        return self.start_frame / FPS

    @property
    def end(self) -> float:
        return max(self.end_frame, self.start_frame + 1) / FPS


def peak_pick(prob: np.ndarray, threshold: float = 0.5, min_distance: int = 2) -> list[tuple[int, int, float]]:
    """Picchi locali per ogni tasto: lista di (frame, tasto, probabilita').

    Non basta la soglia: un onset deve essere un *massimo locale* e i picchi
    troppo vicini vengono soppressi (a 60 fps due onset a 1-2 frame di distanza
    sono lo stesso evento fisico, visto il dwell mediano di 99 ms).
    """
    if prob.size == 0:
        return []
    size = 2 * int(min_distance) + 1
    peaks: list[tuple[int, int, float]] = []
    for key_index in range(prob.shape[1]):
        column = prob[:, key_index]
        if column.size == 0 or column.max() <= threshold:
            continue
        local = maximum_filter1d(column, size=size, mode="constant", cval=-1.0)
        hits = np.flatnonzero((column >= local - 1e-6) & (column > threshold))
        keep: list[int] = []
        for t in hits:
            if keep and (t - keep[-1]) < min_distance:
                if column[t] > column[keep[-1]]:
                    keep[-1] = int(t)
            else:
                keep.append(int(t))
        peaks.extend((t, key_index, float(column[t])) for t in keep)
    peaks.sort()
    return peaks


def notes_from_heads(
    heads: dict[str, np.ndarray],
    threshold_onset: float = 0.5,
    threshold_offset: float = 0.5,
    threshold_state: float = 0.5,
    min_distance: int = 2,
    max_length: int = 900,
    min_length: int = 1,
) -> list[Note]:
    """Decodifica le teste in note.

    La fine si prende dalla testa di rilascio (il dito che lascia il tasto, cioe'
    l'unica cosa visibile); se il rilascio non viene rilevato si prosegue finche'
    la testa "tasto tenuto" resta attiva, cosi' le note lunghe non vengono
    troncate a meta'.
    """
    onset = heads["onset"]
    state = heads.get("frame")
    offset = heads.get("offset")
    velocity = heads.get("velocity")
    total = onset.shape[0]

    notes: list[Note] = []
    for start, key_index, _ in peak_pick(onset, threshold_onset, min_distance):
        end = min(total - 1, start + min_length)
        limit = min(total - 1, start + max_length)

        release = -1
        if offset is not None:
            window = offset[start + 1:limit + 1, key_index]
            hits = np.flatnonzero(window > threshold_offset)
            if hits.size:
                release = start + 1 + int(hits[0])

        if release > 0:
            end = max(release, start + min_length)
        elif state is not None:
            cursor = start
            gap = 0
            while cursor <= limit:
                if state[cursor, key_index] > threshold_state:
                    gap = 0
                    end = cursor
                else:
                    gap += 1
                    if gap > 3:                      # tolleranza: qualche frame di incertezza
                        break
                cursor += 1

        value = 80
        if velocity is not None:
            value = int(np.clip(round(float(velocity[start, key_index]) * 127.0), 1, 127))
        notes.append(Note(start_frame=start, end_frame=int(end), key=key_index, velocity=value))

    notes.sort(key=lambda note: (note.start_frame, note.key))
    return notes


def notes_to_events(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    """(frame di inizio, tasto) — il formato con cui si confronta con il ground truth."""
    if not notes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return (
        np.array([n.start_frame for n in notes], dtype=np.int64),
        np.array([n.key for n in notes], dtype=np.int64),
    )


def notes_to_midi(notes: list[Note], path: str | Path, program: int = 0, fps: float = FPS) -> Path:
    """Scrive un file MIDI. La velocity viene dalla testa dedicata (grossolana: dal video non si legge meglio)."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="piano")
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=keys.index_to_note(note.key),
                start=note.start_frame / fps,
                end=max(note.end_frame, note.start_frame + 1) / fps,
            )
        )
    midi.instruments.append(instrument)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path
