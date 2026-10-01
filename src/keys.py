"""Geometria dei tasti: layout 88 tasti, tasti neri, rilevamento dell'orientamento.

Il modello non vede mai la tastiera "grezza": la vede sempre rettificata in una
striscia canonica di 88 colonne (una per tasto). Questo file definisce quella
griglia e le proprieta' invarianti che ne derivano.
"""

from __future__ import annotations

import numpy as np

N_KEYS = 88          # pianoforte standard: A0 (MIDI 21) .. C8 (MIDI 108)
MIDI_MIN = 21
MIDI_MAX = 108

# classi di altezza dei tasti neri (Do# Re# Fa# Sol# La#)
BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})

# larghezza in pixel di una colonna-tasto nella striscia canonica
KEY_PX = 12
STRIP_W = N_KEYS * KEY_PX
STRIP_H = 96


def is_black(index: int) -> bool:
    """True se il tasto i-esimo (0 = A0) e' nero."""
    return (index + MIDI_MIN) % 12 in BLACK_PITCH_CLASSES


def black_mask() -> np.ndarray:
    """Maschera booleana (88,) dei tasti neri."""
    return np.array([is_black(i) for i in range(N_KEYS)], dtype=bool)


def note_name(note: int) -> str:
    """60 -> 'C4'."""
    return "C C# D D# E F F# G G# A A# B".split()[note % 12] + str(note // 12 - 1)


def note_to_index(note: int) -> int:
    """Numero MIDI -> indice di colonna 0..87."""
    return int(note) - MIDI_MIN


def index_to_note(index: int) -> int:
    """Indice di colonna 0..87 -> numero MIDI."""
    return int(index) + MIDI_MIN


def in_range(index: int) -> bool:
    return 0 <= index < N_KEYS


def black_profile() -> np.ndarray:
    """Profilo di riferimento atteso per l'asse dei tasti.

    Ogni tasto nero e' una colonna "scura" nella parte alta della tastiera,
    quindi il profilo e' 1 sui tasti neri e 0 sui bianchi. Serve a stimare
    l'orientamento (specchio / 180 gradi) di una striscia rettificata.
    """
    return black_mask().astype(np.float32)


def black_key_score(strip: np.ndarray, row_frac: float = 0.55, lo: int = 0, hi: int = N_KEYS) -> float:
    """Quanto la striscia assomiglia a una tastiera con la griglia dei tasti giusta.

    I tasti neri sono le colonne piu' scure della fascia alta della tastiera:
    se la rettifica e' corretta, il profilo di luminosita' per colonna e'
    correlato con il pattern dei neri (correlazione vicina a 1). Se la tastiera
    e' specchiata, la correlazione e' negativa, quindi questo stesso numero
    serve anche a stimare l'orientamento.
    """
    band = strip[: max(1, int(strip.shape[0] * row_frac))].astype(np.float32)
    profile = band.mean(axis=0).reshape(N_KEYS, -1).mean(axis=1)      # una media per colonna-tasto
    observed = -(profile - profile.mean())                            # scuro = positivo
    expected = black_profile() * 2.0 - 1.0                            # nero = +1, bianco = -1
    if observed.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(observed[lo:hi], expected[lo:hi])[0, 1])


def detect_input_flip(strips: np.ndarray) -> bool:
    """True se le strisce sembrano specchiate (tastiera "al contrario").

    Confronta il punteggio dei tasti neri normale e specchiato: e' un test
    geometrico, non serve allenare niente e non costa niente.
    """
    normal = float(np.mean([black_key_score(frame) for frame in strips]))
    flipped = float(np.mean([black_key_score(frame[:, ::-1]) for frame in strips]))
    return flipped > normal


def estimate_horizontal_flip(strip: np.ndarray, row_frac: float = 0.55) -> float:
    """Quanto la striscia sembra specchiata, da -1 (normale) a +1 (specchiata).

    Si guarda solo il "fondo" dei tasti (dove i tasti neri finiscono, cioe' la
    parte piu' interna della tastiera) e si confronta la luminosita' media
    normalizzata con il profilo atteso dei tasti neri e con il suo specchio.
    """
    h, w = strip.shape[:2]
    # i tasti neri sono visibili nella fascia alta della tastiera rettificata
    band = strip[: max(1, int(h * row_frac * 0.5)), :].astype(np.float32)
    column = band.mean(axis=0)
    if column.std() < 1e-6:
        return 0.0
    column = (column - column.mean()) / column.std()

    expected = black_profile()
    n = min(len(column), len(expected))
    a = column[:n]
    b = expected[:n] * 2.0 - 1.0                      # nero = +1, bianco = -1
    normal = float(np.dot(a - a.mean(), b - b.mean()))
    flipped = float(np.dot(a - a.mean(), b[::-1] - b[::-1].mean()))
    total = abs(normal) + abs(flipped) + 1e-9
    return (flipped - normal) / total


def check_alignment(strip: np.ndarray, key_index: int) -> float:
    """Diagnostica: contrasto della colonna di un tasto rispetto ai vicini.

    Serve al controllo M0: se la griglia dei tasti e' giusta, la colonna del
    tasto premuto differisce dai vicini in modo marcato.
    """
    x0, x1 = key_index * KEY_PX, (key_index + 1) * KEY_PX
    here = strip[:, x0:x1].mean()
    left = strip[:, max(0, x0 - KEY_PX):x0].mean()
    right = strip[:, x1:x1 + KEY_PX].mean()
    return float(here - 0.5 * (left + right))
