"""Invarianza geometrica: la tastiera viene rettificata con un'omografia.

Idea chiave del progetto: invece di far *imparare* al modello l'invarianza a
spostamenti, rotazioni, inclinazioni e prospettiva (servirebbero dati
multi-vista che non abbiamo), la geometria viene normalizzata **prima** della
rete. Il dataset annota i 4 angoli del rettangolo della tastiera
(`Point_LT/RT/RB/LB` in metadata.json) per tutte le 107 registrazioni: da quei
4 punti si ricava l'omografia che porta la tastiera sempre nella stessa
striscia canonica. Dopo il warp un tasto premuto sta sempre nella stessa
colonna di pixel, qualunque sia la posizione della camera.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from . import keys

# ordine degli angoli: sinistra-alto, destra-alto, destra-basso, sinistra-basso
CORNER_ORDER = ("Point_LT", "Point_RT", "Point_RB", "Point_LB")


def parse_point(text: str) -> tuple[float, float]:
    """'121, 355' -> (121.0, 355.0)."""
    x, y = (float(v) for v in re.split(r"[ ,]+", text.strip()))
    return x, y


def expand_top(quad: np.ndarray, top_margin: float = 0.4, side_margin: float = 0.0) -> np.ndarray:
    """Allarga il quadrilatero verso l'alto (il lato del pianista) e, se serve, ai lati.

    La formula sta qui e non altrove perche' e' una convenzione condivisa: la stessa
    usata per costruire le strisce del dataset, per la ricerca della tastiera nei
    video nuovi e per il jitter di calibrazione. Ordine angoli: LT, RT, RB, LB.
    """
    lt, rt, rb, lb = (np.asarray(corner, dtype=np.float32) for corner in quad)
    height = 0.5 * (abs(lb[1] - lt[1]) + abs(rb[1] - rt[1]))
    width = 0.5 * (abs(rt[0] - lt[0]) + abs(rb[0] - lb[0]))

    up = np.array([0.0, -top_margin * height], dtype=np.float32)
    left = np.array([-side_margin * width, 0.0], dtype=np.float32)
    right = np.array([side_margin * width, 0.0], dtype=np.float32)
    return np.stack([lt + up + left, rt + up + right, rb + right, lb + left])


def quad_from_metadata(entry: dict, top_margin: float = 0.4, side_margin: float = 0.0) -> np.ndarray:
    """Rettangolo della tastiera (4x2 float32) con i margini di lavoro.

    `top_margin` allarga il quadrilatero verso l'alto (il lato del pianista):
    le dita arrivano da li', e vedere la mano che scende rende piu' facile
    capire quando il tasto viene premuto. Il margine laterale serve invece a
    gestire eventuali imprecisioni della calibrazione.
    """
    corners = np.stack([np.array(parse_point(entry[k]), dtype=np.float32) for k in CORNER_ORDER])
    return expand_top(corners, top_margin=top_margin, side_margin=side_margin)


@dataclass(frozen=True)
class QuadScale:
    """Metrica del quadrilatero, usata per calibrare l'ampiezza del jitter."""

    width: float
    height: float


def quad_scale(quad: np.ndarray) -> QuadScale:
    w = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    h = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    return QuadScale(float(w), float(h))


def jitter_quad(quad: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Perturba i 4 angoli: simula calibrazione imprecisa e pianoforti diversi.

    E' una delle due sorgenti di invarianza del progetto (l'altra e' il flip
    orizzontale con inversione delle etichette): il modello impara a dare la
    stessa risposta anche se la griglia dei tasti e' spostata di qualche pixel.
    """
    if strength <= 0:
        return quad
    scale = quad_scale(quad)
    noise = rng.normal(0.0, strength, size=quad.shape).astype(np.float32)
    noise[:, 0] *= scale.width
    noise[:, 1] *= scale.height
    return quad + noise


def homography(
    quad: np.ndarray,
    out_w: int = keys.STRIP_W,
    out_h: int = keys.STRIP_H,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Omografia 3x3 che porta il quadrilatero della tastiera nel rettangolo di output."""
    dst = np.array(
        [[0.0, 0.0], [out_w, 0.0], [out_w, out_h], [0.0, out_h]], dtype=np.float32
    )
    src = jitter_quad(quad, jitter, rng) if jitter > 0 and rng is not None else quad
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst)


def bounding_box(quad: np.ndarray, shape: tuple[int, int], pad: int = 8) -> tuple[int, int, int, int]:
    """Riquadro (x0, y0, x1, y1) che contiene il quadrilatero, dentro l'immagine.

    Serve a decodificare solo la porzione utile del frame: sui video 1080p60
    questo e' il singolo accorgimento che rende la pipeline utilizzabile.
    """
    h, w = shape
    x0 = int(np.floor(quad[:, 0].min())) - pad
    y0 = int(np.floor(quad[:, 1].min())) - pad
    x1 = int(np.ceil(quad[:, 0].max())) + pad
    y1 = int(np.ceil(quad[:, 1].max())) + pad
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def rectify(frame: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Applica l'omografia al frame e restituisce la striscia in scala di grigi."""
    warped = cv2.warpPerspective(
        frame,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.ndim == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped
