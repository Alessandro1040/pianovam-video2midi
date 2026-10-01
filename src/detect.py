"""Trovare la tastiera da soli, quando i 4 angoli non sono annotati.

**Stato: sperimentale, non ancora affidabile.** Su frame reali la ricerca sbaglia di
13-21 tasti, quindi non e' utilizzabile per produrre MIDI: vedi la sezione
"Trovare la tastiera da soli" nel README e il test marcato `xfail` in
`tests/test_detect.py`, che spiega cosa manca (stimare il *periodo* dei tasti
nell'immagine invece di sperare che una griglia lo indovini).

Il dataset annota i 4 angoli per tutte le sue 107 registrazioni, quindi per quelle
basta `geometry.quad_from_metadata`. Un video qualsiasi no: li' la tastiera va
cercata nell'immagine.

Qui non si allena niente: si cerca. La funzione obiettivo e' gia' nel progetto ed
e' molto selettiva, `keys.black_key_score`. Se la rettifica e' giusta, il profilo
di luminosita' per colonna si accorda con il disegno dei tasti neri, e il numero
sale. Se la scala orizzontale e' sbagliata (il quadrilatero copre 60 tasti invece
di 88) il disegno perde periodicita' e il numero crolla: e' proprio il tipo di
errore che rovina la trascrizione, quindi ottimizzare quel numero lo corregge.

Un dettaglio importante: il disegno dei tasti neri si ripete quasi ogni 12 tasti,
quindi il punteggio da solo ha massimi "a pettine" ogni 12 tasti e non distingue
la tastiera da quella letta un'ottava piu' in la'. Per sciogliere l'ambiguita' si
aggiunge un termine che chiede ai bordi laterali del quadrilatero di cadere su un
bordo verticale forte dell'immagine: il primo tasto (A0) e l'ultimo (C8) sono i
bordi della tastiera e non si spostano di un'ottava.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry, keys

PARAMS = 8      # 4 angoli x (x, y), in coordinate normalizzate del frame


def quad_to_params(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """4x2 in pixel -> 8 numeri normalizzati (x/larghezza, y/altezza per ogni angolo)."""
    h, w = shape
    return (np.asarray(quad, dtype=np.float64) / np.array([w, h], dtype=np.float64)).reshape(-1)


def params_to_quad(params: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """8 numeri normalizzati -> 4x2 in pixel."""
    h, w = shape
    return (
        np.asarray(params, dtype=np.float64).reshape(4, 2) * np.array([w, h], dtype=np.float64)
    ).astype(np.float32)


def _rect_params(cx: float, cy: float, width: float, height: float) -> np.ndarray:
    """Rettangolo centrato in (cx, cy), tutto in unita' normalizzate."""
    half_w, half_h = width / 2.0, height / 2.0
    return np.array(
        [cx - half_w, cy - half_h, cx + half_w, cy - half_h,
         cx + half_w, cy + half_h, cx - half_w, cy + half_h],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Prior:
    """Dove puo' stare la tastiera: intervalli in cui cercare.

    I default vengono dalle 107 registrazioni annotate: la tastiera sta quasi
    sempre nella fascia centrale, larga e poco alta.
    """

    cx: tuple[float, float] = (0.25, 0.75)
    cy: tuple[float, float] = (0.35, 0.80)
    width: tuple[float, float] = (0.55, 1.0)
    height: tuple[float, float] = (0.06, 0.35)
    samples: int = 7


def to_gray(frame: np.ndarray) -> np.ndarray:
    gray = frame if frame.ndim == 2 else frame[..., :3].mean(axis=2)
    return np.ascontiguousarray(gray.astype(np.uint8))


def score_quad(frame: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> float:
    """Punteggio dei tasti neri per un quadrilatero candidato (alto = meglio)."""
    matrix = geometry.homography(quad, out_w, out_h)
    return keys.black_key_score(geometry.rectify(frame, matrix, (out_w, out_h)))


MARGIN = 0.4        # lo stesso margine con cui sono costruite le strisce del dataset
BLACK_FRAC = 0.64   # quota dell'altezza dei tasti occupata dai tasti neri


def bands(
    out_h: int = keys.STRIP_H, margin: float = MARGIN, black_frac: float = BLACK_FRAC
) -> tuple[slice, slice, slice]:
    """Le tre fasce della striscia: sopra la tastiera, tasti neri, solo tasti bianchi.

    Le frazioni non sono inventate: misurate sulle strisce in cache. Nel profilo per
    riga della correlazione col disegno dei neri, le righe 0-26 sono piatte, le righe
    27-71 hanno il disegno, le righe 72-95 tornano piatte.
    """
    top = int(round(out_h * margin / (1.0 + margin)))
    middle = top + int(round((out_h - top) * black_frac))
    return slice(0, top), slice(top, middle), slice(middle, out_h)


def column_profile(strip: np.ndarray) -> np.ndarray:
    """Luminosita' media di ognuna delle 88 colonne-tasto."""
    return strip.mean(axis=0).reshape(keys.N_KEYS, -1).mean(axis=1).astype(np.float64)


def contrast(profile: np.ndarray) -> float:
    """Quanto le colonne sono diverse fra loro (0 = tutte uguali).

    Si usa il contrasto e non la correlazione perche' su una fascia piatta la
    correlazione col disegno dei neri e' rumore, e puo' capitare che valga 0.8
    per caso: e' quello che faceva sbagliare la ricerca all'inizio.
    """
    span = float(np.percentile(profile, 95) - np.percentile(profile, 5))
    return span / (float(profile.mean()) + 1e-6)


def pattern_presence(band: np.ndarray) -> float:
    """Quanto una fascia mostra il disegno periodico dei tasti neri, pesato dal contrasto.

    Il peso serve perche' su una fascia uniforme la correlazione e' rumore e puo'
    valere 0.8 per caso: moltiplicarla per il contrasto la fa sparire dove non c'e'
    niente da vedere, e la lascia intatta dove la tastiera c'e' davvero.
    """
    correlation = keys.black_key_score(band, row_frac=1.0)
    return float(correlation) * min(1.0, contrast(column_profile(band)) / 0.4)


def score_rect(
    frame: np.ndarray,
    rect: np.ndarray,
    out_w: int = keys.STRIP_W,
    out_h: int = keys.STRIP_H,
    margin: float = MARGIN,
) -> float:
    """Quanto un rettangolo *candidato tastiera* (senza margine) spiega l'immagine.

    Tre richieste, controllate sulla striscia rettificata:

    1. nella fascia dei tasti neri il disegno periodico ci deve essere;
    2. nella fascia sopra la tastiera **non** ci deve essere: se c'e', il rettangolo
       e' salito oltre la tastiera e la sta leggendo fuori posto;
    3. davanti **non** ci deve essere: se li' compare il disegno, il rettangolo e'
       troppo corto e ha tagliato la tastiera a meta'.

    La prima versione di questa funzione chiedeva che le fasce 2 e 3 fossero
    *uniformi*. E' falso, e si e' visto subito sui video veri: sopra la tastiera c'e'
    il pianista, e il suo corpo ha contrasto da vendere (1.8-2.8 dove la tastiera ne
    ha 1.2). Cosi' il rettangolo giusto prendeva un punteggio *negativo* (-2.0) e la
    ricerca non poteva che trovare qualcos'altro. Ora si cerca o non si cerca il
    *disegno dei tasti*, che e' l'unica cosa che distingue la tastiera da un pianista.
    """
    quad = geometry.expand_top(rect, margin)
    matrix = geometry.homography(quad, out_w, out_h)
    strip = geometry.rectify(frame, matrix, (out_w, out_h))
    above, black, white = bands(out_h, margin)
    return float(
        pattern_presence(strip[black])
        - pattern_presence(strip[above])
        - 0.5 * pattern_presence(strip[white])
    )


def gradient_x(frame: np.ndarray) -> np.ndarray:
    gray = frame.astype(np.float32)
    return np.abs(np.gradient(gray if gray.ndim == 2 else gray[..., :3].mean(axis=2), axis=1))


def edge_score(quad: np.ndarray, gx: np.ndarray) -> float:
    """Quanto i bordi laterali del quadrilatero cadono su bordi verticali dell'immagine.

    Il punteggio dei tasti neri e' quasi identico per la tastiera letta un'ottava
    piu' in la': questo termine no, perche' guarda dove la tastiera finisce.
    """
    h, w = gx.shape
    left = np.linspace(quad[0], quad[3], 8)
    right = np.linspace(quad[1], quad[2], 8)
    peaks = []
    for x, y in np.concatenate([left, right]):
        xi, yi = int(np.clip(round(float(x)), 2, w - 3)), int(np.clip(round(float(y)), 0, h - 1))
        peaks.append(float(gx[yi, xi - 2 : xi + 3].max()))
    return float(np.mean(peaks))


def _total_score(
    frame: np.ndarray,
    params: np.ndarray,
    shape: tuple[int, int],
    gx: np.ndarray,
    edge_weight: float,
    out: tuple[int, int] = (keys.STRIP_W, keys.STRIP_H),
) -> float:
    """Punteggio dei tasti neri piu' un piccolo premio per i bordi laterali a fuoco."""
    rect = params_to_quad(params, shape)
    base = score_rect(frame, rect, out_w=out[0], out_h=out[1])
    if edge_weight <= 0:
        return base
    reference = float(np.percentile(gx, 95)) + 1e-6
    return base + edge_weight * 0.15 * edge_score(rect, gx) / reference


def _refine(
    frame: np.ndarray,
    params: np.ndarray,
    edge_weight: float,
    steps: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
) -> tuple[np.ndarray, float]:
    """Discesa a coordinate: prova a muovere un angolo alla volta e tiene il meglio."""
    shape = frame.shape[:2]
    gx = gradient_x(frame)
    best = np.asarray(params, dtype=np.float64).copy()
    best_score = _total_score(frame, best, shape, gx, edge_weight)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for axis in range(PARAMS):
                for delta in (step, -step):
                    trial = best.copy()
                    trial[axis] += delta
                    if not -0.25 <= trial[axis] <= 1.25:
                        continue
                    score = _total_score(frame, trial, shape, gx, edge_weight)
                    if score > best_score + 1e-6:
                        best, best_score, improved = trial, score, True
    return best, best_score


def detect(
    frame: np.ndarray,
    prior: Prior | None = None,
    edge_weight: float = 1.0,
    coarse_out: tuple[int, int] = (528, 48),
    coarse_width: int = 960,
    hint: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Cerca la tastiera: griglia grossolana a scala ridotta, poi discesa a coordinate.

    Ritorna (rettangolo della tastiera 4x2 in pixel del frame, punteggio).

    `hint` e' un rettangolo approssimativo da cui partire (basta un riquadro tirato a
    occhio attorno alla tastiera): in quel caso la griglia si salta e si rifinisce
    direttamente. Serve perche' la scala orizzontale va azzeccata allo 0.5% - con 88
    tasti, un errore di larghezza dell'1% sposta l'ultimo tasto di quasi un tasto
    intero e il disegno periodico non si accorda piu' - e una griglia cieca non ha
    quella precisione. Con un punto di partenza ragionevole la discesa la trova.
    """
    prior = prior or Prior()
    gray = to_gray(frame)
    shape = gray.shape[:2]

    if hint is not None:
        refined, refined_score = _refine(gray, quad_to_params(hint, shape), edge_weight)
        return params_to_quad(refined, shape), refined_score

    small = gray
    if gray.shape[1] > coarse_width:
        import cv2

        scale = coarse_width / gray.shape[1]
        small = cv2.resize(
            gray, (coarse_width, max(8, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA
        )
    gx_small = gradient_x(small)

    best_params, best_score = _rect_params(0.5, 0.5, prior.width[0], prior.height[0]), -np.inf
    for cx in np.linspace(prior.cx[0], prior.cx[1], prior.samples):
        for cy in np.linspace(prior.cy[0], prior.cy[1], prior.samples):
            for width in np.linspace(prior.width[0], prior.width[1], prior.samples):
                for height in np.linspace(prior.height[0], prior.height[1], prior.samples):
                    params = _rect_params(cx, cy, width, height)
                    score = _total_score(
                        small, params, small.shape[:2], gx_small, edge_weight, out=coarse_out
                    )
                    if score > best_score:
                        best_params, best_score = params, score

    refined, refined_score = _refine(gray, best_params, edge_weight)
    return params_to_quad(refined, gray.shape[:2]), refined_score


def key_width_px(quad: np.ndarray) -> float:
    """Larghezza di un tasto in pixel dell'immagine: serve a misurare l'errore."""
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    return float(0.5 * (top + bottom) / keys.N_KEYS)


def corner_error_keys(quad: np.ndarray, reference: np.ndarray) -> float:
    """Distanza massima fra i 4 angoli, in unita' di tasto: 0 = perfetto, 1 = un tasto."""
    return float(np.linalg.norm(quad - reference, axis=1).max() / max(key_width_px(reference), 1e-9))
