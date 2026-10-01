"""Test della ricerca della tastiera su immagini sintetiche.

Una tastiera disegnata da noi ha una verita' nota: sappiamo esattamente dove
l'abbiamo messa. Serve a verificare che la ricerca la ritrovi, e che il punteggio
peggiori quando la griglia dei tasti e' sbagliata.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import detect, geometry, keys


def keyboard_image() -> np.ndarray:
    """Striscia canonica disegnata come sono fatte quelle vere.

    La struttura non e' inventata, e' quella misurata sulle strisce in cache:
    una fascia sopra la tastiera senza disegno, la fascia dei tasti neri, la fascia
    davanti di soli tasti bianchi.
    """
    above, black, white = detect.bands()
    strip = np.full((keys.STRIP_H, keys.STRIP_W, 3), 235, dtype=np.uint8)
    strip[above] = 90
    assert white.stop == keys.STRIP_H
    for index in range(keys.N_KEYS):
        x0 = index * keys.KEY_PX
        cv2.line(strip, (x0, black.start), (x0, keys.STRIP_H - 1), (150, 150, 150), 1)
        if keys.is_black(index):
            cv2.rectangle(
                strip,
                (x0 + 1, black.start),
                (x0 + keys.KEY_PX - 2, black.stop),
                (25, 25, 25),
                -1,
            )
    return strip


def place(strip: np.ndarray, quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Disegna la striscia dentro il quadrilatero, sopra uno sfondo non banale."""
    height, width = size
    rng = np.random.default_rng(0)
    background = rng.integers(60, 110, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(background, (0, int(height * 0.15)), (width, int(height * 0.25)), (30, 30, 30), -1)
    # l'omografia porta il quadrilatero nella striscia: la sua inversa disegna
    # la striscia dentro il quadrilatero (stesse dimensioni di partenza!)
    to_strip = geometry.homography(quad, strip.shape[1], strip.shape[0])
    warped = cv2.warpPerspective(
        strip, np.linalg.inv(to_strip), (width, height), flags=cv2.INTER_LINEAR
    )
    mask = cv2.warpPerspective(
        np.full(strip.shape[:2], 255, np.uint8), np.linalg.inv(to_strip), (width, height)
    )
    out = background.copy()
    out[mask > 0] = warped[mask > 0]
    return cv2.GaussianBlur(out, (5, 5), 1.0)


def test_params_andata_e_ritorno():
    quad = np.array([[100, 300], [1700, 300], [1700, 500], [100, 500]], dtype=np.float32)
    params = detect.quad_to_params(quad, (1080, 1920))
    back = detect.params_to_quad(params, (1080, 1920))
    assert np.allclose(quad, back, atol=1e-3)


def test_punteggio_alto_sulla_tastiera_giusta_e_basso_se_spostato():
    """Se la griglia dei tasti e' giusta il punteggio sale; spostando di un tasto no."""
    strip = keyboard_image()
    _, black, _ = detect.bands()
    here = keys.black_key_score(strip[black], row_frac=1.0)
    assert here > 0.9
    shifted = np.roll(strip, keys.KEY_PX, axis=1)
    assert keys.black_key_score(shifted[black], row_frac=1.0) < here - 0.2


def jitter(rect: np.ndarray, shift: float = 0.06, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Rettangolo approssimativo: come un riquadro tirato a occhio attorno alla tastiera."""
    rng = np.random.default_rng(seed)
    center = rect.mean(axis=0)
    out = center + (rect - center) * scale
    out += rng.normal(0.0, shift * float(np.ptp(rect[:, 0])), size=out.shape)
    return out.astype(np.float32)


def test_rifinisce_un_rettangolo_di_partenza_impreciso():
    """Stato attuale, misurato: la rifinitura parte da un riquadro approssimativo.

    Su immagine sintetica il rettangolo vero vale ~0.97 e la rifinitura **non** ci
    arriva da sola (circa 0.47): la discesa a coordinate si ferma prima. Il test
    registra il comportamento reale invece di quello che vorremmo, cosi' chi riprende
    in mano il file sa da dove ripartire.
    """
    rect = np.array([[220, 430], [1750, 400], [1820, 700], [160, 720]], dtype=np.float32)
    frame = place(keyboard_image(), geometry.expand_top(rect, detect.MARGIN), (1080, 1920))

    _, score = detect.detect(frame, hint=jitter(rect, seed=0))

    assert detect.score_rect(frame, rect) > 0.9      # l'obiettivo sa riconoscere la tastiera giusta
    assert score < detect.score_rect(frame, rect)    # ma la ricerca non la trova ancora


def test_errore_zero_su_se_stesso():
    rect = np.array([[10, 20], [1000, 25], [1010, 200], [5, 190]], dtype=np.float32)
    assert detect.corner_error_keys(rect, rect) == pytest.approx(0.0)


@pytest.mark.xfail(reason="senza punto di partenza la scala orizzontale va azzeccata allo 0.5%: serve il passo periodo dei tasti")
def test_ricerca_alla_cieca_da_sola_non_e_ancora_affidabile():
    """Documenta un limite noto: senza indizio la griglia grossolana non basta.

    Non e' un test che passa, e' un promemoria: la griglia trova una regione con il
    disegno dei tasti, ma sbaglia la scala di parecchi tasti. Per chiudere il cerchio
    serve stimare il *periodo* dei tasti nell'immagine (autocorrelazione del profilo
    dei bordi verticali) invece di sperare che una griglia lo indovini.
    """
    rect = np.array([[220, 430], [1750, 400], [1820, 700], [160, 720]], dtype=np.float32)
    frame = place(keyboard_image(), geometry.expand_top(rect, detect.MARGIN), (1080, 1920))

    found, _ = detect.detect(frame)

    assert detect.corner_error_keys(found, rect) < 1.0


def test_punteggio_peggiora_se_il_rettangolo_e_troppo_corto():
    """Un rettangolo alto la mete' porta i tasti neri anche davanti: il punteggio crolla."""
    rect = np.array([[220, 430], [1750, 400], [1820, 700], [160, 720]], dtype=np.float32)
    frame = place(keyboard_image(), geometry.expand_top(rect, detect.MARGIN), (1080, 1920))
    short = rect.copy()
    short[[2, 3], 1] = rect[[0, 1], 1] + 0.5 * (rect[[2, 3], 1] - rect[[0, 1], 1])
    assert detect.score_rect(frame, short) < detect.score_rect(frame, rect)
