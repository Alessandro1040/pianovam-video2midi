"""Test del tracker su una tastiera finta con una camera mossa in modo noto.

La scena e' piatta (una tastiera disegnata sopra uno sfondo con texture), quindi
applicare un'omografia all'immagine e' *esattamente* quello che fa una camera che si
muove davanti a un piano: la posa vera della tastiera la conosciamo in ogni
fotogramma e possiamo misurare l'errore del tracker in unita' di tasto.
"""

from __future__ import annotations

import cv2
import numpy as np

from src import detect, geometry, keys, track

# un quadrilatero "vero", con un po' di prospettiva (come una ripresa da seduti)
QUAD = np.array([[80, 300], [1200, 292], [1196, 560], [84, 556]], dtype=np.float32)
SIZE = (720, 1280)


def tastiera_finta() -> np.ndarray:
    """Striscia canonica disegnata con le misure vere (`detect.bands`)."""
    strip = np.full((keys.STRIP_H, keys.STRIP_W), 235, dtype=np.uint8)
    above, black, _ = detect.bands()
    strip[above] = 95
    for index in range(keys.N_KEYS):
        x0 = index * keys.KEY_PX
        cv2.line(strip, (x0, black.start), (x0, keys.STRIP_H - 1), 170, 1)
        if keys.is_black(index):
            cv2.rectangle(strip, (x0 + 1, black.start), (x0 + keys.KEY_PX - 2, black.stop), 30, -1)
    return strip


def scena(quad: np.ndarray = QUAD, size: tuple[int, int] = SIZE) -> np.ndarray:
    """Fotogramma con la tastiera dentro il quadrilatero, sopra uno sfondo con texture."""
    height, width = size
    rng = np.random.default_rng(3)
    background = cv2.GaussianBlur(rng.integers(70, 120, size=(height, width), dtype=np.uint8).copy(), (0, 0), 1.2)
    to_strip = geometry.homography(quad, keys.STRIP_W, keys.STRIP_H)
    inside = np.linalg.inv(to_strip)
    warped = cv2.warpPerspective(tastiera_finta(), inside, (width, height))
    mask = cv2.warpPerspective(np.full((keys.STRIP_H, keys.STRIP_W), 255, np.uint8), inside, (width, height))
    out = background.copy()
    out[mask > 0] = warped[mask > 0]
    return cv2.GaussianBlur(out, (3, 3), 0.8)


def camera_mossa(frames: int, ampiezza: float = 25.0, size: tuple[int, int] = SIZE, seed: int = 0) -> list[np.ndarray]:
    """Omografie di una camera a mano libera: i 4 angoli ondeggiano con fasi diverse.

    L'ampiezza cresce nei primi 12 fotogrammi (la mano che si scalda), quindi al
    fotogramma 0 la camera e' ferma e la posa iniziale del tracker e' esatta.
    """
    height, width = size
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=(4, 2))
    out = []
    for t in range(frames):
        a = ampiezza * min(1.0, t / 12.0)
        wobble = np.stack(
            [
                a * np.sin(0.35 * t + phases[:, 0]),
                a * 0.6 * np.sin(0.45 * t + phases[:, 1]),
            ],
            axis=1,
        ).astype(np.float32)
        out.append(cv2.getPerspectiveTransform(corners, corners + wobble))
    return out


def scossa(matrice_di_riposo: np.ndarray, matrici: list[np.ndarray]) -> list[np.ndarray]:
    """I fotogrammi della sequenza: la scena vista con la camera mossa."""
    return [cv2.warpPerspective(matrice_di_riposo, m, (SIZE[1], SIZE[0])) for m in matrici]


def pose_vere(matrici: list[np.ndarray], quad: np.ndarray = QUAD) -> np.ndarray:
    """La posa vera della tastiera in ogni fotogramma (i 4 angoli trasformati)."""
    out = []
    for matrix in matrici:
        points = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), matrix)
        out.append(points.reshape(-1, 2))
    return np.stack(out)


def errori(stati: list[track.TrackState], veri: np.ndarray) -> np.ndarray:
    """Errore massimo sugli angoli, in unita' di tasto della tastiera vera."""
    return np.array([detect.corner_error_keys(s.quad, v) for s, v in zip(stati, veri)])


def insegui(frames: list[np.ndarray], quad: np.ndarray = QUAD, config: track.TrackerConfig | None = None) -> list[track.TrackState]:
    """Passa la sequenza al tracker e raccoglie gli stati."""
    tracker = track.KeyboardTracker(quad, config)
    stati = [tracker.first(frames[0])]
    for frame in frames[1:]:
        stati.append(tracker.update(frame))
    return stati


def test_quad_della_matrice_e_quello_di_partenza():
    """Andata e ritorno: la matrice di rettifica deve descrivere il quadrilatero dato."""
    tracker = track.KeyboardTracker(QUAD)
    assert np.allclose(tracker.quad, QUAD, atol=1e-2)


def test_segue_una_camera_che_si_muove():
    """Camera a mano libera: la posa inseguita resta entro mezzo tasto dalla vera."""
    matrici = camera_mossa(48)
    frames = scossa(scena(), matrici)
    stati = insegui(frames)
    error = errori(stati, pose_vere(matrici))

    assert all(s.tracked for s in stati)             # nessun fotogramma "perso"
    assert error.max() < 0.5, f"errore massimo {error.max():.2f} tasti"
    assert error[-1] < 0.5, f"deriva a fine sequenza {error[-1]:.2f} tasti"


def test_le_strisce_inseguite_battono_la_posa_fissa():
    """Il punto dell'inseguimento: le strisce restano quelle giuste.

    Confronto con la striscia del fotogramma 0 (che e' *esattamente* la striscia
    canonica) e con quello che si ottiene oggi tenendo la posa fissa: il tracker deve
    ridurre l'errore di parecchio, non di poco.
    """
    matrici = camera_mossa(48)
    frames = scossa(scena(), matrici)
    stati = insegui(frames)

    ideal = stati[0].strip
    fisse = []
    for frame in frames[1:]:
        matrix = geometry.homography(QUAD, keys.STRIP_W, keys.STRIP_H)
        fisse.append(geometry.rectify(frame, matrix, (keys.STRIP_W, keys.STRIP_H)))

    errore_inseguito = np.mean([np.abs(s.strip.astype(np.float32) - ideal).mean() for s in stati[1:]])
    errore_fisso = np.mean([np.abs(f.astype(np.float32) - ideal).mean() for f in fisse])
    assert errore_inseguito < 0.35 * errore_fisso, (
        f"inseguito {errore_inseguito:.1f} livelli contro {errore_fisso:.1f} di posa fissa"
    )


def test_camera_ferma_non_deriva():
    """Camera ferma: 60 fotogrammi identici non devono muovere la posa."""
    matrici = camera_mossa(60, ampiezza=0.0)
    frames = scossa(scena(), matrici)
    stati = insegui(frames)

    spostamento = np.linalg.norm(stati[-1].quad - stati[0].quad, axis=1).max()
    assert spostamento < 0.1 * detect.key_width_px(QUAD)
    assert all(s.tracked for s in stati)
