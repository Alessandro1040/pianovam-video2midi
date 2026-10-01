"""Test dell'omografia: la geometria e' la parte su cui si regge tutto il resto."""

from __future__ import annotations

import numpy as np

from src import geometry, keys


def fake_entry(offset_x: float = 0.0, offset_y: float = 0.0, scale: float = 1.0) -> dict:
    """Un quadrilatero trapezoidale (prospettiva), non un rettangolo."""
    return {
        "Point_LT": f"{100 + offset_x:.0f}, {300 + offset_y:.0f}",
        "Point_RT": f"{(100 + 1600 * scale) + offset_x:.0f}, {280 + offset_y:.0f}",
        "Point_RB": f"{(100 + 1600 * scale) + offset_x:.0f}, {560 + offset_y:.0f}",
        "Point_LB": f"{100 + offset_x:.0f}, {600 + offset_y:.0f}",
    }


def test_parse_point():
    assert geometry.parse_point("121, 355") == (121.0, 355.0)
    assert geometry.parse_point(" -3,4 ") == (-3.0, 4.0)


def test_margine_sopra_allarga_il_quadrilatero():
    plain = geometry.quad_from_metadata(fake_entry(), top_margin=0.0)
    padded = geometry.quad_from_metadata(fake_entry(), top_margin=0.5)
    # altezza media del quadrilatero, come la calcola il codice
    height = 0.5 * (abs(plain[3, 1] - plain[0, 1]) + abs(plain[2, 1] - plain[1, 1]))
    assert np.allclose(padded[0, 1], plain[0, 1] - 0.5 * height)
    assert np.allclose(padded[1, 1], plain[1, 1] - 0.5 * height)
    assert np.allclose(padded[3, 1], plain[3, 1])          # il lato "pianista" non si muove
    assert np.allclose(padded[2, 1], plain[2, 1])


def test_omografia_manda_gli_angoli_nel_rettangolo():
    quad = geometry.quad_from_metadata(fake_entry(), top_margin=0.4)
    matrix = geometry.homography(quad, 880, 96)
    corners = np.hstack([quad, np.ones((4, 1), dtype=np.float32)])
    projected = (matrix @ corners.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    expected = np.array([[0, 0], [880, 0], [880, 96], [0, 96]], dtype=np.float32)
    assert np.allclose(projected, expected, atol=1e-3)


def test_la_rettifica_allinea_i_tasti_anche_con_prospettiva():
    """Disegno una tastiera finta nel rettangolo di output e la deformo in prospettiva.

    Poi chiedo alla pipeline di raddrizzarla: se il punteggio dei tasti neri torna
    alto, l'omografia sta facendo il suo lavoro.
    """
    import cv2

    quad = geometry.quad_from_metadata(fake_entry(), top_margin=0.0)
    matrix = geometry.homography(quad, keys.STRIP_W, keys.STRIP_H)

    source = synthetic_strip()
    inverse = np.linalg.inv(matrix)
    image = cv2.warpPerspective(source, inverse, (1800, 700))

    warped = geometry.rectify(image, matrix, (keys.STRIP_W, keys.STRIP_H))
    assert keys.black_key_score(warped) > 0.8


def synthetic_strip() -> np.ndarray:
    """Immagine finta della striscia canonica (bianchi chiari, neri scuri)."""
    strip = np.full((keys.STRIP_H, keys.STRIP_W, 3), 230, dtype=np.uint8)
    top = int(keys.STRIP_H * 0.45)
    for index in range(keys.N_KEYS):
        if keys.is_black(index):
            strip[:top, index * keys.KEY_PX:(index + 1) * keys.KEY_PX] = 25
    return strip


def test_bounding_box_resta_dentro_l_immagine():
    quad = geometry.quad_from_metadata(fake_entry(offset_x=-300, offset_y=-200), top_margin=0.4)
    box = geometry.bounding_box(quad, (700, 1800))
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= 1800 and box[3] <= 700
    assert box[2] > box[0] and box[3] > box[1]


def test_jitter_cambia_i_punti_ma_non_l_ordine():
    quad = geometry.quad_from_metadata(fake_entry(), top_margin=0.4)
    rng = np.random.default_rng(0)
    moved = geometry.jitter_quad(quad, 0.02, rng)
    assert moved.shape == quad.shape
    assert not np.allclose(moved, quad)
    assert np.abs(moved - quad).max() < 0.2 * np.abs(quad).max()
