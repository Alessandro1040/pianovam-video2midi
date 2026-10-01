"""Test della geometria canonica dei tasti."""

from __future__ import annotations

import numpy as np

from src import keys


def synthetic_keyboard(flipped: bool = False):
    """Tastiera finta in stile striscia rettificata: tasti bianchi chiari, neri scuri."""
    strip = np.full((keys.STRIP_H, keys.STRIP_W), 230, dtype=np.uint8)
    top = int(keys.STRIP_H * 0.45)
    for index in range(keys.N_KEYS):
        if keys.is_black(index):
            strip[:top, index * keys.KEY_PX:(index + 1) * keys.KEY_PX] = 25
    if flipped:
        strip = strip[:, ::-1].copy()
    return strip


def test_layout_88_tasti():
    assert keys.N_KEYS == 88
    assert keys.index_to_note(0) == 21
    assert keys.index_to_note(87) == 108


def test_i_neri_sono_dove_ci_si_aspetta():
    # A0 (21) e' bianco, A#0 (22) e' nero, e ogni ottava si ripete
    assert not keys.is_black(keys.note_to_index(21))
    assert keys.is_black(keys.note_to_index(22))
    assert keys.is_black(keys.note_to_index(61))
    assert keys.black_mask().sum() == 36      # 36 neri e 52 bianchi su 88 tasti


def test_nomi_delle_note():
    assert keys.note_name(60) == "C4"
    assert keys.note_name(21) == "A0"
    assert keys.note_name(108) == "C8"


def test_punteggio_tasti_neri_alto_se_allineata_e_negativo_se_specchiata():
    """Il pattern dei tasti neri e' quasi simmetrico: cio' che conta e' il confronto.

    La versione allineata deve avere un punteggio alto; la versione specchiata
    resta positiva (il disegno dei neri non e' simmetrico al 100%), ma deve
    essere **chiaramente piu' bassa**: su quello si basa il rilevamento.
    """
    normal = keys.black_key_score(synthetic_keyboard())
    mirrored = keys.black_key_score(synthetic_keyboard(flipped=True))
    assert normal > 0.9
    assert mirrored < 0.8
    assert mirrored < normal - 0.2


def test_rilevamento_specchio():
    assert keys.detect_input_flip(np.stack([synthetic_keyboard()])) is False
    assert keys.detect_input_flip(np.stack([synthetic_keyboard(flipped=True)])) is True
