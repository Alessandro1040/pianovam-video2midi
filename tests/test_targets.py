"""Test dei bersagli: dal TSV di PianoVAM alle matrici dense."""

from __future__ import annotations

import numpy as np

from src import keys, targets

TSV = """#
# onset\tkey_offset\tframe_offset\tnote\tvelocity
1.000000\t1.200000\t1.200000\t60\t100
1.500000\t1.700000\t9.000000\t64\t60
2.000000\t2.050000\t2.100000\t120\t80
"""


def write_tsv(tmp_path) -> str:
    path = tmp_path / "2024-01-01_00-00-00.tsv"
    path.write_text(TSV)
    return str(path)


def test_lettura_tsv(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    assert len(events) == 2                     # la nota 120 e' fuori dai tasti del pianoforte
    assert list(events.onset_frame) == [60, 90]
    assert list(events.key) == [keys.note_to_index(60), keys.note_to_index(64)]
    assert list(events.velocity) == [100, 60]


def test_i_frame_sono_a_60_fps(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    assert events.onset_frame[0] == 60          # 1.0 s
    assert events.off_frame[0] == 72            # 1.2 s
    assert events.off_frame[1] == 102           # key_offset, non frame_offset (9.0 s = pedale)


def test_bersagli_densi(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    dense = targets.dense_targets(events, 0, 200, onset_dilate=0)
    assert dense.shape == (4, 200, keys.N_KEYS)

    key_c4 = keys.note_to_index(60)
    assert dense[0, 60, key_c4] == 1.0                       # onset esatto
    assert dense[0, 59, key_c4] == 0.0                       # senza dilatazione non sbava
    assert dense[1, 61:73, key_c4].all()                     # tenuto dall'onset al rilascio
    assert dense[1, 72, key_c4] == 1.0                       # il frame di rilascio e' ancora "premuto"
    assert dense[2, 72, key_c4] == 1.0                       # testa di rilascio
    assert dense[3, 60, key_c4] == 100 / 127                 # velocity solo nel frame di onset
    assert dense[3, 61, key_c4] == 0.0


def test_dilatazione_onset(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    dense = targets.dense_targets(events, 0, 200, onset_dilate=1)
    key_c4 = keys.note_to_index(60)
    assert dense[0, 59, key_c4] == 1.0 and dense[0, 61, key_c4] == 1.0
    assert dense[0, 58, key_c4] == 0.0
    assert dense[3, 61, key_c4] == 0.0        # la velocity resta sul frame esatto


def test_finestra_fuori_dal_brano(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    dense = targets.dense_targets(events, 30, 20, onset_dilate=0)
    assert dense[:, :, keys.note_to_index(60)].sum() == 0
    subset = events.subset(30, 50)
    assert len(subset) == 0


def test_velocita_non_nulla_nel_midi(tmp_path):
    events = targets.read_tsv(write_tsv(tmp_path))
    notes = targets.events_to_midi_notes(events)
    assert notes[0][2] == 60 and notes[0][3] == 100
    assert abs(notes[0][0] - 1.0) < 1e-9
