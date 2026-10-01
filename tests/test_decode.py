"""Test della decodifica (probabilita' -> note -> MIDI) e delle metriche."""

from __future__ import annotations

import numpy as np
import pytest

from src import decode, keys, metrics, targets


def heads_from_events(events: list[tuple[int, int, int, int]], length: int = 200) -> dict[str, np.ndarray]:
    """Costruisce mappe di probabilita' "perfette" da una lista di note."""
    onset = np.zeros((length, keys.N_KEYS), dtype=np.float32)
    state = np.zeros_like(onset)
    offset = np.zeros_like(onset)
    velocity = np.zeros_like(onset)
    for start, end, key, vel in events:
        onset[start, key] = 0.9
        state[start + 1:end + 1, key] = 0.9
        offset[end, key] = 0.9
        velocity[start, key] = vel / 127
    return {"onset": onset, "frame": state, "offset": offset, "velocity": velocity}


def test_picchi_isolati():
    probability = np.zeros((50, keys.N_KEYS), dtype=np.float32)
    probability[10, 40] = 0.8
    peaks = decode.peak_pick(probability, threshold=0.5, min_distance=2)
    assert len(peaks) == 1
    frame, key, score = peaks[0]
    assert (frame, key) == (10, 40)
    assert score == pytest.approx(0.8, abs=1e-6)


def test_soglia_e_distanza_minima():
    probability = np.zeros((50, keys.N_KEYS), dtype=np.float32)
    probability[10, 3] = 0.4                      # sotto soglia
    probability[20, 5] = 0.9
    probability[21, 5] = 0.7                      # a un frame di distanza: e' lo stesso evento
    peaks = decode.peak_pick(probability, threshold=0.5, min_distance=3)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (20, 5)


def test_decodifica_di_una_nota():
    events = [(10, 20, 40, 100)]
    notes = decode.notes_from_heads(heads_from_events(events))
    assert len(notes) == 1
    note = notes[0]
    assert note.start_frame == 10 and note.end_frame == 20 and note.key == 40
    assert note.velocity == 100


def test_nota_lunga_senza_rilascio_chiaro():
    """Se la testa di rilascio non scatta, la nota si allunga finche' il tasto resta premuto."""
    heads = heads_from_events([(10, 30, 40, 90)])
    heads["offset"][:] = 0.0
    notes = decode.notes_from_heads(heads)
    assert notes[0].end_frame >= 30


def test_scrittura_midi(tmp_path):
    import pretty_midi

    # key sono indici di colonna: 19 -> note 40, 34 -> note 55
    events = [(10, 20, keys.note_to_index(40), 100), (30, 40, keys.note_to_index(55), 70)]
    notes = decode.notes_from_heads(heads_from_events(events))
    assert decode.notes_to_events(notes)[1].tolist() == [keys.note_to_index(40), keys.note_to_index(55)]

    path = decode.notes_to_midi(notes, tmp_path / "out.mid")
    assert path.exists()
    midi = pretty_midi.PrettyMIDI(str(path))
    played = sorted(midi.instruments[0].notes, key=lambda note: note.start)
    assert [note.pitch for note in played] == [40, 55]
    # tolleranza 2 ms: il MIDI quantizza i tempi sui tick della sua risoluzione
    assert abs(played[0].start - 10 / 60) < 2e-3
    assert abs(played[0].end - 20 / 60) < 2e-3
    assert played[0].velocity == 100


def test_metriche_match_perfetto():
    gt = targets.Events(
        onset_frame=np.array([10, 50, 90]),
        off_frame=np.array([20, 60, 100]),
        key=np.array([30, 40, 50], dtype=np.int8),
        velocity=np.full(3, 80, dtype=np.uint8),
    )
    notes = [
        decode.Note(start_frame=10, end_frame=20, key=30, velocity=80),
        decode.Note(start_frame=52, end_frame=60, key=40, velocity=80),   # dentro la tolleranza
        decode.Note(start_frame=90, end_frame=100, key=50, velocity=80),
    ]
    scores = metrics.note_metrics(gt, notes, tolerance=3)
    assert scores["f1"] == 1.0
    assert scores["mean_onset_error_ms"] < 50


def test_metriche_nota_sbagliata_e_falsi_positivi():
    gt = targets.Events(
        onset_frame=np.array([10]),
        off_frame=np.array([20]),
        key=np.array([30], dtype=np.int8),
        velocity=np.array([80], dtype=np.uint8),
    )
    wrong_key = [decode.Note(start_frame=10, end_frame=20, key=31, velocity=80)]
    assert metrics.note_metrics(gt, wrong_key)["f1"] == 0.0

    too_late = [decode.Note(start_frame=20, end_frame=30, key=30, velocity=80)]
    assert metrics.note_metrics(gt, too_late, tolerance=3)["f1"] == 0.0

    extra = [
        decode.Note(start_frame=10, end_frame=20, key=30, velocity=80),
        decode.Note(start_frame=60, end_frame=70, key=44, velocity=80),
    ]
    scores = metrics.note_metrics(gt, extra, tolerance=3)
    assert scores["recall"] == 1.0 and scores["precision"] == 0.5


def test_stato_del_tasto():
    truth = np.zeros((10, keys.N_KEYS), dtype=bool)
    truth[2:5, 10] = True
    scores = metrics.state_metrics(truth, truth.copy())
    assert scores["f1"] == 1.0
    scores = metrics.state_metrics(truth, np.zeros_like(truth))
    assert scores["f1"] == 0.0
