"""Test dell'operazione piu' delicata: ritagliare la vista dalla cache."""

from __future__ import annotations

import numpy as np
import pytest

from src import keys, video


class FakeMeta(dict):
    pass


def test_chunk_files_vuoto(tmp_path):
    assert video.chunk_files(tmp_path) == []


def test_read_strips_su_cache_vuota_restituisce_array_vuoto(tmp_path):
    """Se la cache non esiste ancora non si deve crashare ma tornare un array vuoto."""
    import json

    cache = tmp_path / "rec"
    cache.mkdir()
    (cache / "meta.json").write_text(
        json.dumps({"n_frames": 0, "chunk_frames": 512, "strip": {"height": 96, "width": 1056}})
    )
    strips = video.read_strips(cache, 0, 32)
    assert strips.shape == (0, 96, 1056)


def test_matrice_locale_tiene_conto_del_crop():
    """La matrice deve lavorare nel sistema di riferimento del ritaglio."""
    from src import geometry

    quad = np.array([[100, 300], [1700, 300], [1700, 500], [100, 500]], dtype=np.float32)
    box = (100, 300, 1700, 500)
    matrix = video._local_matrix(quad, box, (keys.STRIP_W, keys.STRIP_H))
    shifted = quad - np.array([100, 300], dtype=np.float32)
    direct = geometry.homography(shifted, keys.STRIP_W, keys.STRIP_H)
    assert np.allclose(matrix, direct)


def test_lettura_fuori_dai_limiti(tmp_path):
    import json

    cache = tmp_path / "rec"
    cache.mkdir()
    (cache / "meta.json").write_text(
        json.dumps({"n_frames": 100, "chunk_frames": 512, "strip": {"height": 96, "width": 1056}})
    )
    strips = video.read_strips(cache, 90, 40)
    assert strips.shape[0] == 0 or strips.shape[0] <= 10
