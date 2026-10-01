"""Test della selezione dei record: varieta' di pianisti e raccolta incrementale.

Il criterio di scelta conta: la prima versione prendeva "le N registrazioni piu'
corte" e produsse 7 minuti di video di tre pianisti, quasi tutti frammenti.
"""

from __future__ import annotations

import json

from src import manifest


def _entry(record: str, split: str, player: str, minutes: float) -> dict:
    hours, rest = divmod(int(minutes * 60), 3600)
    mins, secs = divmod(rest, 60)
    return {
        "record_time": record,
        "split": split,
        "P1_name": player,
        "duration": f"0 days {hours:02d}:{mins:02d}:{secs:02d}.000000",
    }


def _raw(tmp_path, entries: list[dict]):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "metadata.json").write_text(json.dumps({str(i): e for i, e in enumerate(entries)}))
    return raw


def test_duration_of_legge_il_formato_del_dataset(tmp_path):
    raw = _raw(tmp_path, [_entry("r1", "train", "Alfa", 2.5)])
    assert abs(manifest.duration_of(raw, "r1") - 150.0) < 1e-6


def test_player_of_usa_il_nome_del_pianista(tmp_path):
    raw = _raw(tmp_path, [_entry("r1", "train", "Alfa", 3)])
    assert manifest.player_of(raw, "r1") == "Alfa"


def test_plan_gira_fra_i_pianisti(tmp_path):
    """Con 2 record per pianista, i primi tre scelti devono essere di tre pianisti diversi."""
    raw = _raw(tmp_path, [
        _entry("a1", "train", "Alfa", 3), _entry("a2", "train", "Alfa", 4),
        _entry("b1", "train", "Beta", 3), _entry("b2", "train", "Beta", 4),
        _entry("c1", "ext-train", "Gamma", 3), _entry("c2", "ext-train", "Gamma", 4),
    ])
    plan = manifest.plan_records(raw, per_player=2)
    assert len(plan) == 6
    assert len({manifest.player_of(raw, r) for r in plan[:3]}) == 3
    assert sorted(plan) == ["a1", "a2", "b1", "b2", "c1", "c2"]


def test_plan_esclude_i_pianisti_tenuti_da_parte(tmp_path):
    raw = _raw(tmp_path, [
        _entry("a1", "train", "Alfa", 3), _entry("b1", "train", "Beta", 3),
    ])
    plan = manifest.plan_records(raw, per_player=3, exclude_players=("Beta",))
    assert plan == ["a1"]


def test_plan_scarta_i_frammenti_troppo_brevi(tmp_path):
    """22 secondi di un pianista valgono quasi zero: sono il caso che aveva rovinato la prima selezione."""
    raw = _raw(tmp_path, [
        _entry("corto", "train", "Alfa", 0.37), _entry("lungo", "train", "Alfa", 5),
    ])
    assert manifest.plan_records(raw, per_player=5) == ["corto", "lungo"]
    assert manifest.plan_records(raw, per_player=5, min_record_minutes=2) == ["lungo"]


def test_plan_rispetta_il_budget_di_minuti(tmp_path):
    raw = _raw(tmp_path, [
        _entry("a1", "train", "Alfa", 4), _entry("a2", "train", "Alfa", 4),
        _entry("b1", "train", "Beta", 4), _entry("b2", "train", "Beta", 4),
    ])
    plan = manifest.plan_records(raw, per_player=2, max_minutes=10)
    assert plan == ["a1", "b1"]  # il terzo record porterebbe a 12 minuti: budget finito
    assert sum(manifest.duration_of(raw, r) for r in plan) / 60 <= 10


def test_plan_prende_il_record_che_entra_nel_budget(tmp_path):
    """Con poco budget residuo deve scegliere un record piu' corto, non saltare il pianista."""
    raw = _raw(tmp_path, [
        _entry("lungo", "train", "Alfa", 9), _entry("corto", "train", "Alfa", 2),
    ])
    assert manifest.plan_records(raw, per_player=2, max_minutes=5) == ["corto"]


def test_plan_solo_questi_pianisti(tmp_path):
    raw = _raw(tmp_path, [
        _entry("a1", "train", "Alfa", 3), _entry("b1", "train", "Beta", 3),
    ])
    assert manifest.plan_records(raw, per_player=1, only_players=("Beta",)) == ["b1"]


def test_cached_records_guarda_la_cache_non_i_video(tmp_path):
    """La raccolta incrementale cancella i video: l'elenco deve venire dalla cache."""
    cache = tmp_path / "cache"
    for record in ("r1", "r2", "senza_meta"):
        (cache / record).mkdir(parents=True)
        if record != "senza_meta":
            (cache / record / "meta.json").write_text(json.dumps({"n_frames": 10}))
    assert manifest.cached_records(cache) == ["r1", "r2"]


def test_cached_split_records_filtra_per_split(tmp_path):
    raw = _raw(tmp_path, [
        _entry("r1", "train", "Alfa", 3), _entry("r2", "valid", "Beta", 3),
    ])
    cache = tmp_path / "cache"
    for record in ("r1", "r2"):
        (cache / record).mkdir(parents=True)
        (cache / record / "meta.json").write_text(json.dumps({"n_frames": 10}))
    assert manifest.cached_split_records(cache, raw, "valid") == ["r2"]
    assert manifest.cached_split_records(cache, raw, "test") == []
