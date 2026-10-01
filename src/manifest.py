"""Accesso a metadata.json di PianoVAM e agli split ufficiali.

Attenzione al formato: metadata.json e' indicizzato per numero di registrazione
('0', '1', ...) e il nome usato dai file e' `record_time`. Qui la chiave comoda
(`record_time`) diventa quella principale.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import geometry, video

SPLITS = ("train", "ext-train", "valid", "test")


def load_metadata(raw_root: str | Path = "data/raw") -> dict:
    """metadata.json grezzo, indicizzato per numero di registrazione."""
    path = Path(raw_root) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json non trovato in {raw_root}: lancia prima scripts/fetch.py")
    return json.loads(path.read_text())


@lru_cache(maxsize=4)
def _by_record_time(raw_root: str) -> dict:
    return {entry["record_time"]: entry for entry in load_metadata(raw_root).values()}


def record_info(raw_root: str | Path, record: str) -> dict:
    """Descrittore della registrazione (pianista, split, durata, punti di calibrazione...)."""
    table = _by_record_time(str(raw_root))
    if record not in table:
        raise KeyError(f"{record} non presente in metadata.json")
    return table[record]


def available_videos(raw_root: str | Path = "data/raw") -> list[str]:
    """Registrazioni di cui esiste il video scaricato."""
    return sorted(p.stem for p in (Path(raw_root) / "Video").glob("*.mp4"))


def quad_for_record(raw_root: str | Path, record: str, top_margin: float = 0.4) -> np.ndarray:
    """Rettangolo della tastiera annotato dal dataset, con i margini di lavoro."""
    return geometry.quad_from_metadata(record_info(raw_root, record), top_margin=top_margin)


def split_records(raw_root: str | Path, split: str) -> list[str]:
    """record_time di uno split ufficiale, limitatamente a chi ha il video scaricato."""
    return [
        record
        for record in available_videos(raw_root)
        if record_info(raw_root, record).get("split") == split
    ]


def frame_count(cache_root: str | Path, record: str) -> int:
    return int(video.read_meta(Path(cache_root) / record)["n_frames"])


def cached_records(cache_root: str | Path = "data/cache") -> list[str]:
    """Registrazioni con la cache in strisce pronta: e' l'insieme su cui si puo' davvero lavorare.

    Conta la cache, non il video: la raccolta incrementale (`scripts/harvest.py`)
    cancella il video appena la cache e' costruita, quindi l'elenco dei record non
    puo' piu' dipendere dai file .mp4 presenti in data/raw/Video.
    """
    root = Path(cache_root)
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/meta.json"))


def cached_split_records(
    cache_root: str | Path, raw_root: str | Path, split: str
) -> list[str]:
    """Record di uno split ufficiale che hanno la cache (e sono quindi valutabili)."""
    chosen = []
    for record in cached_records(cache_root):
        try:
            info = record_info(raw_root, record)
        except KeyError:
            continue
        if info.get("split") == split:
            chosen.append(record)
    return chosen


def player_of(raw_root: str | Path, record: str) -> str:
    """Pianista principale della registrazione (`P1_name` nel metadata)."""
    info = record_info(raw_root, record)
    return str(info.get("P1_name") or info.get("P2_name") or "?")


def duration_of(raw_root: str | Path, record: str) -> float:
    """Durata in secondi secondo il metadata (0.0 se il campo manca)."""
    text = str(record_info(raw_root, record).get("duration") or "")
    match = re.search(r"(\d+) days (\d+):(\d+):(\d+)(?:\.(\d+))?", text)
    if not match:
        return 0.0
    days, hours, minutes, seconds, frac = match.groups()
    return (
        int(days) * 86400
        + int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + (float("0." + frac) if frac else 0.0)
    )


def plan_records(
    raw_root: str | Path = "data/raw",
    *,
    splits: tuple[str, ...] = ("train", "ext-train"),
    per_player: int = 2,
    exclude_players: tuple[str, ...] = (),
    only_players: tuple[str, ...] = (),
    max_minutes: float | None = None,
    min_record_minutes: float = 0.0,
    shortest_first: bool = True,
) -> list[str]:
    """Sceglie le registrazioni **a giro fra i pianisti**: prima la varietà, poi la quantita'.

    Il criterio "le N piu' corte" (usato all'inizio) riempie il disco di frammenti
    dello stesso pianista: qui si fa un giro round-robin, un pianista alla volta,
    partendo da chi ha piu' materiale disponibile, cosi' che ogni minuto scaricato
    porti una mano nuova invece dell'ennesimo pezzo della stessa persona.

    `min_record_minutes` scarta i frammenti troppo brevi: 22 secondi di un pianista
    valgono quasi zero per il training (e' il caso che aveva prodotto l'insieme
    iniziale), mentre per la valutazione vanno bene anche i frammenti.
    """
    excluded = set(exclude_players)
    allowed = set(only_players) if only_players else None
    by_player: dict[str, list[tuple[float, str]]] = {}
    for entry in load_metadata(raw_root).values():
        if entry.get("split") not in splits:
            continue
        record = entry["record_time"]
        player = str(entry.get("P1_name") or "?")
        if player in excluded or (allowed is not None and player not in allowed):
            continue
        seconds = duration_of(raw_root, record)
        if seconds / 60.0 < min_record_minutes:
            continue
        by_player.setdefault(player, []).append((seconds, record))

    # pianisti con piu' minuti disponibili per primi: sono quelli che possono dare di piu'
    order = sorted(by_player, key=lambda p: -sum(d for d, _ in by_player[p]))
    for player in order:
        by_player[player].sort(reverse=not shortest_first)

    chosen: list[str] = []
    spent = 0.0
    for _round in range(max(per_player, 0)):
        for player in order:
            # il primo record della coda puo' non entrare nel budget residuo, ma un
            # altro record dello stesso pianista si': meglio prenderlo che sprecarlo
            fit = 0
            while fit < len(by_player[player]):
                duration = by_player[player][fit][0]
                if max_minutes is None or spent + duration / 60.0 <= max_minutes:
                    break
                fit += 1
            if fit == len(by_player[player]):
                continue
            duration, record = by_player[player].pop(fit)
            chosen.append(record)
            spent += duration / 60.0
    return chosen

