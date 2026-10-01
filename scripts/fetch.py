"""Scarica da Hugging Face solo i pezzi del dataset che servono.

Il dataset intero e' 45 GB, quindi non si scarica tutto:

* `--all-text` prende metadata.json + TSV/ + MIDI/ di tutte le registrazioni
  (pochi MB): bastano per costruire i bersagli e per stimare cosa e' sensato fare;
* `--select train:2,ext-train:5,valid:2,test:1` sceglie le N registrazioni **piu'
  corte** di ogni split: molte registrazioni diverse rendono di piu' di poche
  registrazioni lunghe, costano meno disco e la cache si costruisce prima;
* solo con `--videos` si scaricano i video (~324 MB per registrazione piena).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "PianoVAM/PianoVAM_v1"
TEXT_PATTERNS = ["metadata.json", "TSV/*", "MIDI/*"]


def duration_seconds(text: str) -> float:
    """'0 days 00:12:25.493.000000' -> 745.5"""
    match = re.search(r"(\d+) days (\d+):(\d+):(\d+)", text or "")
    if not match:
        return float("inf")
    days, hours, minutes, seconds = (int(g) for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def load_metadata(raw: Path) -> dict:
    path = raw / "metadata.json"
    if not path.exists():
        hf_hub_download(REPO, "metadata.json", repo_type="dataset", local_dir=str(raw))
    return json.loads(path.read_text())


def choose(meta: dict, spec: str) -> list[str]:
    """spec = 'train:2,ext-train:5' -> le N registrazioni piu' corte per split."""
    chosen: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        split, _, count = part.partition(":")
        rows = [
            (duration_seconds(v.get("duration", "")), v["record_time"])
            for v in meta.values()
            if v.get("split") == split
        ]
        if not rows:
            raise SystemExit(f"nessuna registrazione per lo split {split!r}")
        rows.sort()
        picked = [record for _, record in rows[: int(count or 1)]]
        chosen += picked
        print(f"  {split:<16} -> {picked}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica un sottoinsieme di PianoVAM v1.2")
    parser.add_argument("--raw", default="data/raw", help="cartella di destinazione")
    parser.add_argument("--all-text", action="store_true", help="metadata/TSV/MIDI di tutte le registrazioni")
    parser.add_argument("--select", default="", help="es. train:2,ext-train:5,valid:2,test:1")
    parser.add_argument("--records", nargs="*", default=[], help="record_time espliciti")
    parser.add_argument("--videos", action="store_true", help="scarica anche i video dei record scelti")
    args = parser.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    print(f"metadata.json -> {raw}")
    meta = load_metadata(raw)
    print(f"registrazioni nel dataset: {len(meta)}")

    if args.all_text:
        print("testi (metadata, TSV, MIDI) di tutte le registrazioni...")
        snapshot_download(
            REPO,
            repo_type="dataset",
            local_dir=str(raw),
            allow_patterns=TEXT_PATTERNS,
        )

    records = list(args.records)
    if args.select:
        print("registrazioni scelte (le piu' corte per split):")
        records += choose(meta, args.select)

    for record in dict.fromkeys(records):
        files = [f"TSV/{record}.tsv", f"MIDI/{record}.mid"]
        if args.videos:
            files.insert(0, f"Video/{record}.mp4")
        for name in files:
            path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(raw))
            size = Path(path).stat().st_size / 1e6
            print(f"  {name:<48} {size:8.1f} MB")

    print("fatto.")


if __name__ == "__main__":
    main()
