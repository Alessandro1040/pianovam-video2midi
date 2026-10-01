"""Scarica i video **un po' alla volta**: scarica -> cache in strisce -> cancella il video.

Serve per lavorare su un disco quasi pieno: il video completo di una registrazione
pesa 7-1100 MB, la sua cache in strisce costa ~1.7 MB per minuto, quindi conviene
tenerla e buttare via il video appena costruita. Con `--per-player` la selezione
gira fra i pianisti, cosi' che il budget porti mani diverse e non sempre la stessa.

    python3 scripts/harvest.py --per-player 2 --max-minutes 240 \\
        --exclude-players Taegyun Minwook Yujeong
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download  # noqa: E402

from src import keys, manifest, video  # noqa: E402

REPO = "PianoVAM/PianoVAM_v1"


def free_mb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e6


def remote_sizes(path: str) -> dict[str, int]:
    """Dimensioni remote note (byte) per nome file, se disponibili."""
    if not path or not Path(path).exists():
        return {}
    return {Path(k).stem: int(v) for k, v in json.loads(Path(path).read_text()).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Raccolta incrementale: video -> cache -> cancella video")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--splits", default="train,ext-train")
    parser.add_argument("--per-player", type=int, default=2, help="registrazioni per pianista")
    parser.add_argument("--exclude-players", nargs="*", default=[], help="pianisti tenuti da parte (mai visti in training)")
    parser.add_argument("--only-players", nargs="*", default=[], help="solo questi pianisti")
    parser.add_argument("--max-minutes", type=float, default=None, help="budget di video da scaricare")
    parser.add_argument("--min-record-minutes", type=float, default=2.0, help="scarta i frammenti piu' corti di cosi' (training)")
    parser.add_argument("--min-free-mb", type=float, default=900, help="non scaricare se il disco libero scende sotto questa soglia")
    parser.add_argument("--keep-videos", action="store_true", help="non cancellare i video dopo la cache")
    parser.add_argument("--sizes-json", default="", help="dimensioni remote note, per stimare i GB")
    parser.add_argument("--top-margin", type=float, default=0.4)
    parser.add_argument("--sizes-only", action="store_true", help="mostra il piano senza scaricare nulla")
    args = parser.parse_args()

    raw, cache = Path(args.raw), Path(args.cache)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    plan = manifest.plan_records(
        raw,
        splits=splits,
        per_player=args.per_player,
        exclude_players=tuple(args.exclude_players),
        only_players=tuple(args.only_players),
        max_minutes=args.max_minutes,
        min_record_minutes=args.min_record_minutes,
    )
    if not plan:
        raise SystemExit("piano vuoto: controlla --splits / --exclude-players")

    sizes = remote_sizes(args.sizes_json)
    print(f"piano: {len(plan)} registrazioni da {len(splits)} split "
          f"({args.per_player} per pianista, esclusi {', '.join(args.exclude_players) or 'nessuno'})")
    per_player: dict[str, list[str]] = {}
    total_s = 0.0
    total_bytes = 0
    for record in plan:
        player = manifest.player_of(raw, record)
        seconds = manifest.duration_of(raw, record)
        per_player.setdefault(player, []).append(record)
        total_s += seconds
        total_bytes += sizes.get(record, 0)
    for player, records in per_player.items():
        print(f"  {player:<10} {len(records)} rec  " + ", ".join(records))
    print(f"totale: {total_s / 60:.0f} min di video | cache stimata {total_s / 60 * 1.7:.0f} MB"
          + (f" | download stimato {total_bytes / 1e9:.1f} GB" if total_bytes else ""))
    if args.sizes_only:
        return

    done = 0
    for record in plan:
        out_dir = cache / record
        if (out_dir / "meta.json").exists():
            print(f"[skip] {record}: cache gia' presente")
            continue

        needed_mb = sizes.get(record, 300 * 1024 * 1024) / 1e6 + 100
        if free_mb(raw) < max(needed_mb, args.min_free_mb):
            print(f"[stop] disco libero {free_mb(raw):.0f} MB: non basta per {record} "
                  f"(serve ~{needed_mb:.0f} MB). Rilancia piu' tardi, la cache resta.")
            break

        for name in (f"TSV/{record}.tsv",):
            hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(raw))
        video_path = Path(hf_hub_download(REPO, f"Video/{record}.mp4", repo_type="dataset", local_dir=str(raw)))

        quad = manifest.quad_for_record(raw, record, top_margin=args.top_margin)
        started = time.time()
        print(f"[{record}] {manifest.player_of(raw, record)}, "
              f"{manifest.duration_of(raw, record) / 60:.1f} min, video {video_path.stat().st_size / 1e6:.0f} MB",
              flush=True)
        meta = video.build_cache(
            video_path,
            quad,
            out_dir,
            strip_hw=(keys.STRIP_H, keys.N_KEYS * keys.KEY_PX),
        )
        elapsed = time.time() - started
        size = sum(p.stat().st_size for p in out_dir.glob("chunk_*.mp4")) / 1e6
        print(f"  {meta['n_frames']} frame in {elapsed:.0f} s -> cache {size:.1f} MB", flush=True)

        if not args.keep_videos:
            video_path.unlink(missing_ok=True)
        done += 1

    print(f"fatto: {done} registrazioni raccolte | cache totale "
          f"{sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) / 1e6:.0f} MB | "
          f"libero {free_mb(raw):.0f} MB")


if __name__ == "__main__":
    main()
