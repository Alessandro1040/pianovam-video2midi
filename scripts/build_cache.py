"""Costruisce la cache delle strisce rettificate a partire dai video in data/raw/Video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import keys, manifest, video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Video -> cache di strisce rettificate")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--records", nargs="*", default=None, help="default: tutti i video presenti")
    parser.add_argument("--top-margin", type=float, default=0.4, help="margine sopra la tastiera (frazione dell'altezza)")
    parser.add_argument("--limit-frames", type=int, default=None, help="solo per prove rapide")
    parser.add_argument("--strip-h", type=int, default=keys.STRIP_H)
    parser.add_argument("--key-px", type=int, default=keys.KEY_PX)
    args = parser.parse_args()

    raw, cache = Path(args.raw), Path(args.cache)
    records = args.records or manifest.available_videos(raw)
    if not records:
        raise SystemExit(f"nessun video in {raw / 'Video'}: lancia prima scripts/fetch.py --videos")

    for record in records:
        source = raw / "Video" / f"{record}.mp4"
        out_dir = cache / record
        if (out_dir / "meta.json").exists() and not args.limit_frames:
            print(f"[skip] {record}: cache gia' presente")
            continue

        quad = manifest.quad_for_record(raw, record, top_margin=args.top_margin)
        started = time.time()
        print(f"[{record}] decodifica {source.name}...", flush=True)
        meta = video.build_cache(
            source,
            quad,
            out_dir,
            strip_hw=(args.strip_h, keys.N_KEYS * args.key_px),
            max_frames=args.limit_frames,
        )
        elapsed = time.time() - started
        size = sum(p.stat().st_size for p in out_dir.glob("chunk_*.mp4")) / 1e6
        speed = meta["n_frames"] / max(elapsed, 1e-6)
        print(
            f"  {meta['n_frames']} frame in {elapsed:.1f} s ({speed:.0f} fps) -> "
            f"{len(video.chunk_files(out_dir))} blocchi, {size:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
