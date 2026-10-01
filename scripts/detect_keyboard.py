"""Trova la tastiera in un video qualsiasi, senza i 4 angoli annotati.

**Sperimentale, non affidabile**: su frame reali sbaglia di 13-21 tasti (vedi
README). Serve come strumento di diagnosi e per guardare l'anteprima, non per
produrre MIDI.

    python3 scripts/detect_keyboard.py 2024-02-15_21-18-14 --out out/tastiera

Per ogni sorgente (un file video o il nome di una registrazione del dataset) prende
alcuni frame sparsi, cerca la tastiera con `src.detect` e tiene la ricerca col
punteggio migliore. Se la sorgente e' una registrazione annotata, il comando
stampa anche l'errore rispetto ai 4 angoli veri, **in unita' di tasto**: sotto 1
significa che la griglia dei tasti e' quella giusta, sopra 2 che la trascrizione
e' spostata di tasti e il modello non puo' farcela.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import detect, geometry, manifest  # noqa: E402


def frames_of(path: Path, count: int, width: int) -> list[np.ndarray]:
    """Alcuni frame sparsi nel video, ridimensionati per lavorare in fretta."""
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = np.linspace(0.05, 0.95, count) * max(total - 1, 1)
    out = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        if frame.shape[1] > width:
            scale = width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        out.append(frame)
    capture.release()
    return out


def resolve(source: str, raw: Path) -> tuple[str | None, Path]:
    """'2024-02-15_21-18-14' -> video del dataset; altrimenti un percorso."""
    candidate = Path(source)
    if candidate.exists():
        return None, candidate
    video = raw / "Video" / f"{source}.mp4"
    if not video.exists():
        raise SystemExit(f"non trovo ne' {candidate} ne' {video}")
    return source, video


def main() -> None:
    parser = argparse.ArgumentParser(description="Ricerca della tastiera in un video")
    parser.add_argument("sources", nargs="+", help="file video oppure record_time del dataset")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--frames", type=int, default=5, help="quanti frame provare")
    parser.add_argument("--width", type=int, default=1600, help="larghezza di lavoro")
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--out", default="out/keyboard", help="cartella delle anteprime")
    parser.add_argument("--json", default=None, help="file JSON con tutti i quadrilateri trovati")
    args = parser.parse_args()

    raw, out_dir = Path(args.raw), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for source in args.sources:
        record, video_path = resolve(source, raw)
        frames = frames_of(video_path, args.frames, args.width)
        if not frames:
            print(f"[{source}] nessun frame leggibile")
            continue

        best_quad, best_score, best_frame = None, -np.inf, None
        for frame in frames:
            quad, score = detect.detect(frame, edge_weight=args.edge_weight)
            if score > best_score:
                best_quad, best_score, best_frame = quad, score, frame

        entry = {"quad": best_quad.tolist(), "score": round(float(best_score), 4)}
        if record is not None:
            reference = manifest.quad_for_record(raw, record, top_margin=0.0)
            scale = best_frame.shape[1] / frames[0].shape[1]
            entry["error_keys"] = round(detect.corner_error_keys(best_quad * scale, reference), 2)
            entry["score_annotated"] = round(
                detect.score_quad(best_frame, reference / scale, 1056, 96), 4
            )
        report[source] = entry

        preview = best_frame.copy()
        cv2.polylines(preview, [np.asarray(best_quad, dtype=np.int32)], True, (0, 255, 0), 3)
        if record is not None:
            cv2.polylines(
                preview,
                [np.asarray(reference / scale, dtype=np.int32)],
                True,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(str(out_dir / f"{Path(source).stem}.png"), preview)

        line = f"[{source}] punteggio {best_score:.3f}"
        if "error_keys" in entry:
            line += (
                f" | errore {entry['error_keys']:.2f} tasti"
                f" | punteggio con gli angoli veri {entry['score_annotated']:.3f}"
            )
        print(line + f" | anteprima {out_dir / f'{Path(source).stem}.png'}", flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"scritto {args.json}")


if __name__ == "__main__":
    main()
