"""Video -> MIDI, end to end.

Due modi d'uso:

* registrazione del dataset (calibrazione presa da metadata.json):
    python3 scripts/predict.py --record 2024-02-17_21-44-37
* video qualsiasi (servono i 4 angoli della tastiera):
    python3 scripts/predict.py --video mio.mp4 --quad "121,355 1839,345 1839,558 120,564"

L'ordine dei punti e': sinistra-alto, destra-alto, destra-basso, sinistra-basso
(gli stessi `Point_LT/RT/RB/LB` di PianoVAM). L'orientamento (tastiera normale o
girata di 180 gradi) viene dedotto dai tasti neri, quindi non serve dichiararlo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import decode, geometry, infer, keys, manifest, metrics, model as model_lib, video  # noqa: E402


def parse_quad(text: str) -> np.ndarray:
    """'x,y x,y x,y x,y' -> (4, 2) float32, nell'ordine LT RT RB LB."""
    points = []
    for chunk in text.replace(";", " ").split():
        x, y = chunk.split(",")
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise SystemExit(f"servono 4 punti (LT RT RB LB), ricevuti {len(points)}")
    return np.array(points, dtype=np.float32)


def choose_threshold(path: str | None, fallback: float) -> float:
    """Se esiste un report di valutazione, usa la soglia scelta li'."""
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        if "threshold" in data:
            print(f"soglia {data['threshold']:.2f} dal report {path}")
            return float(data["threshold"])
    return fallback


def orient_samples(cache_dir: Path, count: int = 16) -> np.ndarray:
    """Alcuni frame sparsi, per decidere l'orientamento della tastiera."""
    total = int(video.read_meta(cache_dir)["n_frames"])
    if total == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    indices = np.linspace(0, total - 1, num=min(count, total)).astype(int)
    frames = [video.read_strips(cache_dir, int(index), 1) for index in indices]
    frames = [frame[0] for frame in frames if frame.shape[0]]
    return np.stack(frames) if frames else np.zeros((0, 0, 0), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video -> MIDI")
    parser.add_argument("--record", default=None, help="registrazione PianoVAM (calibrazione da metadata)")
    parser.add_argument("--video", default=None, help="video arbitrario (richiede --quad)")
    parser.add_argument("--quad", default=None, help="4 angoli 'x,y x,y x,y x,y' (LT RT RB LB)")
    parser.add_argument("--checkpoint", default="runs/m5_2ch/best.pt")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", default=None, help="file MIDI di uscita")
    parser.add_argument("--eval-json", default="runs/m5_2ch/eval.json")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--top-margin", type=float, default=0.4)
    parser.add_argument("--orient", choices=("auto", "normal", "flip"), default="auto")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if not args.record and not args.video:
        raise SystemExit("serve --record (dataset PianoVAM) oppure --video + --quad")

    raw = Path(args.raw)
    if args.record:
        name = args.record
        video_path = raw / "Video" / f"{name}.mp4"
        quad = manifest.quad_for_record(raw, name, top_margin=args.top_margin)
        cache_dir = Path(args.cache) / name
        tsv = raw / "TSV" / f"{name}.tsv"
    else:
        video_path = Path(args.video)
        if not args.quad:
            raise SystemExit("per un video arbitrario serve --quad con i 4 angoli della tastiera")
        name = video_path.stem
        points = parse_quad(args.quad)
        entry = {
            corner: f"{point[0]},{point[1]}"
            for corner, point in zip(geometry.CORNER_ORDER, points)
        }
        quad = geometry.quad_from_metadata(entry, top_margin=args.top_margin)
        cache_dir = Path("out/cache") / name
        tsv = None

    out_path = Path(args.out) if args.out else Path("out") / f"{name}.mid"
    if args.rebuild_cache or not (cache_dir / "meta.json").exists():
        print(f"costruisco la cache di {video_path.name}...")
        video.build_cache(video_path, quad, cache_dir)

    samples = orient_samples(cache_dir)
    if args.orient == "auto" and samples.shape[0]:
        input_flip = keys.detect_input_flip(samples)
        normal = float(np.mean([keys.black_key_score(frame) for frame in samples]))
        mirrored = float(np.mean([keys.black_key_score(frame[:, ::-1]) for frame in samples]))
        print(f"orientamento: tasti neri {normal:.2f} normale vs {mirrored:.2f} specchiata")
    else:
        input_flip = args.orient == "flip"
    print(f"tastiera specchiata: {'si (la raddrizzo io)' if input_flip else 'no'}")

    device = "mps" if torch.backends.mps.is_available() and args.device == "auto" else args.device
    if args.device != "auto":
        device = args.device
    model = model_lib.load_checkpoint(args.checkpoint, device)
    radius = model_lib.temporal_radius_for_checkpoint(args.checkpoint)
    print(
        f"modello: canali {model_lib.in_channels_from_checkpoint(args.checkpoint)}, "
        f"raggio della linea di base {radius}, parametri {model_lib.count_parameters(model):,}"
    )

    threshold = choose_threshold(args.eval_json, args.threshold)
    started = time.time()
    probabilities = infer.predict_recording(
        model,
        cache_dir,
        device=device,
        window=args.window,
        stride=args.stride,
        tta_flip=not args.no_tta,
        input_flip=input_flip,
        temporal_radius=radius,
    )
    notes = decode.notes_from_heads(
        probabilities, threshold_onset=threshold, min_distance=args.min_distance
    )
    decode.notes_to_midi(notes, out_path)
    frames = probabilities["onset"].shape[0]
    print(f"{len(notes)} note in {frames / 60:.1f} s di video ({time.time() - started:.0f} s di calcolo)")
    if notes:
        lengths = np.array([note.end_frame - note.start_frame for note in notes]) / 60.0
        lowest = min(keys.index_to_note(note.key) for note in notes)
        highest = max(keys.index_to_note(note.key) for note in notes)
        print(f"durata media nota {lengths.mean() * 1000:.0f} ms, la piu' lunga {lengths.max():.2f} s")
        print(f"estensione: {keys.note_name(lowest)} - {keys.note_name(highest)}")
    print(f"MIDI scritto in {out_path}")

    if tsv is not None and tsv.exists():
        from src import targets

        counts = metrics.note_metrics(targets.read_tsv(tsv), notes, tolerance=3)
        print(
            f"confronto con {tsv.name}: P {counts['precision']:.3f} R {counts['recall']:.3f} "
            f"F1 {counts['f1']:.3f} (tolleranza 50 ms), errore medio {counts['mean_onset_error_ms']:.1f} ms"
        )


if __name__ == "__main__":
    main()
