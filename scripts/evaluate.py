"""Valutazione: quante note azzecca il modello, e con quale soglia.

Tre numeri, in ordine di importanza:

* **F1 a livello di nota** con tolleranza di 3 frame (50 ms): e' la metrica della
  trascrizione, e conta solo se *sia* il tasto *sia* l'istante sono giusti;
* F1 a livello di frame sullo stato del tasto: il modello vede il tasto tenuto?
* errore medio di onset in ms, calcolato solo sulle note matchate.

La soglia di decisione si sceglie **sul set di validazione** e poi si applica al
test: sceglierla sul test sarebbe barare.
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

from src import decode, infer, manifest, metrics, model as model_lib, targets  # noqa: E402


def resolve_cache(cache_roots: list[Path], record: str) -> Path:
    """In quale cartella sta la cache di questo record.

    Il training legge `data/cache`; i pianisti tenuti da parte per il test di
    generalizzazione stanno in una cartella separata, cosi' non possono finire
    per sbaglio nell'addestramento.
    """
    for root in cache_roots:
        if (root / record / "meta.json").exists():
            return root
    return cache_roots[0]


def probabilities_for(model, record: str, cache_roots: list[Path], args, device: str):
    """Inferenza su una registrazione (con TTA opzionale) e bersagli veri."""
    cache_dir = resolve_cache(cache_roots, record) / record
    probabilities = infer.predict_recording(
        model,
        cache_dir,
        device=device,
        window=args.window,
        stride=args.stride,
        tta_flip=not args.no_tta,
        temporal_radius=getattr(args, "temporal_radius", 15),
    )
    events = infer.events_from_cache(cache_dir, args.raw)
    if args.max_frames:
        limit = min(args.max_frames, probabilities["onset"].shape[0])
        probabilities = {name: value[:limit] for name, value in probabilities.items()}
        events = events.subset(0, limit)
    return probabilities, events


def state_scores(probabilities: dict[str, np.ndarray], events: targets.Events) -> dict:
    """F1 a livello di frame sullo stato del tasto."""
    frames = probabilities["frame"].shape[0]
    truth = targets.dense_targets(events, 0, frames, onset_dilate=0)[1] > 0.5
    return metrics.state_metrics(truth, probabilities["frame"] > 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valutazione note (F1 con tolleranza 50 ms)")
    parser.add_argument("--checkpoint", default="runs/m5_2ch/best.pt")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--cache-extra", nargs="*", default=[], help="altre cartelle di cache (es. i pianisti tenuti da parte)")
    parser.add_argument("--records", nargs="*", default=None, help="default: split valid")
    parser.add_argument("--test-records", nargs="*", default=None, help="default: split test")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60")
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--max-frames", type=int, default=24000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--out", default=None, help="file JSON dove salvare i risultati")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    device = "mps" if torch.backends.mps.is_available() and args.device == "auto" else args.device
    if args.device != "auto":
        device = args.device

    model = model_lib.load_checkpoint(args.checkpoint, device)
    args.temporal_radius = model_lib.temporal_radius_for_checkpoint(args.checkpoint)
    print(
        f"checkpoint {args.checkpoint} su {device}, canali "
        f"{model_lib.in_channels_from_checkpoint(args.checkpoint)}, raggio {args.temporal_radius}, "
        f"TTA flip: {not args.no_tta}"
    )

    valid_records = args.records or manifest.cached_split_records(args.cache, args.raw, "valid")
    test_records = (
        args.test_records
        if args.test_records is not None
        else manifest.cached_split_records(args.cache, args.raw, "test")
    )
    cache_roots = [Path(args.cache)] + [Path(path) for path in args.cache_extra]
    report = {"checkpoint": args.checkpoint, "valid": {}, "test": {}}

    totals = {threshold: {"matched": 0, "n_gt": 0, "n_pred": 0} for threshold in thresholds}
    for record in valid_records:
        started = time.time()
        probabilities, events = probabilities_for(model, record, cache_roots, args, device)
        per_threshold = {}
        for threshold in thresholds:
            notes = decode.notes_from_heads(
                probabilities, threshold_onset=threshold, min_distance=args.min_distance
            )
            counts = metrics.note_metrics(events, notes, tolerance=3)
            per_threshold[threshold] = counts
            for name in totals[threshold]:
                totals[threshold][name] += counts[name]
        report["valid"][record] = {
            "notes": len(events),
            "frames": int(probabilities["onset"].shape[0]),
            "state": state_scores(probabilities, events),
            "per_threshold": {str(key): value for key, value in per_threshold.items()},
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"[valid] {record}: {len(events)} note, {probabilities['onset'].shape[0]} frame "
            f"({time.time() - started:.0f} s)",
            flush=True,
        )

    summary = {}
    for threshold, counts in totals.items():
        precision = counts["matched"] / counts["n_pred"] if counts["n_pred"] else 0.0
        recall = counts["matched"] / counts["n_gt"] if counts["n_gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[threshold] = {"precision": precision, "recall": recall, "f1": f1}
    best_threshold = max(summary, key=lambda value: summary[value]["f1"])
    report["valid_summary"] = {str(key): value for key, value in summary.items()}
    report["threshold"] = best_threshold
    print("\nvalidazione, F1 sulle note al variare della soglia:")
    for threshold, scores in summary.items():
        marker = " <== scelta" if threshold == best_threshold else ""
        print(
            f"  soglia {threshold:.2f}: P {scores['precision']:.3f} R {scores['recall']:.3f} "
            f"F1 {scores['f1']:.3f}{marker}"
        )

    for record in test_records:
        started = time.time()
        probabilities, events = probabilities_for(model, record, cache_roots, args, device)
        notes = decode.notes_from_heads(
            probabilities, threshold_onset=best_threshold, min_distance=args.min_distance
        )
        counts = metrics.note_metrics(events, notes, tolerance=3)
        report["test"][record] = {
            "notes": len(events),
            "metrics": counts,
            "state": state_scores(probabilities, events),
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"[test]  {record}: P {counts['precision']:.3f} R {counts['recall']:.3f} "
            f"F1 {counts['f1']:.3f}  errore onset {counts['mean_onset_error_ms']:.1f} ms "
            f"({time.time() - started:.0f} s)",
            flush=True,
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()

