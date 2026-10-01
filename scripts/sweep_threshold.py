"""Curva dell'F1 sulle note al variare della soglia, su registrazioni a scelta.

E' uno strumento **diagnostico**, non il modo di scegliere il numero da
riportare: quello si sceglie una volta sola sulla validazione ufficiale
(`scripts/evaluate.py`) e si applica al test. Qui invece la soglia viene fatta
scorrere, quindi il massimo di questa tabella e' un numero ottimistico e non va
citato come risultato.

A cosa serve: quando un pianista tenuto fuori va male, distingue due cause che
si assomigliano moltissimo nei numeri.

* se l'F1 **risale** abbassando la soglia, il modello ci vede ma e' mal
  calibrato su quel pianista (i suoi onset stanno piu' in basso che sui pianisti
  visti), e il rimedio e' la calibrazione, non piu' dati;
* se l'F1 **resta basso** a ogni soglia, il modello non vede i tasti che suona:
  quello e' un limite di capacita' o di inquadratura, non di soglia.

Stampa anche la probabilita' media di onset contro la frequenza vera: se la
prima e' molto piu' grande della seconda, il modello sovra-predice (effetto del
`pos_weight` nel training) e la soglia deve rimediare.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import decode, infer, metrics, model as model_lib, targets  # noqa: E402


def resolve_cache(cache_roots: list[Path], record: str) -> Path:
    """In quale cartella sta la cache di questo record (come in evaluate.py)."""
    for root in cache_roots:
        if (root / record / "meta.json").exists():
            return root
    raise FileNotFoundError(f"cache di {record} non trovata in {cache_roots}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 sulle note al variare della soglia (diagnostica)")
    parser.add_argument("--checkpoint", default="runs/m6_2ch/best.pt")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--cache-extra", nargs="*", default=[], help="altre cartelle di cache (es. i pianisti tenuti da parte)")
    parser.add_argument("--records", nargs="+", required=True, help="registrazioni da analizzare")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = registrazione intera (default del diagnostico)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = model_lib.load_checkpoint(args.checkpoint, device)
    radius = model_lib.temporal_radius_for_checkpoint(args.checkpoint)
    print(f"checkpoint {args.checkpoint} su {device}, raggio {radius}, TTA flip: {not args.no_tta}", flush=True)

    cache_roots = [Path(args.cache)] + [Path(path) for path in args.cache_extra]
    for record in args.records:
        started = time.time()
        cache_dir = resolve_cache(cache_roots, record) / record
        probabilities = infer.predict_recording(
            model,
            cache_dir,
            device=device,
            window=args.window,
            stride=args.stride,
            tta_flip=not args.no_tta,
            temporal_radius=radius,
        )
        events = infer.events_from_cache(cache_dir, args.raw)
        if args.max_frames:
            limit = min(args.max_frames, probabilities["onset"].shape[0])
            probabilities = {name: value[:limit] for name, value in probabilities.items()}
            events = events.subset(0, limit)

        frames = probabilities["onset"].shape[0]
        predicted_mean = float(probabilities["onset"].mean())
        # Frequenza vera di onset, calcolata come nel training (onset dilatato di
        # +/-1 frame): senza la dilatazione il confronto con la p predetta sarebbe
        # gonfiato di un fattore ~3 e non sarebbe lo stesso numero di train.py.
        true_mean = float(targets.dense_targets(events, 0, frames, onset_dilate=1)[0].mean())
        print(
            f"\n{record}: {frames} frame ({frames / 60:.1f} s), {len(events)} note vere"
            f" | p onset media {predicted_mean:.5f} contro frequenza vera {true_mean:.5f}"
            f" ({predicted_mean / max(true_mean, 1e-9):.1f}x)"
        )
        print("  soglia |     P      R      F1   | errore onset")
        for threshold in thresholds:
            notes = decode.notes_from_heads(probabilities, threshold_onset=threshold, min_distance=args.min_distance)
            counts = metrics.note_metrics(events, notes, tolerance=3)
            print(
                f"   {threshold:.2f}  | {counts['precision']:.3f}  {counts['recall']:.3f}  {counts['f1']:.3f}  |"
                f" {counts['mean_onset_error_ms']:6.1f} ms  ({counts['n_pred']} note predette)"
            )
        print(f"  ({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
