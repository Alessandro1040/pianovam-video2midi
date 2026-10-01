"""Inferenza su una registrazione intera, con test-time augmentation.

Due accorgimenti che alzano la robustezza senza riaddestrare niente:

* si scorre la registrazione a finestre con `stride` minore della finestra e si
  tiene la media delle probabilita' (i bordi di finestra sono i punti deboli);
* si predice anche la striscia specchiata e si rimette a posto la risposta
  (invertendo l'ordine dei tasti). Se il modello ha imparato la simmetria, la
  media delle due viste e' piu' stabile di ognuna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import data, decode, keys, video


@torch.no_grad()
def forward_features(
    model: torch.nn.Module, features: np.ndarray, device: str, batch_frames: int = 512
) -> dict[str, np.ndarray]:
    """Feature (T, C, H, W) float32 -> 4 mappe di probabilita' (T, 88)."""
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in model.heads}
    for start in range(0, features.shape[0], batch_frames):
        chunk = features[start:start + batch_frames]
        tensor = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)   # (1, T, C, H, W)
        heads = model(tensor)
        for name, value in heads.items():
            outputs[name].append(torch.sigmoid(value)[0].float().cpu().numpy())
    return {name: np.concatenate(value, axis=0) for name, value in outputs.items()}


@torch.no_grad()
def predict_strips(
    model: torch.nn.Module,
    strips: np.ndarray,
    device: str,
    batch_frames: int = 512,
    temporal_radius: int = 15,
) -> dict[str, np.ndarray]:
    """Strisce grezze (T, H, W) uint8 -> 4 mappe di probabilita' (T, 88)."""
    features = data.temporal_features(strips, temporal_radius)
    return forward_features(model, features, device, batch_frames)


def _device_type(device: str) -> str:
    return "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")


@torch.no_grad()
def predict_recording(
    model: torch.nn.Module,
    cache_dir: str | Path,
    device: str = "cpu",
    window: int = 256,
    stride: int = 192,
    tta_flip: bool = True,
    input_flip: bool = False,
    temporal_radius: int = 15,
) -> dict[str, np.ndarray]:
    """Probabilita' su tutta la registrazione (T, 88) per ogni testa.

    Ogni finestra legge `temporal_radius` frame in piu' per lato: le feature di
    differenza ai bordi altrimenti avrebbero una linea di base monca.
    """
    cache_dir = Path(cache_dir)
    total = int(video.read_meta(cache_dir)["n_frames"])
    if total == 0:
        raise ValueError(f"cache vuota: {cache_dir}")
    window = min(window, total)

    accumulators: dict[str, np.ndarray] | None = None
    counts = np.zeros(total, dtype=np.float32)

    starts = list(range(0, max(1, total - window + 1), stride))
    if not starts or starts[-1] + window < total:
        starts.append(max(0, total - window))

    for start in starts:
        read_start = max(0, start - temporal_radius)
        lead = start - read_start
        raw = video.read_strips(cache_dir, read_start, window + 2 * temporal_radius)
        if raw.shape[0] == 0:
            continue
        if input_flip:
            # tastiera "al contrario": si rimette nell'ordine canonico prima di predire
            raw = np.ascontiguousarray(raw[:, :, ::-1])
        features = data.temporal_features(raw, temporal_radius)[lead:lead + window]
        length = features.shape[0]
        if length == 0:
            continue

        heads = forward_features(model, features, device)
        if tta_flip:
            mirrored = forward_features(model, np.ascontiguousarray(features[:, :, :, ::-1]), device)
            for name, value in mirrored.items():
                # la vista specchiata va rimessa a posto invertendo l'ordine dei tasti
                heads[name] = 0.5 * (heads[name] + value[:, ::-1])
        if input_flip:
            heads = {name: value[:, ::-1] for name, value in heads.items()}
        if accumulators is None:
            accumulators = {name: np.zeros((total, keys.N_KEYS), dtype=np.float32) for name in heads}
        for name, value in heads.items():
            accumulators[name][start:start + length] += value
        counts[start:start + length] += 1

    if accumulators is None:
        raise RuntimeError("nessuna finestra decodificata")
    counts = np.maximum(counts, 1.0)[:, None]
    return {name: value / counts for name, value in accumulators.items()}


def events_from_cache(cache_dir: str | Path, raw_root: str | Path = "data/raw"):
    """Eventi annotati della registrazione (per valutare la predizione)."""
    record = Path(cache_dir).name
    meta = video.read_meta(cache_dir)
    return video.load_events(Path(raw_root) / "TSV" / f"{record}.tsv", int(meta["n_frames"]))


def decode_to_midi(probabilities: dict[str, np.ndarray], out_path: str | Path, **kwargs) -> list[decode.Note]:
    """Probabilita' -> note -> file MIDI."""
    notes = decode.notes_from_heads(probabilities, **kwargs)
    decode.notes_to_midi(notes, Path(out_path))
    return notes
