"""M0 — controllo della calibrazione: la griglia dei tasti cade dove deve?

Due verifiche:

1. numerica: i tasti neri veri sono le colonne piu' scure della parte alta
   della tastiera. Se la griglia di 88 colonne e' allineata, la correlazione
   fra il profilo osservato e il profilo atteso dei tasti neri e' alta.
   Sotto ~0.4 qualcosa non torna e conviene guardare i PNG.
2. visiva: salva dei PNG leggibili (striscia intera + zoom sul tasto premuto)
   nei frame intorno a note annotate, cosi' si vede se il tasto che si muove
   sta nella colonna evidenziata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import geometry, keys, manifest, targets, video  # noqa: E402


def column_profile_score(strip: np.ndarray, lo: int = 0, hi: int = keys.N_KEYS) -> float:
    """Vedi `keys.black_key_score`: qui resta solo per comodita' di lettura."""
    return keys.black_key_score(strip, lo=lo, hi=hi)



def frames_of(video_path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decodifica una volta il video e tiene solo i frame richiesti."""
    kept: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        last = max(wanted)
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                kept[index] = frame.to_ndarray(format="bgr24")
            if index > last:
                break
    return kept


def annotate(strip: np.ndarray, key_index: int, label: str, key_px: int, scale: int = 1) -> np.ndarray:
    """Striscia in BGR con la griglia dei tasti (giallo sui Do) e il tasto atteso in rosso."""
    image = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width = image.shape[:2]
    for i in range(keys.N_KEYS + 1):
        x = min(width - 1, i * key_px * scale)
        color = (0, 200, 255) if (i + keys.MIDI_MIN) % 12 == 0 else (0, 90, 0)
        cv2.line(image, (x, 0), (x, height - 1), color, 1)
    x0, x1 = key_index * key_px * scale, (key_index + 1) * key_px * scale - 1
    cv2.rectangle(image, (max(0, x0), 0), (min(width - 1, x1), height - 1), (0, 0, 255), 1)
    if label:
        cv2.putText(image, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def choose_onsets(events: targets.Events, n_frames: int, count: int) -> list[int]:
    """Onset annotati ben distribuiti nel tempo (uno per intervallo, in frame)."""
    frames = np.unique(events.onset_frame)
    if frames.size == 0:
        return []
    edges = np.linspace(0, n_frames, count + 1)
    chosen: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = frames[(frames >= a) & (frames < b)]
        if inside.size:
            chosen.append(int(inside[inside.size // 2]))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Controllo calibrazione (M0)")
    parser.add_argument("--record", default="2024-02-17_21-37-57")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="out/calibration")
    parser.add_argument("--top-margin", type=float, default=0.4)
    parser.add_argument("--key-px", type=int, default=keys.KEY_PX)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    raw = Path(args.raw)
    record = args.record
    video_path = raw / "Video" / f"{record}.mp4"
    out_dir = Path(args.out) / record
    out_dir.mkdir(parents=True, exist_ok=True)

    quad = manifest.quad_for_record(raw, record, top_margin=args.top_margin)
    info = video.probe(video_path)
    box = geometry.bounding_box(quad, (info.height, info.width))
    offset = np.array([box[0], box[1]], dtype=np.float32)
    size = (keys.N_KEYS * args.key_px, keys.STRIP_H)
    matrix = geometry.homography(quad - offset, *size)

    events = targets.read_tsv(raw / "TSV" / f"{record}.tsv")
    onsets = choose_onsets(events, info.n_frames, args.count)
    wanted = {t + delta for t in onsets for delta in (-4, 0, 4)}
    print(f"{record}: {info.n_frames} frame @ {info.fps:.0f} fps, {len(events)} note, onset scelti {onsets}")
    frames = frames_of(video_path, wanted)

    def strip_at(index: int) -> np.ndarray | None:
        frame = frames.get(index)
        if frame is None:
            return None
        return geometry.rectify(frame[box[1]:box[3], box[0]:box[2]], matrix, size)

    for onset in onsets:
        here = events.key[events.onset_frame == onset]
        key_index = int(here[0]) if here.size else 0
        rows = []
        for delta in (-4, 0, 4):
            strip = strip_at(onset + delta)
            if strip is None:
                continue
            label = f"t{delta:+d}  tasti={[keys.note_name(keys.index_to_note(int(k))) for k in here]}" if delta == 0 else f"t{delta:+d}"
            rows.append(annotate(strip, key_index, label, args.key_px))
        if not rows:
            continue
        full = np.concatenate(rows, axis=0)
        x0 = max(0, (key_index - 7) * args.key_px)
        x1 = min(keys.N_KEYS * args.key_px, (key_index + 8) * args.key_px)
        zooms = []
        for delta in (-4, 0, 4):
            strip = strip_at(onset + delta)
            if strip is None:
                continue
            zoom = annotate(strip[:, x0:x1], key_index, "", args.key_px, scale=4)
            pad = max(0, full.shape[1] - zoom.shape[1])
            zooms.append(
                cv2.copyMakeBorder(
                    zoom,
                    0,
                    0,
                    pad // 2,
                    pad - pad // 2,
                    cv2.BORDER_CONSTANT,
                    value=(40, 40, 40),
                )
            )
        full = np.concatenate([full] + zooms, axis=0)
        path = out_dir / f"onset_{onset:06d}.png"
        cv2.imwrite(str(path), full)
        print(f"  scritto {path.name}")

    sample = strip_at(onsets[0]) if onsets else None
    if sample is not None:
        cv2.imwrite(str(out_dir / "crop_grezzo.png"), frames[onsets[0]][box[1]:box[3], box[0]:box[2]])
        cv2.imwrite(str(out_dir / "striscia.png"), sample)
        print(
            "  allineamento tasti neri (1.0 = perfetto): "
            f"totale {column_profile_score(sample):.2f}, "
            f"sinistra {column_profile_score(sample, 0, 44):.2f}, "
            f"destra {column_profile_score(sample, 44, 88):.2f}"
        )


if __name__ == "__main__":
    main()

