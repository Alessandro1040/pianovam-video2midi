"""Video -> cache di strisce rettificate.

Perche' una cache: i video sono 1920x1080 a 60 fps (~324 MB l'uno) e la
porzione utile e' solo la tastiera. Qui il video viene decodificato **una volta**
e trasformato in piccoli video in scala di grigi (la striscia canonica da 88
colonne), divisi in blocchi da `CHUNK_FRAMES` frame: cosi' il training legge
blocchi da pochi MB invece di ri-decoderare 1080p.

La decodifica avviene solo sul riquadro che contiene la tastiera (crop prima del
warp): e' il trucco che rende tutto questo fattibile su un portatile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from . import geometry, keys, targets

CHUNK_FRAMES = 512          # ~8.5 s a 60 fps: un blocco si decodifica in poche decine di ms
CACHE_CRF = 18              # qualita' alta: la striscia e' quasi statica e pesa poco


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def probe(path: str | Path) -> VideoInfo:
    """Durata, fps e dimensioni di un video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 60)
        width, height = stream.codec_context.width, stream.codec_context.height
        n = stream.frames
        if not n:
            n = sum(1 for _ in container.decode(stream))
    return VideoInfo(fps=fps, n_frames=int(n), width=int(width), height=int(height))


def _local_matrix(quad: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    """Omografia calcolata nelle coordinate del crop (x0, y0 sottratti)."""
    x0, y0 = box[0], box[1]
    shifted = quad - np.array([x0, y0], dtype=np.float32)
    return geometry.homography(shifted, size[0], size[1])


def build_cache(
    video_path: str | Path,
    quad: np.ndarray,
    out_dir: str | Path,
    strip_hw: tuple[int, int] = (keys.STRIP_H, keys.STRIP_W),
    chunk_frames: int = CHUNK_FRAMES,
    crf: int = CACHE_CRF,
    max_frames: int | None = None,
    progress_every: int = 3000,
) -> dict:
    """Decodifica il video una volta sola e scrive i blocchi della striscia rettificata."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = strip_hw

    info = probe(video_path)
    with av.open(str(video_path)) as container:
        codec = container.streams.video[0].codec_context
        shape = (codec.height, codec.width)

    box = geometry.bounding_box(quad, shape)
    matrix = _local_matrix(quad, box, (w, h))

    written = 0
    writer = None
    stream = None

    def close_writer() -> None:
        nonlocal writer, stream
        if writer is not None:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
            writer = None
            stream = None

    with av.open(str(video_path)) as container:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"

        for frame in container.decode(in_stream):
            if max_frames is not None and written >= max_frames:
                break
            if written % chunk_frames == 0:
                close_writer()
                writer = av.open(str(out_dir / f"chunk_{written // chunk_frames:05d}.mp4"), mode="w")
                stream = writer.add_stream("libx264", rate=int(round(info.fps)))
                stream.width, stream.height = w, h
                stream.pix_fmt = "gray"
                stream.options = {"crf": str(crf), "preset": "veryfast", "g": str(chunk_frames)}

            image = frame.to_ndarray(format="bgr24")
            x0, y0, x1, y1 = box
            strip = geometry.rectify(image[y0:y1, x0:x1], matrix, (w, h))

            vframe = av.VideoFrame.from_ndarray(strip, format="gray")
            vframe.pts = written
            for packet in stream.encode(vframe):
                writer.mux(packet)

            written += 1
            if progress_every and written % progress_every == 0:
                print(f"    {written} frame ({written / info.fps:.0f} s)", flush=True)

    close_writer()

    meta = {
        "record": video_path.stem,
        "fps": info.fps,
        "n_frames": written,
        "source": {"width": info.width, "height": info.height, "n_frames": info.n_frames},
        "strip": {"height": h, "width": w},
        "chunk_frames": chunk_frames,
        "quad": quad.tolist(),
        "box": list(box),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def chunk_files(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("chunk_*.mp4"))


def read_meta(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "meta.json").read_text())


def _decode_chunk(path: Path, want: tuple[int, int]) -> np.ndarray:
    """Decodifica un blocco e restituisce i frame locali [want[0], want[1])."""
    lo, hi = want
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= hi:
                break
            if i >= lo:
                frames.append(frame.to_ndarray(format="gray"))
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8)


def read_strips(cache_dir: str | Path, start: int, n_frames: int) -> np.ndarray:
    """Strisce [start, start + n_frames) come array (n, H, W) uint8.

    Legge solo i blocchi che servono: con blocchi da 512 frame una finestra
    di training tocca al massimo due file.
    """
    meta = read_meta(cache_dir)
    chunk_frames = int(meta["chunk_frames"])
    total = int(meta["n_frames"])
    height, width = meta["strip"]["height"], meta["strip"]["width"]

    start = max(0, int(start))
    stop = min(total, start + int(n_frames))
    if stop <= start:
        return np.zeros((0, height, width), dtype=np.uint8)

    files = chunk_files(cache_dir)
    parts: list[np.ndarray] = []
    cursor = start
    while cursor < stop:
        index = cursor // chunk_frames
        if index >= len(files):
            break
        chunk_start = index * chunk_frames
        local_lo = cursor - chunk_start
        local_hi = min(chunk_frames, stop - chunk_start)
        part = _decode_chunk(files[index], (local_lo, local_hi))
        if part.size == 0:
            break
        parts.append(part)
        cursor = chunk_start + local_hi

    if not parts:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.concatenate(parts, axis=0)


def load_events(tsv_path: str | Path, n_frames: int | None = None) -> targets.Events:
    """Eventi di nota della registrazione (dal TSV originale)."""
    events = targets.read_tsv(tsv_path)
    if n_frames is None:
        return events
    keep = events.onset_frame < n_frames
    return targets.Events(
        events.onset_frame[keep], events.off_frame[keep], events.key[keep], events.velocity[keep]
    )

