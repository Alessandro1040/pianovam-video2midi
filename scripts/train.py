"""Training del modello video -> MIDI.

Perdita: quattro teste, ognuna con il suo peso.

* `onset` / `frame` / `offset`: BCE su logits con `pos_weight`, perche' i
  positivi sono rarissimi (un tasto su 88, per ~2-3 frame, ~9 volte al secondo);
* `velocity`: MSE solo nei frame di onset effettivi (altrove non c'e' niente da
  predire). Peso basso: e' un extra, il modello non deve inseguirla.

Le soglie di decisione **non** si scelgono a mano: alla fine si cerca quella che
massimizza l'F1 sulle note del set di validazione (scripts/evaluate.py), cosi' il
punto di lavoro e' calibrato sui dati.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, keys, manifest, model as model_lib, video  # noqa: E402

LOSS_WEIGHTS = {"onset": 1.0, "frame": 0.5, "offset": 0.5, "velocity": 0.1}
POS_WEIGHT = {"onset": 8.0, "frame": 8.0, "offset": 8.0}


class BatchSampler:
    """Costruisce i batch decodificando il meno possibile.

    La decodifica dei blocchi e' la voce piu' cara del training: si tiene un
    blocco da ~1000 frame in memoria per registrazione e si estraggono le
    finestre da li', avanzando in modo sequenziale (letture cache-friendly).
    """

    def __init__(self, dataset: data.ClipDataset, batch_size: int, rng: np.random.Generator, augment: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = rng
        self.augment = augment
        self.extra = 2 * (dataset.config.augment.shift_frames if augment else 0) + 1
        self.need = dataset.config.read_frames(augment)
        self.cursor = {record: 0 for record in dataset.records}
        self.blocks: dict[str, tuple[int, np.ndarray]] = {}

    def _block(self, record: str, block_frames: int = 900) -> tuple[int, np.ndarray]:
        total = self.dataset.lengths[record]
        cached = self.blocks.get(record)
        if cached is not None and cached[0] + self.need < total - 1 and self.rng.random() < 0.75:
            return cached
        start = self.cursor.get(record, 0)
        if start + self.need >= total:
            start = int(self.rng.integers(0, max(1, total // 2)))
        count = min(block_frames + self.need, total - start)
        strips = video.read_strips(self.dataset.cache_root / record, start, count)
        self.cursor[record] = min(total - self.need, start + block_frames)
        self.blocks[record] = (start, strips)
        return self.blocks[record]

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(strips (B, T, H, W), bersagli (B, 4, T, 88)) float32/float32."""
        inputs, targets_list = [], []
        for _ in range(self.batch_size):
            record = self.dataset.records[int(self.rng.integers(len(self.dataset.records)))]
            start, strips = self._block(record)
            if strips.shape[0] < self.need:
                continue
            offset = int(self.rng.integers(0, strips.shape[0] - self.need + 1))
            window = strips[offset:offset + self.need]
            inputs.append(window)
            targets_list.append((record, start + offset, window))

        batch_inputs, batch_targets = [], []
        for record, target_start, window in targets_list:
            example, dense, _ = data.make_example(
                window, self.dataset.events[record], target_start, self.dataset.config, self.rng, self.augment
            )
            batch_inputs.append(example)
            batch_targets.append(dense)
        return torch.from_numpy(np.stack(batch_inputs)), torch.from_numpy(np.stack(batch_targets))


def estimate_priors(dataset: data.ClipDataset, samples: int = 24) -> dict[str, float]:
    """Frequenza media dei positivi per ogni testa: serve a inizializzare i bias."""
    rng = np.random.default_rng(0)
    totals = np.zeros(3, dtype=np.float64)
    frames = 0
    for index in range(samples):
        item = dataset[index]
        dense = item["targets"].numpy()
        totals += dense[:3].sum(axis=(1, 2))
        frames += dense.shape[1] * dense.shape[2]
    priors = {
        "onset": float(totals[0] / max(frames, 1)),
        "frame": float(totals[1] / max(frames, 1)),
        "offset": float(totals[2] / max(frames, 1)),
    }
    return priors


def compute_loss(
    outputs: dict[str, torch.Tensor], targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE pesata sulle tre teste di evento + MSE sulla velocity nei soli onset."""
    total = torch.zeros((), device=targets.device)
    detail: dict[str, float] = {}
    for index, name in enumerate(("onset", "frame", "offset")):
        target = targets[:, index]
        weight = torch.tensor(POS_WEIGHT.get(name, 1.0), device=targets.device)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs[name], target, pos_weight=weight)
        total = total + LOSS_WEIGHTS[name] * loss
        detail[name] = float(loss.detach())

    onset_mask = targets[:, 0] > 0.5
    if onset_mask.any():
        prediction = torch.sigmoid(outputs["velocity"])[onset_mask]
        loss = nn.functional.mse_loss(prediction, targets[:, 3][onset_mask])
    else:
        loss = torch.zeros((), device=targets.device)
    total = total + LOSS_WEIGHTS["velocity"] * loss
    detail["velocity"] = float(loss.detach())
    return total, detail


@torch.no_grad()
def validate(model: torch.nn.Module, dataset: data.ClipDataset, device: str, windows: int = 12) -> dict:
    """Perdita di validazione e calibrazione della testa di onset."""
    model.eval()
    totals: dict[str, float] = {}
    predicted: list[float] = []
    positive = 0.0
    count = 0
    for index in range(windows):
        item = dataset[index]
        strips = item["strips"].unsqueeze(0).to(device)
        targets = item["targets"].unsqueeze(0).to(device)
        outputs = model(strips)
        _, detail = compute_loss(outputs, targets)
        for name, value in detail.items():
            totals[name] = totals.get(name, 0.0) + value
        predicted.append(float(torch.sigmoid(outputs["onset"]).mean()))
        positive += float(targets[:, 0].mean())
        count += 1
    count = max(count, 1)
    return {
        "val_onset_loss": totals.get("onset", float("nan")) / count,
        "val_frame_loss": totals.get("frame", float("nan")) / count,
        "val_offset_loss": totals.get("offset", float("nan")) / count,
        "val_pred_mean": sum(predicted) / count,
        "val_target_mean": positive / count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training video -> MIDI")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--train-records", nargs="*", default=None, help="default: split train + ext-train")
    parser.add_argument("--val-records", nargs="*", default=None, help="default: split valid")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--clip-frames", type=int, default=64)
    parser.add_argument("--temporal-radius", type=int, default=15,
                        help="frame di contesto per il canale di differenza (0 = un solo canale)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-windows", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--init-from", default=None, help="checkpoint da cui ripartire (solo pesi)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = manifest.cached_records(args.cache) or manifest.available_videos(args.raw)
    train_records = args.train_records or [
        record
        for record in all_records
        if manifest.record_info(args.raw, record).get("split") in ("train", "ext-train")
    ]
    val_records = args.val_records or [
        record for record in all_records if manifest.record_info(args.raw, record).get("split") == "valid"
    ]
    if not train_records:
        raise SystemExit("nessuna registrazione di training: costruisci prima la cache")
    if not val_records:
        val_records = train_records[-1:]
        print(f"attenzione: nessuna registrazione di validazione, uso {val_records}")

    print(f"device: {device}")
    print(f"train ({len(train_records)}): {train_records}")
    print(f"val   ({len(val_records)}): {val_records}")

    train_config = data.DatasetConfig(
        cache_root=args.cache,
        raw_root=args.raw,
        clip_frames=args.clip_frames,
        temporal_radius=args.temporal_radius,
        enabled=True,
    )
    val_config = data.DatasetConfig(
        cache_root=args.cache,
        raw_root=args.raw,
        clip_frames=args.clip_frames,
        temporal_radius=args.temporal_radius,
        enabled=False,
    )
    train_ds = data.ClipDataset(train_records, train_config, seed=args.seed)
    val_ds = data.ClipDataset(val_records, val_config, seed=args.seed + 1000)
    print(f"frame totali in training: {train_ds.total_frames}")

    model = model_lib.KeyNet(in_channels=train_config.in_channels).to(device)
    priors = estimate_priors(train_ds, samples=12)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=True))
        print(f"pesi caricati da {args.init_from}")
    else:
        model.init_head_bias(priors)
    field = model.receptive_field_frames()
    print(f"parametri: {model_lib.count_parameters(model):,}  campo recettivo: {field} frame ({field / 60:.2f} s)")
    print("prior stimati: " + ", ".join(f"{k}={v:.5f}" for k, v in priors.items()))
    (out_dir / "args.json").write_text(
        json.dumps(vars(args) | {"priors": priors, "device": device}, indent=2)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch

    def lr_at(step: int) -> float:
        """Fattore di learning rate (0..1): LambdaLR lo moltiplica per il lr base."""
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    sampler = BatchSampler(train_ds, args.batch_size, np.random.default_rng(args.seed), augment=True)
    log_path = out_dir / "log.csv"
    log_path.write_text("epoch,step,loss,onset,frame,offset,velocity,lr,seconds\n")
    run_training(args, model, sampler, val_ds, optimizer, scheduler, device, out_dir, log_path)


def run_training(args, model, sampler, val_ds, optimizer, scheduler, device, out_dir, log_path) -> None:
    """Loop principale: passi con log, validazione a fine epoca, salvataggio del migliore."""
    best = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        running: dict[str, float] = {}
        for _ in range(args.steps_per_epoch):
            strips, targets = sampler.batch()
            strips = strips.to(device)
            targets = targets.to(device)
            outputs = model(strips)
            loss, detail = compute_loss(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            running["loss"] = running.get("loss", 0.0) + float(loss.detach())
            for name, value in detail.items():
                running[name] = running.get(name, 0.0) + value
            step += 1

            if step % args.log_every == 0:
                done = {key: value / args.log_every for key, value in running.items()}
                print(
                    f"ep {epoch} step {step:>5}  loss {done['loss']:.4f}  onset {done['onset']:.4f} "
                    f"frame {done['frame']:.4f} offset {done['offset']:.4f} vel {done['velocity']:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                with open(log_path, "a") as handle:
                    handle.write(
                        f"{epoch},{step},{done['loss']:.5f},{done['onset']:.5f},{done['frame']:.5f},"
                        f"{done['offset']:.5f},{done['velocity']:.5f},{scheduler.get_last_lr()[0]:.6f},"
                        f"{time.time() - started:.1f}\n"
                    )
                running = {}

        metrics = validate(model, val_ds, device, windows=args.val_windows)
        print(
            f"ep {epoch} fine: val onset {metrics['val_onset_loss']:.4f} frame {metrics['val_frame_loss']:.4f} "
            f"offset {metrics['val_offset_loss']:.4f} | media predetta {metrics['val_pred_mean']:.5f} "
            f"vs target {metrics['val_target_mean']:.5f} ({time.time() - started:.0f} s)",
            flush=True,
        )
        torch.save(model.state_dict(), out_dir / "last.pt")
        # Si salva anche la singola epoca. `last.pt` viene sovrascritto e
        # `best.pt` e' scelto sulla loss di onset della validazione, che guarda
        # pianisti gia' visti e satura molto prima dell'F1 sulle note: su m6 e'
        # andata esattamente cosi', best.pt e' l'epoca 4 mentre l'epoca 9 sugli
        # stessi due record di Yujeong fa 0.932/0.940 contro 0.912/0.852. Con
        # `ep{N}.pt` la scelta dell'epoca si puo' rifare a posteriori guardando
        # l'F1, che e' la metrica che ci interessa, invece di essere decisa per
        # sempre dalla loss a meta' training.
        torch.save(model.state_dict(), out_dir / f"ep{epoch}.pt")
        if metrics["val_onset_loss"] < best:
            best = metrics["val_onset_loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  nuovo migliore (onset {best:.4f}) -> best.pt", flush=True)

    print(f"fatto: pesi in {out_dir}")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    main()

