"""Metriche: confronto fra note predette e note del dataset.

La metrica di riferimento per la trascrizione e' il match di eventi con
tolleranza: due note della stessa altezza si considerano la stessa nota se i
loro onset distano meno di `tolerance_frames` (3 frame a 60 fps = 50 ms).
Per un solo asse (il tempo) il matching ottimo si ottiene con una scansione
greedy in ordine: si prende sempre la coppia piu' vicina possibile.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TOLERANCE = 3          # frame


def _match_one_key(gt: np.ndarray, pred: np.ndarray, tolerance: int) -> tuple[int, float]:
    """Greedy ottimo su un asse: restituisce (numero di match, errore medio assoluto in frame)."""
    gt = np.sort(np.asarray(gt, dtype=np.int64))
    pred = np.sort(np.asarray(pred, dtype=np.int64))
    i = j = matched = 0
    errors: list[int] = []
    while i < gt.size and j < pred.size:
        delta = pred[j] - gt[i]
        if abs(delta) <= tolerance:
            matched += 1
            errors.append(abs(int(delta)))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return matched, (float(np.mean(errors)) if errors else float("nan"))


def match_events(
    gt_frames: np.ndarray,
    gt_keys: np.ndarray,
    pred_frames: np.ndarray,
    pred_keys: np.ndarray,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict:
    """Eventi (frame, tasto) matchati per tasto, con tolleranza temporale."""
    gt_frames = np.asarray(gt_frames, dtype=np.int64)
    pred_frames = np.asarray(pred_frames, dtype=np.int64)
    gt_keys = np.asarray(gt_keys, dtype=np.int64)
    pred_keys = np.asarray(pred_keys, dtype=np.int64)

    matched = 0
    weighted_error = 0.0
    counted = 0
    for key_index in np.unique(np.concatenate([gt_keys, pred_keys])) if gt_keys.size or pred_keys.size else []:
        n, mean_error = _match_one_key(
            gt_frames[gt_keys == key_index], pred_frames[pred_keys == key_index], tolerance
        )
        matched += n
        if n and not np.isnan(mean_error):
            weighted_error += mean_error * n
            counted += n

    precision = matched / pred_frames.size if pred_frames.size else 0.0
    recall = matched / gt_frames.size if gt_frames.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_gt": int(gt_frames.size),
        "n_pred": int(pred_frames.size),
        "matched": int(matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_onset_error_ms": float(weighted_error / counted * 1000.0 / 60.0) if counted else float("nan"),
    }


def state_metrics(gt_state: np.ndarray, pred_state: np.ndarray) -> dict:
    """F1 a livello di frame sullo stato del tasto (matrici booleane (T, 88))."""
    gt = np.asarray(gt_state, dtype=bool)
    pred = np.asarray(pred_state, dtype=bool)
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def note_metrics(gt_events, pred_notes, tolerance: int = DEFAULT_TOLERANCE) -> dict:
    """Comodita': eventi del dataset contro note decodificate."""
    pred_frames = np.array([n.start_frame for n in pred_notes], dtype=np.int64)
    pred_keys = np.array([n.key for n in pred_notes], dtype=np.int64)
    return match_events(
        gt_events.onset_frame, gt_events.key, pred_frames, pred_keys, tolerance=tolerance
    )
