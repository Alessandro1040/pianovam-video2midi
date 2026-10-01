#!/bin/bash
# Training "m6": stessa ricetta di m5_2ch, cambia solo la quantita' di dati
# (7,5 -> 219 minuti di video, 3 -> 6 pianisti). Serve a rispondere a una domanda
# sola: quanto dell'F1 basso su un pianista nuovo era mancanza di dati?
#
# La cache usata e' data/cache; train.py prende da li' i record con split
# train/ext-train (26 record, 788 872 frame) e scarta valid/test. I tre pianisti
# tenuti fuori per il test di generalizzazione stanno in data/holdout e NON
# vengono letti qui.
#
# Lancio consigliato (immunune alla chiusura del terminale):
#   nohup bash scripts/train_m6.sh > logs/train_m6.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 -u scripts/train.py \
  --cache data/cache \
  --out runs/m6_2ch \
  --temporal-radius 15 \
  --epochs 10 \
  --steps-per-epoch 400 \
  --batch-size 8 \
  --clip-frames 64 \
  --lr 2e-3 \
  --val-records 2024-02-15_21-33-36 2024-09-02_12-45-12 \
  --device mps \
  --log-every 50
