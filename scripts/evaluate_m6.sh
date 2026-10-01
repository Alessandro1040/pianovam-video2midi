#!/bin/bash
# Valutazione finale di m6, in due parti che rispondono a due domande diverse:
#
#   1. eval_full.json   -> split ufficiale. Il test e' il record di Changyun, che
#      e' l'unico pianista mai visto da nessun training del dataset: e' l'unico
#      numero confrontabile alla pari con m5_2ch (0.143) e baseline_1ch (0.128).
#   2. eval_holdout.json -> i sei record dei tre pianisti tenuti fuori a mano
#      (Taegyun, Minwook, Yujeong), presi da data/holdout. La soglia resta quella
#      scelta sulla valid ufficiale, cosi' i tenuti-fuori non toccano nulla.
#
# Le due parti girano anche se una delle due fallisce: interessa avere un
# risultato, non che il secondo comando non parta.
#
# Griglia di soglie estesa fino a 0.90 rispetto al default (che si ferma a 0.60):
# nella valutazione intermedia la soglia scelta e' stata proprio 0.60, cioe' il
# bordo della griglia, mentre il modello predice il 6,7x la frequenza vera di
# onset. Con il bordo toccato non si sa se l'ottimo sia dentro o fuori, quindi si
# allarga. Non e' una forzatura del protocollo: la soglia continua a essere
# scelta SOLO sulle due registrazioni di validazione ufficiale.
GRID="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"

cd "$(dirname "$0")/.."

python3 -u scripts/evaluate.py \
  --checkpoint runs/m6_2ch/best.pt \
  --thresholds "$GRID" \
  --out runs/m6_2ch/eval_full.json
echo "== eval_full concluso (exit $?) =="

python3 -u scripts/evaluate.py \
  --checkpoint runs/m6_2ch/best.pt \
  --cache-extra data/holdout \
  --thresholds "$GRID" \
  --test-records 2024-02-20_18-08-35 2024-02-22_11-58-09 2024-09-04_19-52-57 \
                 2024-09-04_21-09-37 2024-09-05_20-46-25 2024-09-05_21-13-49 \
  --out runs/m6_2ch/eval_holdout.json
echo "== eval_holdout concluso (exit $?) =="

# 3. Diagnostica: gli stessi due record di Yujeong, ma con `last.pt` (epoca 9)
#    invece che con `best.pt` (epoca 4, scelta in base alla val ufficiale).
#    Serve a rispondere a una domanda precisa: il criterio di scelta del
#    checkpoint (onset loss su pianisti GIA' visti) sta scegliendo l'epoca
#    giusta per i pianisti NUOVI? Questo risultato NON e' un numero da
#    riportare come principale: scegliere l'epoca guardando la holdout
#    sarebbe proprio il trucco che vogliamo evitare. E' solo un controllo.
python3 -u scripts/evaluate.py \
  --checkpoint runs/m6_2ch/last.pt \
  --cache-extra data/holdout \
  --thresholds "$GRID" \
  --test-records 2024-09-05_20-46-25 2024-09-05_21-13-49 \
  --out runs/m6_2ch/eval_last_ep9_diagnostica.json
echo "== diagnostica last.pt conclusa (exit $?) =="
