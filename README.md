# pianovam-video2midi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/pianovam-quickstart/blob/main/pianovam_video2midi.ipynb)

Da **video di pianoforte** a **MIDI**, usando solo l'immagine: niente audio,
niente sensori, niente tracce dei tasti. Il modello funziona anche se il
pianoforte è inquadrato da un altro punto di vista, perché la geometria viene
normalizzata *prima* della rete.

---

## L'idea in una riga

> L'invarianza a posizione, rotazione e prospettiva **non si fa imparare** al
> modello (servirebbero riprese diverse dello stesso pezzo): la si ottiene per
> costruzione, rettificando la tastiera con i 4 angoli annotati nel dataset.
> Alla rete arriva sempre la stessa immagine canonica: 88 colonne, una per tasto.

Il dataset PianoVAM v1.2 annota per tutte le 107 registrazioni i punti
`Point_LT/RT/RB/LB`, cioè il rettangolo della tastiera. Da quei 4 punti si calcola
l'omografia che porta quella regione in una striscia di 96 × 1056 pixel
(88 tasti × 12 px). Dopo il warp, un tasto premuto sta **sempre nella stessa
colonna**, qualunque sia la posizione della telecamera.

## La pipeline

```
video.mp4 ──► [1] crop sul riquadro della tastiera + omografia prospettica
              ──► [2] striscia canonica 96×1056 in scala di grigi (cache su disco)
              ──► [3] due canali: immagine + differenza dalla linea di base del tasto
              ──► [4] CNN per frame: da ogni striscia a 88 vettori (uno per tasto)
              ──► [5] TCN dilatato sul tempo: contesto di ±31 frame (~1.05 s)
              ──► [6] 4 teste: onset / tasto tenuto / rilascio / velocity
              ──► decodifica a picchi ──► note ──► file .mid
```

| passo | dove | dettaglio |
|---|---|---|
| 1-2 | `src/geometry.py`, `src/video.py` | decodifica **solo del riquadro** della tastiera (il resto del 1080p si butta) |
| 3 | `src/data.py` `temporal_features` | **il canale che fa la differenza**: vedi sotto |
| 4 | `src/model.py` `KeyStem` | `adaptive_avg_pool2d` porta la larghezza esattamente a 88 colonne: la griglia dei tasti non si disallinea mai |
| 5 | `src/model.py` `TemporalBlock` | conv 2D dilatate (dilatazione sul tempo, kernel 3 sui tasti), campo recettivo **63 frame = 1.05 s** |
| 6 | `scripts/train.py` | BCE con `pos_weight` sulle tre teste di evento, MSE mascherata sulla velocity |
| — | `src/decode.py` | picchi locali sopra soglia + soppressione ravvicinata, fine nota dal rilascio |

## Le scelte che contano (e perché)

**Invarianza, tre sorgenti diverse.**
1. *Geometria* — il warp prospettico (sopra). Copre posizione, rotazione,
   inclinazione, prospettiva e zoom.
2. *Specchio sinistra/destra* — durante il training la striscia viene specchiata
   e **le etichette dei tasti invertite con lei** (`tasto i ↔ tasto 87-i`):
   non è un trucco per fare più dati, è la simmetria esatta del problema.
   All'inferenza la vista specchiata viene mediata con quella normale (TTA).
3. *Aspetto* — luminosità, contrasto, gamma, sfocatura, rumore; e un po' di
   *jitter* sui 4 angoli della calibrazione, per non dipendere dalla precisione
   del punto annotato.

**Il secondo canale (e perché esiste).** Un modello che guarda solo l'immagine
rischia di imparare *il pianoforte*, non *il gesto*: nei test un modello a un
canale prende F1 0,68 sul pianista visto in training e crolla a 0,13 su un
pianista mai visto, con un sintomo preciso — la probabilità di onset sui tasti
veri scende da 0,42 a 0,05 (non è un problema di tempo né di tasti: è che
*non vede* le pressioni). Il secondo canale toglie di mezzo quella dipendenza:

```
canale 0 = frame normalizzato
canale 1 = frame - media mobile centrata ±15 frame   (diviso 32 livelli di grigio)
```

Il canale 1 non dice "questo tasto è chiaro", dice **"questo tasto è diverso da
com'era un istante fa"**: è il gesto, non l'aspetto. È l'equivalente, applicato
alla tastiera, del rilevamento del movimento nei video di sorveglianza, e non
costa quasi niente (una somma cumulata, O(1) per frame).

**L'orientamento (tastiera girata di 180°) si deduce dalla geometria**, non dal
modello: i tasti neri sono le colonne scure della fascia alta della striscia, e
il pattern dei neri è leggermente asimmetrico (2+3). `keys.black_key_score` vale
~0,93 quando la striscia è dritta e ~0,6 quando è specchiata: basta confrontare
le due ipotesi.

**Il rilascio è `key_offset`, non `frame_offset`.** Nel dataset `frame_offset`
include il pedale di risonanza, che dall'alto **non si vede**: chiedere al
modello di predirlo sarebbe chiedergli un miracolo. La nota tiene finché il dito
è sul tasto.

**Una testa per la velocity**, ma con peso basso (0,1): dal video si stima male,
e vale la pena solo perché il MIDI suoni meno piatto.

## Stato attuale (misurato, non promesso)

Il punto di partenza è un dataset da **10 registrazioni** (7 di training, 2 di
validazione, 1 di test, ~27 000 frame = 7,5 minuti di video): il dataset completo
sono 45 GB e qui c'è un portatile da 16 GB. Serve come prova che la pipeline
funziona e per capire dove si rompe.

| verifica | risultato | come si controlla |
|---|---|---|
| allineamento griglia tasti | **0.87 - 0.93** di correlazione con i tasti neri su tutte le registrazioni | `scripts/check_calibration.py` |
| campo recettivo | **63 frame (1.05 s)** dichiarato e verificato con un impulso | `tests/test_model.py` |
| test | **41 test** verdi (geometria, target, modello, decodifica, metriche) | `python3 -m pytest tests -q` |
| velocità | ~7× il tempo reale su M5 (178 s di video → 27 s di calcolo, con TTA) | `scripts/predict.py` |

F1 sulle note (tolleranza 50 ms), soglia scelta sulla validazione (0,40):

| registrazione | pianista | visto in training? | 1 canale | **2 canali** (primo training, 7,5 min) |
|---|---|---|---|---|
| 2024-09-02_12-45-12 | Junhyung | **sì** (97 s in ext-train) | 0,686 | **0,738** (P 0,92 R 0,62) |
| 2024-02-15_21-33-36 | Doha | no | 0,447 | 0,441 |
| 2024-02-17_21-44-37 | Changyun | no | 0,128 | **0,143** |
| media validazione | | | 0,559 | **0,574** |

Dietro questi numeri ci sono **7 minuti e mezzo di video di tre pianisti**: 308 s
di jiwoo, 97 s di Junhyung, 45 s di Yonghyun. Il modello è, di fatto, il
trascrittore di quei tre. Il dataset intero offre, nel solo train ufficiale,
**53 registrazioni = 14 ore di video di 5 pianisti** (altri 5 in ext-train): è lì
che sta la differenza fra 0,14 e un numero utilizzabile.

Errore medio sugli onset delle note corrette: **22-28 ms** (1-2 frame).

**Il numero che conta è la differenza fra la prima e l'ultima riga.** Il modello
impara il compito, ma resta attaccato all'aspetto della scena. La diagnosi, con i
numeri:

* sul pianista visto, la probabilità di onset sui tasti veri è **0,42**;
* sul pianista mai visto è **0,05**, eppure la geometria è a posto (0,92 di
  allineamento dei tasti neri), non c'è nessuna deriva temporale (ritardo ottimo
  +2 frame su ±900 provati) e non sono i tasti sbagliati;
* quindi il modello *non vede* le pressioni, non le sbaglia di posizione.

Cosa cambia fra le due scene: le mani. Nelle registrazioni di training sono
chiare e sfocate (un alone sopra i tasti), in quella di test sono dita scure e
nitide su tasti più spenti. Con 7 registrazioni il modello prende la strada più
corta: impara l'aspetto della mano invece del movimento del tasto. Il canale di
differenza è nato per togliere di mezzo quella scorciatoia e si vede che aiuta
(0,686 → 0,738 sul pianista visto, 0,128 → 0,143 sull'altro), ma **non basta**:
servono più registrazioni, ed è esattamente quello che il dataset ha.

Da guardare insieme, questa parte più dei numeri in sé.

### Cosa cambia con 10× i dati (training `m6`, epoca 9)

Il secondo training (`runs/m6_2ch`, 10 epoche, ~174 minuti di video da **6 pianisti**
invece di 7 minuti e mezzo da 3) risponde alla domanda che la tabella qui sopra lascia
aperta. F1 sulle note, tolleranza 50 ms, **tutto il record** (nessun troncamento), soglia
scelta ogni volta sulla **validazione ufficiale** (che contiene solo pianisti visti):

| registrazione | pianista | epoca 4 (`best.pt`, soglia 0,60) | **epoca 9 (`last.pt`, soglia 0,70)** |
|---|---|---|---|
| 2024-02-17_21-44-37 | **Changyun** (mai visto, test ufficiale) | 0,644 | **0,732** (P 0,79 R 0,68) |
| 2024-09-05_20-46-25 | Yujeong (mai visto) | 0,912 | **0,932** |
| 2024-09-05_21-13-49 | Yujeong (mai visto) | 0,852 | **0,940** |
| 2024-02-20_18-08-35 | Taegyun (mai visto) | 0,777 | **0,824** |
| 2024-02-22_11-58-09 | Taegyun (mai visto) | 0,744 | **0,798** |
| 2024-09-04_19-52-57 | Minwook (mai visto) | 0,383 | **0,470** |
| 2024-09-04_21-09-37 | Minwook (mai visto) | 0,416 | **0,478** |
| **aggregato sui 3 tenuti fuori** (20 027 note vere) | | P 0,81 R 0,55 **F1 0,651** | P 0,84 R 0,61 **F1 0,709** |

Errore medio sugli onset delle note corrette: **12,5 ms** su Changyun, **15,4 ms** sul
gruppo tenuto fuori. Tre cose da leggere in questa tabella, in ordine di importanza:

1. **i dati erano il collo di bottiglia, l'architettura no.** Changyun passa da 0,14 (m5,
   7,5 minuti di dati) a 0,73 (m6, 174 minuti): con dieci volte i dati la stessa rete e la
   stessa decodifica diventano utilizzabili;
2. **la loss di validazione si satura prima dell'F1.** `best.pt` è l'epoca 4 perché il
   criterio in `train.py` guarda la loss posteriore di validazione — ma l'F1 sulle **note**
   continua a salire fino all'epoca 9, a ogni soglia, sia sui pianisti visti sia su quelli
   tenuti fuori (sul valid ufficiale 0,823 → 0,870 a soglia 0,70). La loss è dominata dalle
   celle negative facili; l'F1 dipende solo dalle celle di onset. Il criterio giusto è
   **l'F1 di validazione** — calcolato sui pianisti visti, senza guardare né Yujeong né il
   holdout — ed è quello che il notebook Colab usa per scegliere i pesi;
3. **resta un buco, e non è di calibrazione.** Su Minwook (musica densa e veloce) il modello
   è preciso ma scrive un terzo delle note: P 0,87 R 0,32. Abbassando la soglia a 0,30
   recupera ~+0,09 di F1 ma la precision scende a 0,55. Lì limita la **capacità** del
   modello, non la soglia: è il motivo per cui la tabella tiene separati i pianisti invece
   di mediare tutto insieme. Sul pianista visto, invece, il modello *sovra*-predice gli
   onset di 6,5× (colpa del `pos_weight`); su Minwook solo di 1,7×, e la stessa soglia fissa
   taglia troppo.




## Cosa c'è nei file

```
src/
  keys.py        layout degli 88 tasti, tasti neri, punteggio di allineamento, orientamento
  geometry.py    omografia dai 4 punti annotati, jitter di calibrazione, crop del riquadro
  video.py       video -> cache di strisce rettificate (blocchi da 512 frame)
  targets.py     TSV -> onset / tasto tenuto / rilascio / velocity
  data.py        dataset a finestre + aumentazioni (specchio con inversione dei tasti)
  model.py       KeyStem (CNN per frame) + TCN dilatato + 4 teste
  decode.py      picchi -> note -> MIDI
  metrics.py     F1 sulle note con tolleranza 50 ms, F1 sullo stato del tasto
  infer.py       inferenza su registrazione intera, con TTA e finestre sovrapposte
  manifest.py    metadata.json, split ufficiali, calibrazione per registrazione
scripts/
  fetch.py               scarica da Hugging Face solo cio' che serve
  check_calibration.py   M0: i tasti sono allineati? (controllo numerico + PNG da guardare)
  build_cache.py         costruisce la cache delle strisce
  train.py               training
  evaluate.py            F1 sulle note + scelta della soglia su validazione
  predict.py             video -> MIDI, end to end
tests/                   37 test (geometria, target, modello, decodifica, metriche)
```

## Uso, dal principio

```bash
# 0) dipendenze: torch, av, opencv-python, scipy, pretty_midi, huggingface_hub, pytest

# 1) dati: metadati/TSV/MIDI di tutto il dataset (pochi MB)...
python3 scripts/fetch.py --all-text
# ...poi i video un po' alla volta: scarica, costruisce la cache e cancella il video
#    (la selezione gira fra i pianisti, non "i piu' corti")
python3 scripts/harvest.py --per-player 5 --max-minutes 240 --min-record-minutes 2 \
  --exclude-players Taegyun Minwook Yujeong
#    e gli stessi pianisti tenuti fuori, in una cache separata, per il test onesto
python3 scripts/harvest.py --cache data/holdout --per-player 2 --only-players Taegyun Minwook Yujeong

# 2) controllo della calibrazione (da guardare i PNG in out/calibration/)
python3 scripts/check_calibration.py --record 2024-02-17_21-37-57

# 3) (opzionale) la cache si costruisce dentro harvest.py; build_cache.py serve
#    se si hanno gia' i video su disco
python3 scripts/build_cache.py

# 4) training m6 (misurato: ~2,7 s per passo; un'epoca da 400 passi ~18 minuti,
#    10 epoche ~3 ore su M5 senza GPU). La ricetta sta in scripts/train_m6.sh,
#    e cosi' non dipende dal terminale: nohup bash scripts/train_m6.sh
#    > logs/train_m6.log 2>&1 &
bash scripts/train_m6.sh

# 5) valutazione onesta: la soglia si scegle sulla validazione ufficiale, poi si
#    misura sui tre pianisti tenuti fuori (che la soglia non l'hanno toccata)
python3 scripts/evaluate.py --checkpoint runs/m6_2ch/best.pt --out runs/m6_2ch/eval_full.json
python3 scripts/evaluate.py --checkpoint runs/m6_2ch/best.pt --cache-extra data/holdout \
  --test-records 2024-02-20_18-08-35 2024-02-22_11-58-09 2024-09-04_19-52-57 \
                2024-09-04_21-09-37 2024-09-05_20-46-25 2024-09-05_21-13-49 \
  --out runs/m6_2ch/eval_holdout.json

# 6) video -> MIDI (usa la soglia trovata al passo 5)
python3 scripts/predict.py --record 2024-02-17_21-44-37 --out out/pezzo.mid
```

I pesi da usare sono `runs/m6_2ch/last.pt` (**epoca 9**, due canali, ~174 minuti di
video da 6 pianisti); `runs/m6_2ch/best.pt` è l'epoca 4, quella scelta dal criterio
vecchio (loss posteriore di validazione) e le sta dietro di ~0,06 di F1 sui pianisti
tenuti fuori. `runs/m5_2ch/best.pt` è lo stesso modello con 7,5 minuti di video da 3
pianisti e `runs/baseline_1ch/best.pt` è la versione a un canale, cioè i due
termini di paragone. I report stanno in `runs/*/eval_*.json`. Un esempio di
uscita già pronto è `out/test_changyun_2ch.mid` (178 s di video trascritti in 27 s).

**Attenzione alla metrica prima di leggere un numero**: dopo questa raccolta lo
split `valid` contiene due pianisti che il modello *ha* visto (Doha e Junhyung),
quindi il suo F1 non dice niente sulla generalizzazione. I tre numeri che contano
sono i pianisti tenuti fuori (Taegyun, Minwook, Yujeong) e Changyun (3 minuti,
l'unico mai visto in nessun training del dataset). Il codice di valutazione
conta la cache, non i video (`manifest.cached_records`), perché i video vengono
cancellati dopo la raccolta.

Su un **video tuo** servono solo i 4 angoli della tastiera (in ordine
sinistra-alto, destra-alto, destra-basso, sinistra-basso):

```bash
python3 scripts/predict.py --video mio_video.mp4 \
  --quad "121,355 1839,345 1839,558 120,564" --out out/mio.mid
```

Se la tastiera è inquadrata "al contrario", non serve dirlo: `predict.py`
confronta il punteggio dei tasti neri nelle due ipotesi e la raddrizza da solo
(`--orient auto`, oppure `normal`/`flip` per forzare).


## Come si allarga davvero: dati, non architettura

Quello che limita non è il modello (285 564 parametri) ma **quanti pianisti ha
visto**. Il primo training è stato fatto su 7 registrazioni scelte come "le più
corte per split", e il risultato è stato: **7,5 minuti di video da 3 pianisti**.
Con quella base il modello impara la mano di quelle tre persone e su un pianista
nuovo la probabilità di onset sui tasti veri scende da 0,42 a 0,05.

I numeri del dataset (misurati con `HfApi` sul repo, non stimati):

| split | durata | pianisti |
|---|---|---|
| train | 838 min | Yonghyun 455, Taegyun 187, Doha 116, Junhyung 41, Jiyun 39 |
| ext-train | 166 min | Junhyung 66, Minwook 55, Yujeong 18, Hyunsung 14, jiwoo 13 |
| valid | 91 min | Yonghyun, Yujeong, Doha, Junhyung |
| test | 101 min | Junhyung 50, Yonghyun 34, Doha 14, **Changyun 3** |

**Il dettaglio che cambia tutto**: nei due split di valutazione ufficiali 4
pianisti su 4 hanno registrazioni anche in training. L'unico pianista mai visto
da nessuno split di training è **Changyun, 3 minuti** (ed è il record su cui il
modello segna F1 0,14). Quindi gli split ufficiali **non misurano** la
generalizzazione a un pianista nuovo: per misurarla bisogna costruirsi il test.

### Protocollo: i pianisti tenuti fuori

`scripts/harvest.py` scarica un video, costruisce la cache e **cancella il
video**: la cache costa ~1,7 MB per minuto contro ~20 MB per minuto di video, e
su un disco da 4 GB liberi questo è quello che rende possibile raccogliere
qualche ora invece di qualche minuto. La selezione gira **fra i pianisti**
(`--per-player`), non "i più corti": è l'errore che aveva prodotto i 7,5 minuti.

```bash
# training: 6 pianisti, tenendo fuori Taegyun, Minwook e Yujeong
python3 scripts/harvest.py --per-player 5 --max-minutes 240 --min-record-minutes 2 \
  --exclude-players Taegyun Minwook Yujeong --sizes-json sizes.json
# test di generalizzazione: gli stessi 3 pianisti, mai visti in training
# (in una cache separata: cosi' non possono finire dentro il training per sbaglio)
python3 scripts/harvest.py --cache data/holdout --per-player 2 --only-players Taegyun Minwook Yujeong
# valutazione: la cache dei tenuti-da-parte si passa a parte
python3 scripts/evaluate.py --checkpoint runs/m6_2ch/best.pt \
  --records <record_taegyun> <record_minwook> --cache-extra data/holdout
```

Il numero da guardare non è più l'F1 sullo split valid (dopo questa raccolta
contiene solo pianisti visti), ma l'F1 **sui tre pianisti tenuti fuori** e sul
record di Changyun.

## Provarlo in Colab (esempio del dataset, o un video tuo)

Il notebook è [`pianovam_video2midi.ipynb`](https://github.com/Alessandro1040/pianovam-quickstart/blob/main/pianovam_video2midi.ipynb),
nello stesso repo pubblico del notebook del dataset (il badge **Open in Colab** è in cima a
questa pagina). Scegli un esempio con il suo indice (1..107): guardi il **video con l'audio
originale** e confronti il **MIDI registrato dal pianoforte** con quello **stimato dal
modello** — P/R/F1 a 50 ms, piano roll dei due MIDI uno sopra l'altro, e ascolto A/B
sintetizzato con lo stesso strumento, così l'unica differenza che senti sono le note che
il modello prende e quelle che si perde. Con `MODE = "upload"` carichi un video tuo:
niente riferimento con cui confrontarsi, ma il MIDI stimato esce lo stesso.

Al codice e ai pesi il notebook ci arriva da solo: **clona questo repo** (è pubblico, ~6 MB con
dentro `src/` e i pesi dell'epoca 9). Se il clone non passa ripiega su uno zip di ~1 MB, quello
di `python3 scripts/pack_colab_bundle.py`; se hai già la cartella aperta in locale la usa e basta.

Per un video tuo il notebook chiede i 4 angoli della tastiera **a mano**, leggendoli
su una griglia disegnata sopra il primo frame: il rilevamento automatico (sezione
successiva) non è ancora abbastanza preciso per fidarsene.

Due avvertenze pratiche: il video di una registrazione pesa ~340 MB e il notebook elabora i
primi `SECONDS` secondi (60 di default) — su una registrazione intera i tempi diventano
minuti, non secondi; e il modello non ha mai visto inquadrature diverse da quelle del
dataset, quindi su un video con un'altra angolazione aspettati molto meno.

## Trovare la tastiera da soli (sperimentale, non ancora affidabile)

Per un video del dataset i 4 angoli sono annotati. Per un video qualsiasi
servirebbe trovarli: `src/detect.py` ci prova *senza* addestrare niente, usando
come funzione obiettivo il disegno periodico dei tasti neri (`black_key_score`),
e `scripts/detect_keyboard.py` mostra un'anteprima col riquadro trovato.

Misurato su frame reali (3 registrazioni): **sbaglia di 13–21 tasti**, quindi
oggi non è utilizzabile per la trascrizione. La diagnosi è nei test
(`tests/test_detect.py`, il test marcato `xfail`):

* la struttura della striscia è stata **misurata** sulle cache (righe 0–26 senza
  disegno, 27–71 con i tasti neri, 72–95 di nuovo senza) e l'obiettivo costruito
  su quella misura riconosce il rettangolo giusto (0,56–0,87) mentre il primo
  tentativo — che chiedeva "sopra la tastiera deve essere uniforme" — gli dava
  punteggio **negativo** (sopra c'è il pianista, non una parete);
* resta il problema della **scala**: con 88 tasti la larghezza va azzeccata allo
  0,5%, altrimenti l'ultimo tasto si sposta di un tasto intero; una griglia
  grossolana non ha quella precisione, e il massimo globale del punteggio su un
  video vero **non è** il rettangolo giusto (0,95 su una zona sbagliata contro
  0,56 su quella giusta).

Il passo che serve è stimare il **periodo dei tasti** nell'immagine
(autocorrelazione del profilo dei bordi verticali, poi comb-fit su 88 posizioni)
invece di sperare che una griglia lo indovini. Finché non c'è, il modulo resta
marcato sperimentale e il percorso supportato è `--quad` con i 4 angoli.

## Limiti noti (onesti)

* **La tastiera va indicata**: 4 angoli a mano (`--quad`) o l'annotazione del
  dataset. La ricerca automatica esiste ma non è affidabile (sopra).
* **Pedale invisibile**: dall'alto non si vede, quindi nel MIDI non c'è. Le note
  si accorciano quando il dito lascia il tasto, non quando si alza il pedale.
* **Velocity grossolana**: stimata dal video, peso 0,1 nella perdita: vale come
  "piano/forte", non come dinamica fine. L'audio del dataset (che qui non si usa
  di proposito) la stimerebbe molto meglio.
* **Blur**: 7 registrazioni del dataset hanno tastiera o mani sfocate; non sono
  nella selezione attuale e vanno valutate a parte prima di dire che il modello
  "funziona su tutto".
* **Due mani**: la testa non distingue mano destra e sinistra; con le due mani
  sovrapposte sullo stesso tasto il modello vede un tasto solo.
* **Scala di grigi**: si perde il colore (non serve per i tasti, aiuterebbe
  invece a separare la pelle dal legno: da provare in futuro).

## Cosa farei come passo successivo

1. **Più registrazioni**: è il moltiplicatore più grande e costa solo disco.
2. **Ponderare la perdita per tasto**: gli errori sulle ottave basse contano di più.
3. **Velocity dall'audio**: è l'unico pezzo che il video non può dare bene, e il
   dataset ha l'audio allineato.
4. **Soglie per tasto** invece che una sola globale, se dalla validazione emerge
   un bias sistematico (per esempio sui tasti neri coperti dalle dita).
