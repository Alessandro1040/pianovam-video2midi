"""Geometria dei tasti: layout 88 tasti, tasti neri, rilevamento dell'orientamento.

Il modello non vede mai la tastiera "grezza": la vede sempre rettificata in una
striscia canonica di 88 colonne (una per tasto). Questo file definisce quella
griglia e le proprieta' invarianti che ne derivano.

Quando invece la tastiera arriva da un video **non annotato** la griglia non e' nota
(il quadrilatero puo' coprire 20 tasti, non 88): `estimate_key_span` misura il
*periodo* dei tasti direttamente nel profilo della banda, invece di presumere che ne
stiano 88 dentro la striscia. E' la differenza che rende leggibile un video a mano
libera: raggruppare un primo piano in 88 colonne fa crollare il punteggio dei tasti
neri a zero, e da li' in poi tutto il resto sbaglia scala.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

N_KEYS = 88          # pianoforte standard: A0 (MIDI 21) .. C8 (MIDI 108)
MIDI_MIN = 21
MIDI_MAX = 108

# classi di altezza dei tasti neri (Do# Re# Fa# Sol# La#)
BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})

# larghezza in pixel di una colonna-tasto nella striscia canonica
KEY_PX = 12
STRIP_W = N_KEYS * KEY_PX
STRIP_H = 96


def is_black(index: int) -> bool:
    """True se il tasto i-esimo (0 = A0) e' nero."""
    return (index + MIDI_MIN) % 12 in BLACK_PITCH_CLASSES


def black_mask() -> np.ndarray:
    """Maschera booleana (88,) dei tasti neri."""
    return np.array([is_black(i) for i in range(N_KEYS)], dtype=bool)


def note_name(note: int) -> str:
    """60 -> 'C4'."""
    return "C C# D D# E F F# G G# A A# B".split()[note % 12] + str(note // 12 - 1)


def note_to_index(note: int) -> int:
    """Numero MIDI -> indice di colonna 0..87."""
    return int(note) - MIDI_MIN


def index_to_note(index: int) -> int:
    """Indice di colonna 0..87 -> numero MIDI."""
    return int(index) + MIDI_MIN


def in_range(index: int) -> bool:
    return 0 <= index < N_KEYS


def black_profile() -> np.ndarray:
    """Profilo di riferimento atteso per l'asse dei tasti.

    Ogni tasto nero e' una colonna "scura" nella parte alta della tastiera,
    quindi il profilo e' 1 sui tasti neri e 0 sui bianchi. Serve a stimare
    l'orientamento (specchio / 180 gradi) di una striscia rettificata.
    """
    return black_mask().astype(np.float32)


# Dove stanno i tasti neri dentro la striscia. Le frazioni non sono inventate: misurate
# sulle strisce in cache (righe 0-26 piatte, 27-71 col disegno dei neri, 72-95 piatte), e
# sono le stesse che usa `detect.bands`: se le due definizioni divergessero, le soglie
# misurate su un video non varrebbero piu' sull'altro.
BAND_MARGIN = 0.4        # quota di altezza sopra la tastiera
BAND_BLACK_FRAC = 0.64   # quota dei tasti occupata dai tasti neri


def black_rows(
    out_h: int = STRIP_H, margin: float = BAND_MARGIN, black_frac: float = BAND_BLACK_FRAC
) -> slice:
    """Le righe della striscia occupate dai tasti neri (le punte)."""
    top = int(round(out_h * margin / (1.0 + margin)))
    return slice(top, top + int(round((out_h - top) * black_frac)))


def black_key_band(strip: np.ndarray, rows: slice | None = None) -> np.ndarray:
    """Luminosita' media per colonna della fascia dei tasti neri, a risoluzione di pixel.

    E' il profilo su cui si misura il passo dei tasti: a risoluzione di pixel, non
    raggruppato in 88 colonne, altrimenti la misura presupporrebbe la risposta che cerca.
    `rows` sceglie la fascia (default: `black_rows`).
    """
    band = strip[black_rows(strip.shape[0]) if rows is None else rows]
    return band.astype(np.float64).mean(axis=0)


def black_key_score(strip: np.ndarray, row_frac: float = 0.55, lo: int = 0, hi: int = N_KEYS) -> float:
    """Quanto la striscia assomiglia a una tastiera con la griglia dei tasti giusta.

    I tasti neri sono le colonne piu' scure della fascia alta della tastiera:
    se la rettifica e' corretta, il profilo di luminosita' per colonna e'
    correlato con il pattern dei neri (correlazione vicina a 1). Se la tastiera
    e' specchiata, la correlazione e' negativa, quindi questo stesso numero
    serve anche a stimare l'orientamento.

    Presuppone che la striscia sia larga `N_KEYS` tasti (e' cosi' che il modello la
    vuole). Se l'inquadratura e' stretta il raggruppamento in 88 colonne mescola i
    tasti veri e il numero perde senso: per sapere *quanti* tasti ci sono davvero
    serve `estimate_key_span` sulla banda grezza (`black_key_band`), non questo.

    `row_frac` sceglie quante righe in alto guardare: `1.0` significa "tutte", ed e'
    quello che passa `detect.pattern_presence`, che gli consegna gia' la fascia dei
    neri ritagliata da `detect.bands`.
    """
    band = strip[: max(1, int(strip.shape[0] * row_frac))].astype(np.float32)
    profile = band.mean(axis=0).reshape(N_KEYS, -1).mean(axis=1)      # una media per colonna-tasto
    observed = -(profile - profile.mean())                            # scuro = positivo
    expected = black_profile() * 2.0 - 1.0                            # nero = +1, bianco = -1
    if observed.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(observed[lo:hi], expected[lo:hi])[0, 1])


# Un tasto nero e' largo 13.7 mm e un tasto bianco 23.5 mm: il semitono (la colonna
# della griglia uniforme) e' largo esattamente quanto un tasto nero, quindi un tasto
# bianco vale 23.5/13.7 = 1.71 semitoni e 7 tasti bianchi = 12 semitoni.
BLACK_OVER_WHITE = 13.7 / 23.5


def black_centres() -> np.ndarray:
    """Dove stanno i tasti neri dentro un'ottava, in unita' di tasto bianco.

    Le classi di altezza dei neri sono anche la loro colonna dentro l'ottava
    (do# = 1, re# = 3, fa# = 6, sol# = 8, la# = 10 semitoni dopo il do): cinque per
    ottava, *non sei*, e sbagliare questo elenco rende inutile la misura del passo.
    """
    return BLACK_OVER_WHITE * np.array(sorted(BLACK_PITCH_CLASSES), dtype=np.float64)


def key_model(
    width_white: float,
    n_samples: int,
    phase: float = 0.0,
    sub: int = 4,
    black_width: float = BLACK_OVER_WHITE,
) -> np.ndarray:
    """Frazione di tasto nero vista da ogni colonna di una banda modello.

    `width_white` e' la larghezza di un tasto bianco in colonne della banda, `phase`
    e' di quanto il disegno e' spostato rispetto all'inizio della banda. Il disegno si
    ripete ogni 7 tasti bianchi (12 semitoni) e ogni tasto nero e' largo `black_width`
    tasti bianchi: la larghezza e' un parametro perche' sulle immagini l'ombra rende i
    neri un po' piu' larghi del valore fisico.
    """
    period = 7.0 * width_white
    x = (np.arange(n_samples * sub, dtype=np.float64) / sub + phase) % period
    half = 0.5 * black_width * width_white
    dark = np.zeros(x.shape, dtype=bool)
    for centre in black_centres():
        dark |= np.abs(x - centre * width_white) <= half
    return dark.reshape(n_samples, sub).mean(axis=1)


def _otsu(values: np.ndarray) -> float:
    """Soglia di Otsu (0..255) di un profilo di luminosita', in numpy puro."""
    counts = np.histogram(np.clip(values, 0.0, 255.0), bins=256, range=(0.0, 255.0))[0]
    counts = counts.astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 128.0
    levels = np.arange(256, dtype=np.float64)
    weight = np.cumsum(counts)
    moment = np.cumsum(counts * levels)
    best, threshold = -1.0, 128.0
    for cut in range(255):
        low, high = float(weight[cut]), total - float(weight[cut])
        if low <= 0.0 or high <= 0.0:
            continue
        between = low * high * (moment[cut] / low - (moment[-1] - moment[cut]) / high) ** 2
        if between > best:
            best, threshold = between, float(cut)
    return threshold


def _runs(mask: np.ndarray) -> int:
    """Numero di regioni non vuote contigue in una maschera booleana."""
    flat = np.ravel(np.asarray(mask, dtype=bool))
    if flat.size == 0:
        return 0
    return int(np.count_nonzero(flat & ~np.concatenate([[False], flat[:-1]])))


@dataclass(frozen=True)
class KeySpan:
    """Quanti tasti copre una banda, con quanta confidenza e con che sfasamento.

    `span` sono i tasti coperti, `score` e' la correlazione col disegno corretta per
    il numero di tasti neri visti, `phase` e' lo sfasamento del disegno in colonne
    della banda (serve a sapere *dove* cadono i tasti, non solo quanti sono) e
    `flipped` dice se conviene leggere la banda specchiata. Uno `score` negativo
    significa che nella banda il disegno dei tasti non c'e': la stima va scartata.

    `flipped` non e' una risposta: il disegno dei neri specchiato e' il disegno stesso
    spostato di 8 semitoni, quindi da una sola banda l'orientamento e' indecidibile e
    la stima lo sceglie solo in base ai bordi. Chi ha bisogno dell'orientamento vero ha
    bisogno degli estremi della tastiera (`detect_input_flip`).
    """

    span: float
    score: float
    phase: float
    flipped: bool

    @property
    def trustworthy(self) -> bool:
        """La stima e' utilizzabile?

        La soglia separa "nessun disegno di tasti" (0.0 - 0.46 nei casi di prova) da
        "disegno trovato" (0.70 - 0.91). Pero' **non basta**: su un video reale a mano
        libera un frame in cui il quadrilatero e' ormai uscito dalla tastiera ha reso 6
        tasti con 0.66, indistinguibile da un frame buono. Una singola misura dice solo
        che c'e' un disegno periodico, non che sia quello giusto: per usarla serve il
        consenso fra piu' frame (`consensus_span`).
        """
        return self.span > 0.0 and self.score >= 0.6


# Tasti coperti da provare quando non si sa nulla: 4..39 uno per uno, poi a due a due.
# Sotto i 4 tasti il disegno dei neri spiega troppo poco per essere credibile.
SPAN_CANDIDATES = np.concatenate([np.arange(4.0, 40.0), np.arange(40.0, 90.0, 2.0)])


def estimate_key_span(
    profile: np.ndarray,
    lo: int = 0,
    hi: int = N_KEYS,
    candidates: np.ndarray | None = None,
    phases: int = 64,
) -> KeySpan:
    """Stima quanti tasti copre una banda, invece di presumere che ne copra 88.

    `profile` e' la luminosita' media per colonna della banda dei tasti neri, a
    **risoluzione di pixel**: raggrupparla in 88 colonne (come fa `black_key_score`)
    presuppone la risposta che qui si vuole trovare. `lo`/`hi` sono la finestra di
    contenuto in colonne canoniche: le colonne fuori dalla finestra sono fuori
    dall'inquadratura.

    Per ogni candidato si costruisce il disegno dei tasti neri, lo si fa scorrere in
    fase (e nei due versi) e lo si correla col profilo: vince il candidato che spiega
    il disegno *con il numero giusto di tasti neri*, perche' un modello con pochi
    tasti neri si accorda per caso con la tendenza lenta della luminosita' (ed e' il
    motivo per cui una correlazione pura sceglierebbe sempre "pochi tasti"). Il costo
    e' qualche decimo di secondo: va bene per misurare, non per ogni frame -- per
    inseguire basta passare `candidates` attorno alla stima precedente.
    """
    x0, x1 = int(lo) * KEY_PX, int(hi) * KEY_PX
    band = np.asarray(profile, dtype=np.float64)[x0:x1]
    if band.size < 2 * KEY_PX or float(band.std()) < 1e-6:
        return KeySpan(0.0, -1.0, 0.0, False)

    dark = band <= _otsu(band)
    blobs = max(1, _runs(dark))
    # I tasti neri coprono 5/7 di un tasto bianco largo: la frazione scura dice quanto
    # e' largo un nero su *questa* immagine (l'ombra lo allarga oltre il valore fisico).
    black_width = BLACK_OVER_WHITE * min(1.0, 2.4 * float(dark.mean()))
    observed = band.mean() - band
    observed = observed / (observed.std() + 1e-12)
    length = band.size

    best = KeySpan(0.0, -1.0, 0.0, False)
    grid = SPAN_CANDIDATES if candidates is None else np.asarray(candidates, dtype=np.float64)
    for span in grid:
        if span <= 0.0:
            continue
        width_white = (length / float(span)) / BLACK_OVER_WHITE
        step = (7.0 * width_white) / max(1, int(phases))
        for flipped in (False, True):
            view = observed[::-1] if flipped else observed
            for phase in np.arange(0.0, 7.0 * width_white, step):
                model = key_model(width_white, length, float(phase), black_width=black_width)
                spread = float(model.std())
                if spread < 1e-9:
                    continue
                score = float(np.dot(view, model - model.mean()) / (length * spread))
                modelled = _runs(key_model(width_white, length, float(phase), sub=8,
                                           black_width=black_width) > 0.5)
                if modelled <= 0:
                    continue
                score *= min(modelled, blobs) / float(max(modelled, blobs))
                if score > best.score:
                    best = KeySpan(float(span), score, float(phase), flipped)
    return best


def white_key_width(span: float, lo: int = 0, hi: int = N_KEYS) -> float:
    """Larghezza di un tasto bianco in colonne della striscia, dalla stima del passo."""
    return ((hi - lo) * KEY_PX / max(float(span), 1e-9)) / BLACK_OVER_WHITE


def consensus_span(spans: Sequence[KeySpan], tolerance: float = 2.0) -> KeySpan:
    """La misura che piu' misure confermano, invece della media di tutte.

    Su un video la stessa tastiera viene misurata decine di volte: se ne scarti una
    diversa dalle altre e' quasi sempre perche' il quadrilatero e' finito fuori dalla
    tastiera (camera che si muove, mano che copre), e *mediarla* con le buone
    rovinerebbe la stima. Si cerca quindi il gruppo piu' numeroso di stime che
    concordano entro `tolerance` tasti -- a parita' di numero vince quello con i
    punteggi piu' alti -- e si restituisce la mediana del gruppo: due tasti di
    differenza fra un frame e l'altro sono normali (un tasto nero sul bordo), venti no.

    Il risultato e' una stima di consenso: `score` e' la mediana dei punteggi del
    gruppo, `phase`/`flipped` vengono dal membro con il punteggio piu' alto.
    """
    usable = [s for s in spans if s.score > 0.0 and s.span > 0.0]
    if not usable:
        return KeySpan(0.0, -1.0, 0.0, False)

    best: tuple[int, float] | None = None
    group: list[KeySpan] = []
    for candidate in usable:
        neighbours = [s for s in usable if abs(s.span - candidate.span) <= tolerance]
        weight = (len(neighbours), float(sum(s.score for s in neighbours)))
        if best is None or weight > best:
            best, group = weight, neighbours
    leader = max(group, key=lambda s: s.score)
    return KeySpan(
        float(np.median([s.span for s in group])),
        float(np.median([s.score for s in group])),
        leader.phase,
        leader.flipped,
    )


def detect_input_flip(strips: np.ndarray) -> bool:
    """True se le strisce sembrano specchiate (tastiera "al contrario").

    Confronta il punteggio dei tasti neri normale e specchiato: e' un test
    geometrico, non serve allenare niente e non costa niente.
    """
    normal = float(np.mean([black_key_score(frame) for frame in strips]))
    flipped = float(np.mean([black_key_score(frame[:, ::-1]) for frame in strips]))
    return flipped > normal


def estimate_horizontal_flip(strip: np.ndarray, row_frac: float = 0.55) -> float:
    """Quanto la striscia sembra specchiata, da -1 (normale) a +1 (specchiata).

    Si guarda solo il "fondo" dei tasti (dove i tasti neri finiscono, cioe' la
    parte piu' interna della tastiera) e si confronta la luminosita' media
    normalizzata con il profilo atteso dei tasti neri e con il suo specchio.
    """
    h, w = strip.shape[:2]
    # i tasti neri sono visibili nella fascia alta della tastiera rettificata
    band = strip[: max(1, int(h * row_frac * 0.5)), :].astype(np.float32)
    column = band.mean(axis=0)
    if column.std() < 1e-6:
        return 0.0
    column = (column - column.mean()) / column.std()

    expected = black_profile()
    n = min(len(column), len(expected))
    a = column[:n]
    b = expected[:n] * 2.0 - 1.0                      # nero = +1, bianco = -1
    normal = float(np.dot(a - a.mean(), b - b.mean()))
    flipped = float(np.dot(a - a.mean(), b[::-1] - b[::-1].mean()))
    total = abs(normal) + abs(flipped) + 1e-9
    return (flipped - normal) / total


def check_alignment(strip: np.ndarray, key_index: int) -> float:
    """Diagnostica: contrasto della colonna di un tasto rispetto ai vicini.

    Serve al controllo M0: se la griglia dei tasti e' giusta, la colonna del
    tasto premuto differisce dai vicini in modo marcato.
    """
    x0, x1 = key_index * KEY_PX, (key_index + 1) * KEY_PX
    here = strip[:, x0:x1].mean()
    left = strip[:, max(0, x0 - KEY_PX):x0].mean()
    right = strip[:, x1:x1 + KEY_PX].mean()
    return float(here - 0.5 * (left + right))


# ---------------------------------------------------------------------------
# Cosa si puo' sapere di un video che non mostra tutta la tastiera
#
# Un primo piano non contiene l'ottava: il disegno dei neri si ripete ogni 12
# semitoni, quindi dice *dove* si e' nel ciclo (la classe) ma non in quale ottava.
# Contiene pero' tre cose, che insieme riducono il campo da 88 tasti a poche
# candidate: la classe delle altezze, la larghezza della finestra visibile e lo
# spostamento della finestra fra un frame e l'altro (che allarga la finestra
# all'unione di tutto il video). Il verso, invece, *non* c'e': rettificata, la
# tastiera di un pianoforte e' simmetrica a meno di 8 semitoni, quindi lo specchio
# ha lo stesso punteggio dell'originale su ogni frame -- l'ho misurato, non e' una
# supposizione. Per il verso servono l'audio o l'utente.
# ---------------------------------------------------------------------------


def _px_per_semitone(span: float, lo: int = 0, hi: int = N_KEYS) -> float:
    """Pixel della striscia per semitono, dalla stima del passo."""
    return (hi - lo) * KEY_PX / max(float(span), 1e-9)


def _at_semitone_step(band: np.ndarray, span: float, lo: int = 0, hi: int = N_KEYS) -> np.ndarray:
    """Il profilo di una banda ricampionato a un campione per semitono.

    Serve a togliere di mezzo la scala: nel profilo cosi' com'e' un semitono vale
    un numero di pixel diverso per ogni frame, e il disegno dei neri non si puo'
    confrontare con quello canonico.
    """
    step = _px_per_semitone(span, lo, hi)
    quanti = max(1, int(band.size / step))
    return np.interp(np.arange(quanti) * step, np.arange(band.size), band)


def _cycle_mask() -> np.ndarray:
    """Il disegno dei neri per classe di altezza MIDI (1 = nero) su 12 semitoni."""
    mask = np.zeros(12, dtype=np.float64)
    for pitch_class in sorted(BLACK_PITCH_CLASSES):
        mask[pitch_class] = 1.0
    return mask


def left_pitch_class(
    band: np.ndarray, span: float, lo: int = 0, hi: int = N_KEYS
) -> tuple[int, float, float]:
    """Classe di altezza (0 = Do) del semitono piu' a sinistra visibile.

    Il disegno dei neri si correla con il disegno canonico per tutti e 12 gli
    scostamenti: vince il migliore e il *margine* sul secondo posto dice quanto la
    lettura e' solida. Il margine e' la parte importante: agli incroci fra ottave
    vicini il disegno ammette due letture quasi equivalenti, e senza margine non si
    distingue una lettura buona da una casuale.

    Ritorna `(classe, punteggio, margine)`, con classe -1 se la banda e' piatta.
    """
    osservato = -_at_semitone_step(band, span, lo, hi)
    osservato = osservato - osservato.mean()
    if osservato.size < 12 or float(osservato.std()) < 1e-9:
        return -1, 0.0, 0.0

    ciclo = _cycle_mask()
    punteggi = []
    for scostamento in range(12):
        modello = np.roll(np.tile(ciclo, osservato.size // 12 + 2), scostamento)[: osservato.size]
        modello = modello - modello.mean()
        if float(modello.std()) < 1e-9:
            punteggi.append(0.0)
            continue
        punteggi.append(
            float(np.dot(osservato, modello) / (osservato.size * osservato.std() * modello.std()))
        )
    ordine = np.argsort(punteggi)[::-1]
    # con modello[n] = ciclo[(n - s) % 12] la posizione n = s e' un Do, quindi la
    # prima colonna della banda vale la classe -s
    return int(-ordine[0] % 12), float(punteggi[ordine[0]]), float(punteggi[ordine[0]] - punteggi[ordine[1]])


def drift_semitones(
    bands: Sequence[np.ndarray],
    spans: Sequence[float],
    lo: int = 0,
    hi: int = N_KEYS,
    search: int = 250,
) -> float:
    """Di quanti semitoni si e' spostata la finestra visibile lungo la tastiera.

    Positivo significa che la finestra e' andata verso i tasti acuti (nell'immagine
    la tastiera scorre a sinistra). Si correla il profilo di un frame con quello del
    frame precedente: lo spostamento fra due frame e' molto minore di un ciclo di 12
    semitoni, quindi *questa* misura non e' ambigua, e accumulandola si sa di quanto
    si e' allargata la parte di tastiera vista dal video.

    E' indipendente dalla classe delle altezze, quindi confrontarle e' un controllo
    vero: `KeyWindow` lo fa e scarta i video in cui le due non concordano.
    """
    totale = 0.0
    for prima, dopo, span in zip(bands, bands[1:], spans):
        if prima.size != dopo.size or dopo.size <= 4 * search or span <= 0.0:
            continue
        a = dopo - dopo.mean()
        b = prima - prima.mean()
        if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
            continue
        correl = np.correlate(b, a, mode="full")[b.size - 1 - search: b.size - 1 + search + 1]
        picco = int(np.argmax(correl)) - search
        totale += picco / _px_per_semitone(span, lo, hi)
    return totale


@dataclass
class KeyWindow:
    """Dove sta la tastiera e quanto e' larga, per quanto il video puo' dirlo.

    `span` sono i semitoni visibili in un frame, `union` i semitoni *distinti* visti
    nell'intero video (cioe' `span` piu' la deriva): in un primo piano la deriva
    allarga la finestra, quindi l'unione e' quello che vincola dove puo' stare la
    tastiera. `first_class` e' la classe di altezza (0 = Do) del semitono piu' a
    sinistra visibile e `residual` e' la contraddizione, in semitoni, fra la classe e
    la deriva: le due misure sono indipendenti e devono raccontare la stessa storia,
    quindi un `residual` grande vuol dire che una delle due ha letto male -- e' il
    controllo che una soglia sul punteggio non puo' dare.

    `candidates` elenca le note MIDI possibili per il semitono piu' a sinistra: la
    classe e' nota, l'ottava no, e quante candidate restano lo decide `union` (nel
    video di prova, 23 tasti per frame e 30 di unione: 5 candidate su 88).
    `central` sceglie quella che mette la finestra vicino al centro della tastiera:
    e' un'ipotesi dichiarata, non una misura, ma e' quella che sbaglia di meno
    finche' non si sa altro -- e si puo' sempre fissare a mano.
    """

    span: float
    union: float
    first_class: int = -1
    class_margin: float = 0.0
    score: float = 0.0
    drift: float = 0.0
    residual: float = 0.0

    @property
    def trustworthy(self) -> bool:
        """Basta per costruire una striscia in scala e restringere le ottave?

        Non si guarda `class_margin`: su un primo piano pulito di 23 tasti il margine
        e' naturalmente sottile (0.05 nella prova sintetica, perche' si vedono solo
        due cicli), mentre su un video reale rumoroso sale (0.20-0.38 misurati) --
        usarlo come cancello scarterebbe i casi facili e accetterebbe i difficili. Il
        controllo vero e' `residual`, perche' confronta due misure indipendenti.
        """
        return (
            self.span > 0.0
            and self.score >= 0.6
            and self.first_class >= 0
            and self.union <= N_KEYS
            and abs(self.residual) <= 2.0
        )

    def candidates(self, n_keys: int = N_KEYS) -> list[int]:
        """Note MIDI possibili per il semitono piu' a sinistra visibile.

        Devono avere la classe misurata e lasciare spazio alla finestra dentro la
        tastiera. Senza la classe (banda piatta) si torna a tutte le note possibili;
        se la finestra e' piu' larga della tastiera la lista resta vuota, che e' il
        modo di dire che le misure si contraddicono.
        """
        if self.first_class < 0:
            return list(range(MIDI_MIN, MIDI_MAX + 1))
        larga = int(np.ceil(max(self.union, 1.0)))
        prima = MIDI_MIN + (self.first_class - MIDI_MIN) % 12
        return [nota for nota in range(prima, MIDI_MAX + 2 - larga, 12)]

    def central(self, n_keys: int = N_KEYS) -> int:
        """La candidata piu' vicina al centro della tastiera (ipotesi, non misura).

        Solleva `ValueError` se non ci sono candidate: significherebbe che la finestra
        stimata e' piu' larga della tastiera, quindi le misure si contraddicono e non
        c'e' niente da scegliere (`trustworthy` lo dice prima).
        """
        possibili = self.candidates(n_keys)
        if not possibili:
            raise ValueError("nessuna finestra possibile: le misure si contraddicono")
        centro = 0.5 * (MIDI_MIN + MIDI_MAX + 1)
        return min(possibili, key=lambda nota: abs(nota + 0.5 * self.union - centro))


def key_window(
    bands: Sequence[np.ndarray],
    spans: Sequence[KeySpan],
    lo: int = 0,
    hi: int = N_KEYS,
    search: int = 250,
    min_score: float = 0.5,
    tolerance: float = 2.0,
) -> KeyWindow:
    """Tutte le informazioni sull'ottava che si possono cavare da piu' frame.

    `bands` sono i profili dei tasti neri e `spans` le stime degli stessi frame, nello
    stesso ordine. I frame si scelgono con *due* regole, perche' i guasti sono due: il
    punteggio (sotto `min_score` la banda non contiene nemmeno un disegno di tasti
    credibile) e il consenso sul passo, che scarta il frame in cui il quadrilatero e'
    finito fuori dalla tastiera -- sul video di prova un frame cosi' misura 6 tasti con
    punteggio 0.66, indistinguibile da un frame buono se si guarda solo il punteggio.
    Lasciarli entrare non e' innocuo: la deriva e la classe verrebbero lette su
    spazzatura, e il `residual` lo direbbe solo alla fine.

    La classe si legge sul *primo* frame tenuto (se e' debole lo dice `class_margin`) e
    la deriva si accumula sui frame consecutivi: fra due frame campionati lo spostamento
    deve restare sotto mezzo ciclo di 12 semitoni, quindi conviene campionare fitto
    quando la camera si muove in fretta.
    """
    indici = [i for i, s in enumerate(spans) if s.span > 0.0 and s.score >= min_score]
    if not indici:
        return KeyWindow(0.0, 0.0)
    consensus = consensus_span([spans[i] for i in indici], tolerance)
    tenuti = [i for i in indici if abs(spans[i].span - consensus.span) <= tolerance]
    if not tenuti:
        return KeyWindow(0.0, 0.0)

    span = float(np.median([spans[i].span for i in tenuti]))
    score = float(np.median([spans[i].score for i in tenuti]))
    prima, _, margine = left_pitch_class(bands[tenuti[0]], spans[tenuti[0]].span, lo, hi)
    ultima, _, _ = left_pitch_class(bands[tenuti[-1]], spans[tenuti[-1]].span, lo, hi)
    deriva = drift_semitones([bands[i] for i in tenuti], [spans[i].span for i in tenuti], lo, hi, search)
    residuo = 0.0
    if prima >= 0 and ultima >= 0 and len(tenuti) > 1:
        # la classe *cresce* con la deriva: verificato su misura controllata (un
        # contenuto spostato di 3 semitoni a destra da' deriva -3 e classe -3)
        residuo = (ultima - prima - deriva + 6.0) % 12.0 - 6.0
    return KeyWindow(
        span=span,
        union=span + abs(deriva),
        first_class=prima,
        class_margin=float(margine),
        score=score,
        drift=float(deriva),
        residual=float(residuo),
    )


def window_from_strips(
    strips: Sequence[np.ndarray],
    lo: int = 0,
    hi: int = N_KEYS,
    candidates: np.ndarray | None = None,
    rows: slice | None = None,
    min_score: float = 0.5,
) -> KeyWindow:
    """La finestra visibile, misurata su alcune strisce rettificate.

    Mette insieme i pezzi: la fascia dei neri di ogni striscia (`black_key_band`), la
    stima del passo e la sintesi di `key_window`. E' il modo in cui il pipeline puo'
    accorgersi che il quadrilatero non copre 88 tasti: se `span` non e' 88, la striscia
    che il modello si aspetta e' un'altra, e le note escono con ottava e scala sbagliate.
    `candidates` restringe la ricerca del passo quando si sta inseguendo un video (vedi
    `estimate_key_span`).
    """
    bande = [black_key_band(strip, rows) for strip in strips]
    stime = [estimate_key_span(banda, lo, hi, candidates) for banda in bande]
    return key_window(bande, stime, lo, hi, min_score=min_score)

