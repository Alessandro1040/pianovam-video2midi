"""Test della geometria canonica dei tasti."""

from __future__ import annotations

import numpy as np
import pytest

from src import detect, keys


def synthetic_keyboard(flipped: bool = False):
    """Tastiera finta in stile striscia rettificata: tasti bianchi chiari, neri scuri."""
    strip = np.full((keys.STRIP_H, keys.STRIP_W), 230, dtype=np.uint8)
    top = int(keys.STRIP_H * 0.45)
    for index in range(keys.N_KEYS):
        if keys.is_black(index):
            strip[:top, index * keys.KEY_PX:(index + 1) * keys.KEY_PX] = 25
    if flipped:
        strip = strip[:, ::-1].copy()
    return strip


def test_layout_88_tasti():
    assert keys.N_KEYS == 88
    assert keys.index_to_note(0) == 21
    assert keys.index_to_note(87) == 108


def test_i_neri_sono_dove_ci_si_aspetta():
    # A0 (21) e' bianco, A#0 (22) e' nero, e ogni ottava si ripete
    assert not keys.is_black(keys.note_to_index(21))
    assert keys.is_black(keys.note_to_index(22))
    assert keys.is_black(keys.note_to_index(61))
    assert keys.black_mask().sum() == 36      # 36 neri e 52 bianchi su 88 tasti


def test_nomi_delle_note():
    assert keys.note_name(60) == "C4"
    assert keys.note_name(21) == "A0"
    assert keys.note_name(108) == "C8"


def test_punteggio_tasti_neri_alto_se_allineata_e_negativo_se_specchiata():
    """Il pattern dei tasti neri e' quasi simmetrico: cio' che conta e' il confronto.

    La versione allineata deve avere un punteggio alto; la versione specchiata
    resta positiva (il disegno dei neri non e' simmetrico al 100%), ma deve
    essere **chiaramente piu' bassa**: su quello si basa il rilevamento.
    """
    normal = keys.black_key_score(synthetic_keyboard())
    mirrored = keys.black_key_score(synthetic_keyboard(flipped=True))
    assert normal > 0.9
    assert mirrored < 0.8
    assert mirrored < normal - 0.2


def test_rilevamento_specchio():
    assert keys.detect_input_flip(np.stack([synthetic_keyboard()])) is False
    assert keys.detect_input_flip(np.stack([synthetic_keyboard(flipped=True)])) is True


def band_profile(strip: np.ndarray) -> np.ndarray:
    """Luminosita' media per colonna della fascia dei tasti neri, a risoluzione di pixel."""
    _, black, _ = detect.bands()
    return strip[black].mean(axis=0).astype(np.float64)


def test_i_tasti_neri_sono_cinque_per_ottava_non_sei():
    """Il disegno dei neri e' cinque per ottava (1, 3, 6, 8, 10), larghi un semitono.

    E' l'invariante su cui si regge la stima del passo: con sei tasti neri per ottava la
    frazione scura del modello passerebbe da 5/7 a 6/7, il modello non somiglierebbe piu'
    a nessun pianoforte, e la stima sceglierebbe numeri di tasti a caso.
    """
    centres = keys.black_centres()
    assert centres.shape == (5,)
    assert keys.black_centres().tolist() == pytest.approx(
        [keys.BLACK_OVER_WHITE * p for p in sorted(keys.BLACK_PITCH_CLASSES)]
    )
    span_px = 4 * 12 * keys.KEY_PX                      # quattro ottave, in colonne canoniche
    model = keys.key_model(width_white=keys.white_key_width(keys.N_KEYS), n_samples=span_px, sub=8)
    assert keys._runs(model > 0.5) == 4 * 5             # quattro ottave, cinque neri ciascuna
    assert float(model.mean()) == pytest.approx(5 * keys.BLACK_OVER_WHITE / 7, abs=0.01)


def test_stima_del_passo_sulla_striscia_canonica():
    """Verita' nota: la striscia canonica copre 88 tasti, e la stima lo deve dire."""
    span = keys.estimate_key_span(band_profile(synthetic_keyboard()))
    assert span.span == pytest.approx(88, abs=1)
    assert span.score > 0.6
    assert span.trustworthy


def test_stima_del_passo_di_un_primo_piano():
    """Tastiera letta da vicino: ~22 tasti in striscia, non 88.

    E' il caso che rompe la pipeline a mano libera: 88 colonne su 22 tasti danno un
    punteggio dei tasti neri vicino a zero, e da li' in poi ogni nota sbaglia scala.
    """
    strip = synthetic_keyboard()
    close_up = strip[:, 24 * keys.KEY_PX:46 * keys.KEY_PX]
    zoomed = np.ascontiguousarray(close_up[:, np.repeat(np.arange(close_up.shape[1]), 4)])
    span = keys.estimate_key_span(band_profile(zoomed))
    assert span.span == pytest.approx(22, abs=1)
    assert span.score > 0.6
    assert span.trustworthy


def test_stima_del_passo_e_cieca_allo_specchio():
    """Una sola banda non dice in che verso si legge la tastiera.

    Il disegno dei tasti neri specchiato e' il disegno stesso **spostato di 8 semitoni**
    ({1,3,6,8,10} + 8 = {2,4,6,9,11}, che e' il suo specchio): quindi una banda
    specchiata si spiega benissimo anche senza girarla. La stima del passo resta valida
    (88 tasti), l'orientamento lo decide `detect_input_flip`, che ha in mano gli estremi
    della tastiera.
    """
    span = keys.estimate_key_span(band_profile(synthetic_keyboard(flipped=True)))
    assert span.span == pytest.approx(88, abs=1)
    assert span.score > 0.8


def test_stima_del_passo_su_una_banda_piatta_non_crede_a_se_stessa():
    """Dove il disegno dei tasti non c'e' la stima deve dichiararlo, non inventare."""
    flat = keys.estimate_key_span(np.full(keys.STRIP_W, 90.0))
    assert flat.span == 0.0 and flat.score < 0.0
    assert not flat.trustworthy


def test_stima_del_passo_rifinibile_attorno_a_un_valore_noto():
    """Con una stima grossolana in mano basta cercare attorno: e' la modalita' inseguimento."""
    strip = synthetic_keyboard()
    close_up = strip[:, 24 * keys.KEY_PX:46 * keys.KEY_PX]
    zoomed = np.ascontiguousarray(close_up[:, np.repeat(np.arange(close_up.shape[1]), 4)])
    span = keys.estimate_key_span(
        band_profile(zoomed), candidates=np.array([20.0, 21.0, 22.0, 23.0, 24.0])
    )
    assert span.span == pytest.approx(22.0, abs=1.0)
    assert span.score > 0.6


def test_larghezza_del_tasto_bianco_dalla_stima():
    # Nella striscia canonica un tasto bianco e' 1.71 colonne (23.5 mm contro 13.7 mm)
    assert keys.white_key_width(keys.N_KEYS) == pytest.approx(1.71 * keys.KEY_PX, rel=0.01)
    # La stessa finestra di pixel con la meta' dei tasti: tasti larghi il doppio
    assert keys.white_key_width(44) == pytest.approx(2.0 * keys.white_key_width(keys.N_KEYS))


def test_il_consenso_scarta_la_misura_isolata():
    """Su un video le misure buone sono tante e concordi, quella sbagliata e' sola.

    E' il caso vero: al frame 300 del video a mano libera (quadrilatero ormai fuori dalla
    tastiera) la stima e' 6 tasti con punteggio 0.66, indistinguibile da un frame buono.
    Il consenso la lascia fuori invece di mediarla con le buone.
    """
    spans = [
        keys.KeySpan(23.0, 0.78, 10.0, False),
        keys.KeySpan(24.0, 0.72, 20.0, False),
        keys.KeySpan(22.0, 0.83, 30.0, False),
        keys.KeySpan(23.0, 0.70, 40.0, False),
        keys.KeySpan(6.0, 0.66, 0.0, False),        # l'isolata da buttare
    ]
    consensus = keys.consensus_span(spans)
    assert consensus.span == pytest.approx(23.0)
    assert consensus.score == pytest.approx(0.75)   # mediana dei punteggi del gruppo
    assert consensus.phase == 30.0                  # dal membro con il punteggio piu' alto
    assert consensus.trustworthy
    # Senza niente di utilizzabile non si inventa una scala
    empty = keys.consensus_span([keys.KeySpan(0.0, -1.0, 0.0, False)])
    assert empty.span == 0.0 and not empty.trustworthy
def test_classe_di_altezza_della_striscia_canonica():
    """La banda dei neri dice la classe (0 = Do) del tasto piu' a sinistra.

    Sulla striscia canonica il tasto piu' a sinistra e' La0 (MIDI 21), classe 9: e' la
    parte di informazione che un primo piano conserva anche senza mostrare la tastiera
    intera, e che basta a ridurre le ottave possibili da 88 a poche.
    """
    banda = synthetic_keyboard()[0].astype(np.float64)
    classe, punteggio, margine = keys.left_pitch_class(banda, keys.N_KEYS)
    assert classe == 9
    assert punteggio > 0.9
    assert margine > 0.2          # con 88 tasti (sette cicli) la lettura e' netta
    # Una banda piatta non dice niente, e lo dice invece di inventare
    assert keys.left_pitch_class(np.full(keys.STRIP_W, 128.0), keys.N_KEYS) == (-1, 0.0, 0.0)


def test_la_classe_scende_quando_la_tastiera_scorre_a_destra():
    """Due letture indipendenti dello stesso spostamento devono concordare.

    Un profilo spostato di un semitono a destra mostra un tasto piu' grave nella prima
    colonna, quindi la classe scende di 1; la deriva, misurata correlando i profili,
    scende della stessa quantita'. E' il controllo che serve: il punteggio dice che
    c'e' un disegno di tasti, solo il confronto dice se e' quello giusto.
    """
    banda = synthetic_keyboard()[0].astype(np.float64)
    spostata = np.roll(banda, keys.KEY_PX)          # un semitono a destra
    assert keys.left_pitch_class(spostata, keys.N_KEYS)[0] == 8
    assert keys.drift_semitones([banda, spostata], [float(keys.N_KEYS)]) == pytest.approx(-1.0)


def _close_up(strip, start: int, keys_wide: int = 23):
    """Cio' che la rettifica fa di un primo piano: `keys_wide` tasti su tutta la larghezza."""
    cut = strip[:, start * keys.KEY_PX:(start + keys_wide) * keys.KEY_PX].astype(np.float64)
    return np.interp(np.linspace(0, cut.shape[1] - 1, keys.STRIP_W), np.arange(cut.shape[1]), cut[0])


def _strip_of(band: np.ndarray) -> np.ndarray:
    """La banda data ripetuta su tutta l'altezza: una striscia rettificata senza prospettiva."""
    return np.tile(np.asarray(band, dtype=np.uint8), (keys.STRIP_H, 1))


def _flat_band(dark: int = 25, light: int = 230) -> np.ndarray:
    """Il profilo dei neri di una striscia rettificata piatta, a risoluzione di pixel."""
    return np.where(np.repeat(keys.black_mask(), keys.KEY_PX), dark, light)


def _close_up_strip(strip, start: int, keys_wide: int = 23):
    """Lo stesso primo piano come lo vede il modello: banda ripetuta su tutta l'altezza."""
    return _strip_of(_close_up(strip, start, keys_wide))


def test_la_banda_grezza_trova_i_tasti_neri():
    """La fascia dei neri a risoluzione di pixel: la misura che non presuppone il passo.

    E' a contrasto pieno (25 contro 230) su una striscia rettificata piatta, e resta
    ordinata anche sulla striscia sintetica, dove i tasti neri sono corti (si fermano
    prima della fascia, quindi la loro media sale da 25 a ~155) senza perdere il
    contrasto con i bianchi (230) -- ed e' il contrasto che serve alla misura.
    """
    banda = keys.black_key_band(_strip_of(_flat_band()))
    assert banda.shape == (keys.STRIP_W,)
    assert banda[np.repeat(keys.black_mask(), keys.KEY_PX)].mean() == pytest.approx(25.0)
    assert banda[~np.repeat(keys.black_mask(), keys.KEY_PX)].mean() == pytest.approx(230.0)

    realistica = keys.black_key_band(synthetic_keyboard())
    neri = realistica[np.repeat(keys.black_mask(), keys.KEY_PX)].mean()
    bianchi = realistica[~np.repeat(keys.black_mask(), keys.KEY_PX)].mean()
    assert neri < bianchi
    assert bianchi == pytest.approx(230.0, abs=1.0)


def test_una_striscia_intera_non_fa_scattare_l_avviso():
    """Il caso normale del dataset: il quadrilatero copre tutti gli 88 tasti.

    Ed e' anche il controllo che la banda usata per misurare il passo e' quella giusta:
    la stima cade su 88 tasti con punteggio alto e legge la classe giusta (la striscia
    canonica parte da La0).
    """
    finestra = keys.window_from_strips([_strip_of(_flat_band())])
    assert finestra.span == pytest.approx(float(keys.N_KEYS), abs=1.0)
    assert finestra.first_class == 9               # la striscia canonica parte da La0
    assert finestra.trustworthy


def test_il_punteggio_a_88_tasti_non_vede_l_inquadratura_stretta():
    """Perche' serve una misura nuova.

    Il punteggio a 88 tasti chiede "i neri sono dove dico io?", quindi su un primo piano
    di 23 tasti il raggruppamento in 88 colonne mescola i tasti veri e la risposta e'
    rumore (-0.003): *non puo' avvertire* che la striscia e' un'altra. La larghezza
    invece si misura lo stesso, e la nota vera resta fra le candidate.
    """
    intera = synthetic_keyboard()
    vicino = _close_up_strip(intera, 39)                  # 23 tasti da Do4 (MIDI 60)
    assert keys.black_key_score(intera) > 0.99            # la striscia intera torna
    assert abs(keys.black_key_score(vicino)) < 0.2        # il primo piano no: rumore

    finestra = keys.window_from_strips([vicino])
    assert finestra.span == pytest.approx(23.0, abs=0.6)
    assert finestra.first_class == 0                      # Do, il tasto da cui parte
    assert 60 in finestra.candidates()                    # la nota vera e' fra le candidate
    # e la stessa misura funziona con la griglia stretta, come quando si insegue
    inseguendo = keys.window_from_strips([vicino], candidates=np.arange(18.0, 30.0, 0.5))
    assert inseguendo.span == pytest.approx(23.0, abs=0.6)


def test_primo_piano_che_scorre_senza_mostrare_la_tastiera():
    """Il caso vero: 23 tasti inquadrati che scorrono di 3 semitoni.

    Si ritaglia la striscia canonica come farebbe la rettifica di un primo piano e si
    sposta il ritaglio, come una camera che scorre. La classe sale di 3 e la deriva lo
    conferma: la finestra visibile scorre di 3 semitoni, quindi l'unione di tutto il
    video e' larga 26 semitoni invece di 23, e le ottave possibili restano poche.
    """
    strip = synthetic_keyboard()
    primo = _close_up(strip, 39)                 # 39 = Do4 (MIDI 60), classe 0
    dopo = _close_up(strip, 42)                  # la camera e' scorsa di 3 semitoni
    span = keys.estimate_key_span(primo)
    assert span.span == pytest.approx(23.0, abs=0.6)
    assert keys.left_pitch_class(primo, span.span)[0] == 0
    assert keys.left_pitch_class(dopo, span.span)[0] == 3
    finestra = keys.key_window([primo, dopo], [span, span])
    assert finestra.span == pytest.approx(23.0, abs=0.6)
    assert finestra.union == pytest.approx(26.0, abs=0.6)
    assert finestra.drift == pytest.approx(3.0, abs=0.6)
    assert finestra.residual == pytest.approx(0.0, abs=0.6)     # le due misure concordano
    assert finestra.trustworthy


def test_le_candidate_dell_ottava_sono_poche():
    """I numeri veri del video a mano libera: 23 tasti per frame, 30 di unione, classe 3.

    L'ottava non e' nell'immagine, quindi non si puo' misurare: si restringe. Delle 88
    note possibili ne restano 5, una per ottava; la piu' vicina al centro della tastiera
    e' Re#3, che mette la finestra su Re#3..La5 -- un'ipotesi dichiarata, che contiene il
    Do centrale, e che si puo' sempre fissare a mano con una nota vera.
    """
    finestra = keys.KeyWindow(
        span=23.0, union=30.2, first_class=3, class_margin=0.34, score=0.78, drift=-7.2, residual=-0.8
    )
    assert finestra.trustworthy
    assert finestra.candidates() == [27, 39, 51, 63, 75]
    assert keys.note_name(finestra.central()) == "D#3"
    # la finestra copre `ceil(union)` semitoni, quindi dal primo all'ultimo compreso
    assert keys.note_name(finestra.central() + int(np.ceil(finestra.union)) - 1) == "A5"


def test_una_finestra_piu_larga_della_tastiera_non_da_candidate():
    """Se le misure si contraddicono non si sceglie l'ipotesi meno peggio: si dice no."""
    impossibile = keys.KeyWindow(span=80.0, union=91.0, first_class=9, class_margin=0.3, score=0.8)
    assert impossibile.candidates() == []
    assert not impossibile.trustworthy
    with pytest.raises(ValueError):
        impossibile.central()


def test_la_finestra_scarta_i_frame_fuori_dalla_tastiera():
    """Due guasti diversi, due regole: punteggio basso e passo fuori dal consenso.

    Sono i casi veri del video a mano libera: agli ultimi frame la camera esce dalla
    tastiera e la banda misura 6 tasti con punteggio 0.66 (indistinguibile da un frame
    buono, se si guarda solo il punteggio) oppure 18 tasti con punteggio 0.08. Se
    entrassero, la deriva e la classe verrebbero lette su spazzatura: sul video vero e'
    successo, e il residuo e' passato da -0.8 a -5.2 marcando la misura come inutilizzabile.
    """
    strip = synthetic_keyboard()
    primo = _close_up(strip, 39)
    dopo = _close_up(strip, 42)
    buono = keys.estimate_key_span(primo)
    fuori = keys.KeySpan(6.0, 0.66, 0.0, False)      # quadrilatero fuori dalla tastiera
    cieco = keys.KeySpan(18.0, 0.08, 0.0, False)     # banda senza disegno di tasti

    finestra = keys.key_window([primo, dopo, primo, primo], [buono, buono, fuori, cieco])
    assert finestra.drift == pytest.approx(3.0, abs=0.6)
    assert finestra.residual == pytest.approx(0.0, abs=0.6)
    assert finestra.trustworthy

    # senza niente di utilizzabile non si inventa una scala
    vuota = keys.key_window([primo], [cieco])
    assert vuota.span == 0.0 and not vuota.trustworthy
    assert not keys.key_window([], []).trustworthy

