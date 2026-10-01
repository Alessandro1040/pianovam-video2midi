"""Inseguire la tastiera nel tempo: il "detector + tracker", come fa MediaPipe.

Perche' serve: `geometry.py` normalizza la geometria **una volta sola**, con i 4
angoli dati all'inizio. Se la camera si muove durante la registrazione (un telefono
tenuto a mano) quella normalizzazione vale solo per il primo fotogramma e tutti gli
altri sono stesi con la griglia dei tasti sbagliata. Misurato sul video di prova a
mano libera: 99% dei fotogrammi con la griglia spostata di oltre un tasto, 76% di
oltre un tasto *visibile*.

La ricetta e' quella dei tracker con re-detection:

* il **detector** c'e' gia' (`detect.py`): fornisce la posa iniziale (o quella
  dell'annotazione) e serve a ri-agganciarsi quando il tracker perde l'oggetto;
* il **tracker** insegue la tastiera fotogramma per fotogramma: prende i punti
  d'interesse della *striscia rettificata* del fotogramma precedente (cioe' in
  coordinate canoniche, dove ogni tasto sta sempre nella sua colonna), rende il
  fotogramma nuovo con la matrice precedente e misura col flusso ottico quanto
  manca ancora per rimettere i tasti a posto. La correzione viene **composta** con
  la posa precedente, quindi la posa resta assoluta e la deriva non si accumula.

Due scelte che contano, entrambe misurate:

* si inseguono **i tasti**, non lo sfondo. Lo sfondo e' piu' lontano e si muove
  meno: compensarlo lascia la tastiera che si muove (il rumore del canale di
  movimento non cambia: 25,7 -> 26,3 livelli). La tastiera invece e' **planare**, e
  per un oggetto piano un'omografia e' il modello *esatto* del movimento, non
  un'approssimazione;
* i punti si prendono **solo sulle righe dei tasti** (`detect.bands`): sopra la
  tastiera c'e' il pianista, che si muove davvero e non va inseguito.

Le mani sono occlusori mobili: per questo l'omografia si stima con RANSAC, che le
scarta come outlier, esattamente come un tracker di volti scarta la mano che passa
davanti. Quando gli inlier non bastano (sfocatura da scatto, salto troppo grande,
tastiera uscita dall'inquadratura) il tracker **tiene la posa precedente** e lo
dichiara (`TrackState.tracked = False`), invece di inventare una posa.

Il costo e' minimo perche' tutto avviene nella striscia (96x1056 pixel), non nel
fotogramma 1080p: il flusso ottico costa meno della decodifica stessa.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import detect, geometry, keys

# --- parametri del tracker, in pixel della striscia canonica (1 tasto = 12 px) ---
MAX_FEATURES = 600          # quanti punti d'interesse seguire
FEATURE_QUALITY = 0.01      # qualita' minima dei punti (Shi-Tomasi)
FEATURE_MIN_DISTANCE = 4    # distanza minima fra i punti, in pixel di striscia
LK_WINDOW = 15              # finestra del flusso ottico
LK_LEVELS = 3               # livelli di piramide: regge spostamenti di ~60 px
RANSAC_PX = 3.0             # soglia di inlier dell'omografia
MIN_INLIERS = 12            # sotto questo numero la stima non e' credibile
MIN_RATIO = 0.35            # ... ne' sotto questa frazione dei punti seguiti
MIN_SPREAD = 0.15           # gli inlier devono coprire almeno questa parte della striscia
MAX_STEP_KEYS = 12.0        # salto massimo credibile fra due fotogrammi, in tasti
MAX_SCALE_STEP = 0.12       # cambio di scala massimo credibile, per fotogramma
MAX_HEIGHT_STEP = 0.25      # cambio di altezza (prospettiva) massimo, per fotogramma
ANCHOR_EVERY = 30           # ogni quanti fotogrammi rifinire la fase dei tasti (0 = mai)
ANCHOR_PX = 4.0             # ampiezza della rifinitura, in pixel di striscia
ANCHOR_SCALE = 0.02         # ampiezza della rifinitura, in frazione di larghezza
ANCHOR_GAIN = 0.01          # la rifinitura si accetta solo se migliora di tanto
HEADROOM = 0.25             # di quanto si allarga il ritaglio attorno alla posa precedente
INSIDE_GRID = 5             # griglia con cui si misura quanto della tastiera e' inquadrata


@dataclass
class TrackerConfig:
    """Parametri del tracker. I default sono quelli misurati sui video del dataset."""

    strip_hw: tuple[int, int] = (keys.STRIP_H, keys.STRIP_W)
    max_features: int = MAX_FEATURES
    feature_quality: float = FEATURE_QUALITY
    feature_min_distance: float = FEATURE_MIN_DISTANCE
    lk_window: int = LK_WINDOW
    lk_levels: int = LK_LEVELS
    ransac_px: float = RANSAC_PX
    min_inliers: int = MIN_INLIERS
    min_ratio: float = MIN_RATIO
    min_spread: float = MIN_SPREAD
    max_step_keys: float = MAX_STEP_KEYS
    max_scale_step: float = MAX_SCALE_STEP
    max_height_step: float = MAX_HEIGHT_STEP
    anchor_every: int = ANCHOR_EVERY
    anchor_px: float = ANCHOR_PX
    anchor_scale: float = ANCHOR_SCALE
    anchor_gain: float = ANCHOR_GAIN
    headroom: float = HEADROOM

    @property
    def size(self) -> tuple[int, int]:
        """(larghezza, altezza) della striscia: l'ordine che vuole OpenCV."""
        return self.strip_hw[1], self.strip_hw[0]


@dataclass(frozen=True)
class TrackState:
    """Posa della tastiera in un fotogramma, con la striscia gia' stabilizzata."""

    quad: np.ndarray        # 4x2 float32, pixel del fotogramma (LT, RT, RB, LB)
    matrix: np.ndarray      # 3x3: fotogramma -> striscia canonica
    strip: np.ndarray       # (H, W) uint8, rettificata con questa posa
    confidence: float       # frazione di inlier del flusso ottico (0 = posa tenuta)
    inliers: int
    tracked: bool           # False = stima non credibile, tenuta la posa precedente
    anchored: bool          # True se in questo fotogramma e' scattata la rifinitura
    inside: float           # 0..1: quanta parte della tastiera sta dentro l'inquadratura


def _shift(matrix: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """La stessa omografia, ma nelle coordinate del ritaglio `box` del fotogramma."""
    x0, y0 = box[0], box[1]
    return matrix @ np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])


def quad_of(
    matrix: np.ndarray, strip_hw: tuple[int, int] = (keys.STRIP_H, keys.STRIP_W)
) -> np.ndarray:
    """Il quadrilatero nel fotogramma che la matrice manda nel rettangolo canonico."""
    h, w = strip_hw
    dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)
    src = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(matrix))
    return src.reshape(-1, 2).astype(np.float32)


def scale_quad(quad: np.ndarray, factor: float) -> np.ndarray:
    """Ingrandisce il quadrilatero attorno al suo centro (serve al margine di ricerca)."""
    centre = quad.mean(axis=0, keepdims=True)
    return (centre + (quad - centre) * factor).astype(np.float32)


def inside_fraction(quad: np.ndarray, shape: tuple[int, int], grid: int = INSIDE_GRID) -> float:
    """Quanta parte del quadrilatero cade dentro il fotogramma (0..1).

    Misura la cosa che nessun tracker puo' inventare: se la tastiera e' uscita
    dall'inquadratura, li' non c'e' piu' niente da seguire.
    """
    h, w = shape
    u = np.linspace(0.0, 1.0, grid)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    top = quad[0][None, None, :] * (1 - uu)[..., None] + quad[1][None, None, :] * uu[..., None]
    bottom = quad[3][None, None, :] * (1 - uu)[..., None] + quad[2][None, None, :] * uu[..., None]
    points = top * (1 - vv)[..., None] + bottom * vv[..., None]
    inside = (
        (points[..., 0] >= 0) & (points[..., 0] < w) & (points[..., 1] >= 0) & (points[..., 1] < h)
    )
    return float(inside.mean())


def keyband_mask(strip_hw: tuple[int, int] = (keys.STRIP_H, keys.STRIP_W)) -> np.ndarray:
    """Maschera con 1 sui tasti e 0 fuori: sopra c'e' il pianista, davanti le sue mani.

    Le fasce sono quelle misurate sulle cache (`detect.bands`): righe 0-26 sopra la
    tastiera, 27-71 con il disegno dei tasti, 72-95 di nuovo piatte, cioe' davanti ai
    tasti. Si tiene solo la fascia centrale: la fascia alta perche' li' c'e' il
    pianista, la fascia bassa perche' li' ci sono mani e braccia, che si muovono
    davvero e tirerebbero la stima dietro di se'.
    """
    h, w = strip_hw
    _, black, _ = detect.bands(h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[black, :] = 255
    return mask


def pick_features(strip: np.ndarray, mask: np.ndarray, config: TrackerConfig) -> np.ndarray | None:
    """Punti d'interesse sui tasti: angoli dei tasti neri e delle cuciture fra i bianchi."""
    return cv2.goodFeaturesToTrack(
        strip,
        maxCorners=config.max_features,
        qualityLevel=config.feature_quality,
        minDistance=config.feature_min_distance,
        mask=mask,
        blockSize=5,
    )


def _flow(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray, config: TrackerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Flusso ottico dei punti fra due strisce, tenendo solo quelli ritrovati."""
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(config.lk_window, config.lk_window),
        maxLevel=config.lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    good = status.reshape(-1) == 1
    return points[good], moved[good]


def _pattern_score(strip: np.ndarray, bands: tuple[slice, slice, slice]) -> float:
    """Punteggio dei tasti neri sulla striscia, senza ri-rettificare.

    E' la stessa formula di `detect.score_rect` (il disegno periodico dei neri deve
    esserci nella fascia dei tasti e *non* sopra ne' davanti), ma applicata a una
    striscia che esiste gia': serve alla rifinitura, che deve costare poco.
    """
    above, black, white = bands
    return float(
        detect.pattern_presence(strip[black])
        - detect.pattern_presence(strip[above])
        - 0.5 * detect.pattern_presence(strip[white])
    )


def _anchor_transform(
    size: tuple[int, int], dx: float, dy: float, scale_x: float, scale_y: float
) -> np.ndarray:
    """Traslazione + zoom attorno al centro della striscia (le coordinate dei tasti)."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    return np.array(
        [
            [scale_x, 0.0, cx * (1 - scale_x) + dx],
            [0.0, scale_y, cy * (1 - scale_y) + dy],
            [0.0, 0.0, 1.0],
        ]
    )


def anchor_candidates(px: float, scale: float) -> list[tuple[float, float, float, float]]:
    """Le posizioni della rifinitura: traslazioni e zoom in una griglia fissa e *limitata*.

    Il limite non e' un dettaglio: il disegno dei tasti neri si ripete ogni 12 pixel,
    quindi una ricerca libera potrebbe scivolare di un'ottava e "migliorare" il
    punteggio restando sbagliata. Qui la rifinitura e' solo un anti-deriva: si muove
    al massimo di `px` pixel e `scale` di larghezza, e non di piu'.
    """
    out: list[tuple[float, float, float, float]] = []
    for step_x in (-px, 0.0, px):
        for step_y in (-px, 0.0, px):
            for step_s in (-scale, 0.0, scale):
                out.append((step_x, step_y, step_s, step_s))
    return out


class KeyboardTracker:
    """Detector + tracker per la tastiera: una posa (e una striscia) per fotogramma.

    Uso, il percorso completo:

        tracker = KeyboardTracker(quad)          # la posa iniziale (annotazione o detector)
        stato = tracker.first(fotogramma_0)      # striscia del primo fotogramma
        for fotogramma in resto_del_video:
            stato = tracker.update(fotogramma)   # la tastiera viene inseguita
            ...                                  # stato.strip e' la striscia stabilizzata
    """

    def __init__(self, quad: np.ndarray, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.size = self.config.size
        self.strip_hw = self.config.strip_hw
        self.mask = keyband_mask(self.strip_hw)
        self.bands = detect.bands(self.strip_hw[0])
        self.shape: tuple[int, int] | None = None
        self.matrix = geometry.homography(
            np.asarray(quad, dtype=np.float32), self.size[0], self.size[1]
        )
        self.previous: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.frame_index = -1
        self.lost_streak = 0
        self.counters = {"held": 0, "rejected": 0, "anchored": 0}

    @property
    def quad(self) -> np.ndarray:
        """Il quadrilatero della posa corrente, in pixel del fotogramma."""
        return quad_of(self.matrix, self.strip_hw)

    def set_pose(self, quad: np.ndarray) -> None:
        """Riporta il tracker su una posa esterna (annotazione del dataset o detector)."""
        self.matrix = geometry.homography(
            np.asarray(quad, dtype=np.float32), self.size[0], self.size[1]
        )

    def render(self, gray: np.ndarray, matrix: np.ndarray | None = None, headroom: float = 0.0) -> np.ndarray:
        """Striscia canonica del fotogramma con la matrice indicata (default: la corrente).

        Il ritaglio si prende attorno al quadrilatero *di quella matrice*. Con
        `headroom` il ritaglio e' piu' largo del necessario: serve al flusso ottico, che
        confronta la posa vecchia col fotogramma nuovo e quindi deve avere un po' di
        margine attorno. Senza `headroom` si ritaglia esattamente quello che serve.
        """
        matrix = self.matrix if matrix is None else matrix
        quad = quad_of(matrix, self.strip_hw)
        if headroom > 0:
            quad = scale_quad(quad, 1.0 + headroom)
        box = geometry.bounding_box(quad, gray.shape[:2], pad=8)
        x0, y0, x1, y1 = box
        return geometry.rectify(gray[y0:y1, x0:x1], _shift(matrix, box), self.size)

    def first(self, frame: np.ndarray) -> TrackState:
        """Primo fotogramma: la posa e' quella con cui il tracker e' stato costruito."""
        gray = detect.to_gray(frame)
        self.shape = gray.shape[:2]
        self.frame_index = 0
        strip = self.render(gray, headroom=0.0)
        self._remember(strip)
        return self._state(strip, confidence=1.0, inliers=0, tracked=True)

    def update(self, frame: np.ndarray) -> TrackState:
        """Fotogramma successivo: insegue la tastiera e restituisce la striscia stabilizzata."""
        gray = detect.to_gray(frame)
        if self.shape is None or self.previous is None or self.points is None:
            return self.first(frame)
        self.frame_index += 1

        # il fotogramma nuovo reso con la posa precedente: se il mondo non si fosse
        # mosso, questa striscia sarebbe identica a quella del fotogramma precedente
        naive = self.render(gray, headroom=self.config.headroom)
        delta, inliers, ratio = self._estimate(self.previous, naive)

        tracked = False
        if delta is not None:
            candidate = np.linalg.inv(delta) @ self.matrix
            if self._plausible(candidate):
                self.matrix = candidate
                tracked = True
            else:
                self.counters["rejected"] += 1

        if tracked:
            self.lost_streak = 0
        else:
            self.lost_streak += 1
            self.counters["held"] += 1

        anchored = self._maybe_anchor(gray) if tracked else False
        strip = self.render(gray, headroom=0.0)
        self._remember(strip)
        return self._state(
            strip,
            confidence=ratio if tracked else 0.0,
            inliers=inliers,
            tracked=tracked,
            anchored=anchored,
        )

    # ------------------------------------------------------------- interni
    def _remember(self, strip: np.ndarray) -> None:
        """La striscia appena prodotta diventa il riferimento del fotogramma successivo."""
        self.previous = strip
        self.points = pick_features(strip, self.mask, self.config)

    def _estimate(
        self, previous: np.ndarray, naive: np.ndarray
    ) -> tuple[np.ndarray | None, int, float]:
        """Omografia striscia->striscia che rimette i tasti a posto, e i suoi inlier.

        `delta` porta i punti come stavano nella striscia precedente dove sono finiti
        nella striscia "ingenua": la correzione da comporre con la posa precedente e'
        la sua inversa. Si ritorna `None` se la stima non e' credibile, e in quel caso
        il chiamante tiene la posa che aveva.
        """
        if self.points is None or len(self.points) < 4:
            return None, 0, 0.0
        p1, p2 = _flow(previous, naive, self.points, self.config)
        if len(p1) < 4:
            return None, 0, 0.0
        delta, mask = cv2.findHomography(
            p1, p2, cv2.RANSAC, self.config.ransac_px, maxIters=2000, confidence=0.995
        )
        if delta is None or mask is None:
            return None, 0, 0.0
        keep = mask.reshape(-1).astype(bool)
        inliers = int(keep.sum())
        ratio = inliers / float(len(p1))
        if inliers < self.config.min_inliers or ratio < self.config.min_ratio:
            return None, inliers, ratio
        # gli inlier devono essere sparsi sulla tastiera: una nuvola di punti su un
        # tasto solo produce un'omografia che "spiega" tutto e non vuol dire niente
        spread = p1[keep].reshape(-1, 2)
        if (
            np.ptp(spread[:, 0]) < self.config.min_spread * self.size[0]
            or np.ptp(spread[:, 1]) < self.config.min_spread * self.size[1]
        ):
            return None, inliers, ratio
        return delta, inliers, ratio

    def _plausible(self, candidate: np.ndarray) -> bool:
        """Scarta le correzioni impossibili: salto, zoom o prospettiva troppo grandi.

        Il flusso ottico ogni tanto *aggancia* il disegno giusto un'ottava piu' in la'
        (i tasti neri si ripetono), e quella e' una posa sbagliata che il punteggio dei
        tasti neri non distingue. Il vincolo che la salva e' fisico: fra due fotogrammi
        consecutivi la tastiera non puo' saltare venti tasti.
        """
        if not np.all(np.isfinite(candidate)):
            return False
        quad = quad_of(candidate, self.strip_hw)
        if not np.all(np.isfinite(quad)):
            return False

        here = self.quad
        width = detect.key_width_px(here)
        if width <= 1e-6:
            return False
        if float(np.linalg.norm(quad - here, axis=1).max()) > self.config.max_step_keys * width:
            return False
        if abs(detect.key_width_px(quad) / width - 1.0) > self.config.max_scale_step:
            return False
        before_h = geometry.quad_scale(here).height
        after_h = geometry.quad_scale(quad).height
        if before_h > 1e-6 and abs(after_h / before_h - 1.0) > self.config.max_height_step:
            return False
        return True

    def _maybe_anchor(self, gray: np.ndarray) -> bool:
        """Rifinitura periodica della fase dei tasti: l'anti-deriva del tracker.

        Il flusso ottico e' preciso ma leggermente distorto: su una registrazione lunga
        la griglia dei tasti puo' scivolare di qualche pixel. Qui, ogni `anchor_every`
        fotogrammi, si prova a spostare e zoomare la striscia di poco e si tiene la
        posizione col punteggio dei tasti neri migliore: e' un detector **locale**, che
        parte da una posa gia' buona e quindi non ha il problema della scala che rende
        inaffidabile la ricerca da zero (vedi la sezione in `detect.py`).
        """
        cfg = self.config
        if not cfg.anchor_every or self.frame_index % cfg.anchor_every:
            return False
        start_score = _pattern_score(self.render(gray, headroom=0.0), self.bands)
        best, best_score = self.matrix, start_score
        steps = ((cfg.anchor_px, cfg.anchor_scale), (cfg.anchor_px / 4, cfg.anchor_scale / 4))
        for px, scale in steps:
            for dx, dy, sx, sy in anchor_candidates(px, scale):
                candidate = _anchor_transform(self.size, dx, dy, 1.0 + sx, 1.0 + sy) @ best
                score = _pattern_score(self.render(gray, candidate, headroom=0.0), self.bands)
                if score > best_score:
                    best, best_score = candidate, score
        if best_score > start_score + cfg.anchor_gain:
            self.matrix = best
            self.counters["anchored"] += 1
            return True
        return False

    def _state(
        self,
        strip: np.ndarray,
        confidence: float,
        inliers: int,
        tracked: bool,
        anchored: bool = False,
    ) -> TrackState:
        quad = self.quad
        return TrackState(
            quad=quad,
            matrix=self.matrix.copy(),
            strip=strip,
            confidence=float(confidence),
            inliers=int(inliers),
            tracked=bool(tracked),
            anchored=bool(anchored),
            inside=inside_fraction(quad, self.shape or strip.shape[:2]),
        )


def summarise(confidence: np.ndarray, tracked: np.ndarray, inside: np.ndarray) -> dict:
    """I numeri da stampare dopo aver inseguito una registrazione intera.

    Servono a rispondere alla domanda "il tracker ha tenuto?": quante posizioni sono
    state misurate e quante *tenute*, con che confidenza, e per quanti fotogrammi la
    tastiera non era tutta dentro l'inquadratura (quello nessun tracker lo risolve).
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    tracked = np.asarray(tracked, dtype=bool).reshape(-1)
    inside = np.asarray(inside, dtype=np.float64).reshape(-1)
    kept = confidence[tracked]
    return {
        "frames": int(confidence.size),
        "tracked_frames": int(tracked.sum()),
        "held_frames": int((~tracked).sum()),
        "held_fraction": float((~tracked).mean()) if confidence.size else 0.0,
        "mean_confidence": float(kept.mean()) if kept.size else 0.0,
        "min_confidence": float(kept.min()) if kept.size else 0.0,
        "mean_inside": float(inside.mean()) if inside.size else 0.0,
        "frames_incomplete": int((inside < 0.999).sum()),
    }
