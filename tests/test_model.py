"""Test del modello, del campionamento del dataset e dell'aumentazione a specchio."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import data, keys, model, targets


def test_forma_dell_uscita():
    net = model.KeyNet()
    assert net.in_channels == 2
    strips = torch.randn(2, 8, 2, keys.STRIP_H, keys.STRIP_W)
    heads = net(strips)
    assert set(heads) == {"onset", "frame", "offset", "velocity"}
    for value in heads.values():
        assert value.shape == (2, 8, keys.N_KEYS)


def test_modello_a_un_canale():
    """I modelli piu' vecchi (una sola immagine, senza differenza) devono restare caricabili."""
    net = model.KeyNet(in_channels=1)
    heads = net(torch.randn(1, 4, 1, keys.STRIP_H, keys.STRIP_W))
    assert heads["onset"].shape == (1, 4, keys.N_KEYS)


def test_campo_recettivo_dichiarato():
    """Dilatazioni 1,2,4,8,16 con kernel 3 -> 1 + 2*(1+2+4+8+16) = 63 frame (1.05 s a 60 fps)."""
    net = model.KeyNet()
    assert net.receptive_field_frames() == 63


def test_campo_recettivo_misurato_con_impulso():
    """Verifica empirica: un impulso in un frame influenza al massimo +-31 frame.

    Il valore teorico e' 63 frame (1.05 s a 60 fps); qui si controlla che il
    supporto dell'impulso stia esattamente dentro quel tetto.
    """
    net = model.KeyNet().eval()
    length, position = 200, 100
    strips = torch.zeros(1, length, 2, keys.STRIP_H, keys.STRIP_W)
    with torch.no_grad():
        baseline = net(strips)["onset"][0]
        strips[0, position] = 3.0                  # impulso in un solo frame
        perturbed = net(strips)["onset"][0]

    changed = (perturbed - baseline).abs().max(dim=1).values > 0.0
    indices = torch.nonzero(changed).flatten()
    half = (net.receptive_field_frames() - 1) // 2
    assert int(indices.min()) >= position - half
    assert int(indices.max()) <= position + half
    assert indices.numel() >= 61                   # il supporto e' pieno, non bucherellato
    # e fuori dal campo recettivo il segnale e' esattamente zero
    assert bool((perturbed[: position - half] - baseline[: position - half] == 0).all())


def test_config_del_checkpoint(tmp_path):
    """Il numero di canali e il raggio si ricavano dai pesi e da args.json."""
    net = model.KeyNet()
    path = tmp_path / "best.pt"
    torch.save(net.state_dict(), path)
    assert model.in_channels_from_checkpoint(path) == 2
    assert model.temporal_radius_for_checkpoint(path) == 15      # default senza args.json

    (tmp_path / "args.json").write_text('{"temporal_radius": 7}')
    assert model.temporal_radius_for_checkpoint(path) == 7

    single = tmp_path / "old.pt"
    torch.save(model.KeyNet(in_channels=1).state_dict(), single)
    assert model.in_channels_from_checkpoint(single) == 1
    assert model.temporal_radius_for_checkpoint(single) == 0     # un canale -> nessuna differenza

    loaded = model.load_checkpoint(path, device="cpu")
    assert loaded.in_channels == 2


def test_bias_delle_teste_parte_dal_prior():
    net = model.KeyNet()
    net.init_head_bias({"onset": 0.001, "frame": 0.01, "offset": 0.001})
    for name, prior in (("onset", 0.001), ("frame", 0.01), ("offset", 0.001)):
        bias = float(net.heads[name].bias.detach())
        assert abs(bias - np.log(prior / (1 - prior))) < 1e-6
    assert model.count_parameters(net) > 100_000


def test_normalizzazione_in_range():
    strips = np.array([[[0, 128, 255]]], dtype=np.uint8)
    out = data._normalise(strips)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0, abs=1e-3)


def test_canale_di_differenza_isola_il_cambiamento():
    """Il canale 1 deve accendersi solo dove e quando l'immagine cambia."""
    window = np.full((21, 4, 4), 100, dtype=np.uint8)
    window[10, 0, 0] = 140                      # un tasto che cambia per un frame solo
    features = data.temporal_features(window, radius=5)
    assert features.shape == (21, 2, 4, 4)

    difference = features[:, 1]
    assert difference[10, 0, 0] > 0.5           # illuminato esattamente li'
    assert difference[10, 1, 1] == 0.0          # e da nessun'altra parte
    assert difference[0, 0, 0] == 0.0
    # il canale assoluto, invece, vede solo "una zona piu' chiara delle altre"
    assert features[10, 0, 0, 0] > features[10, 0, 1, 1]


def test_temporal_features_senza_raggio():
    window = np.full((5, 4, 4), 100, dtype=np.uint8)
    features = data.temporal_features(window, radius=0)
    assert features.shape == (5, 1, 4, 4)


def test_aumentazione_fotometrica_non_cambia_forma():
    config = data.AugmentConfig()
    strips = np.full((4, 8, 32), 120, dtype=np.uint8)
    out = data.augment_strip(strips, np.random.default_rng(0), config)
    assert out.shape == strips.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_lo_specchio_inverte_anche_le_etichette():
    """Il punto chiave dell'aumentazione: immagine specchiata <-> tasti invertiti."""
    config = data.DatasetConfig(clip_frames=8)
    config.augment.flip_prob = 1.0                  # specchio forzato
    config.augment.brightness = 0.0
    config.augment.contrast = 0.0
    config.augment.gamma = 0.0
    config.augment.noise_std = 0.0
    config.augment.blur_prob = 0.0
    config.augment.shift_frames = 0

    window = np.zeros((8, keys.STRIP_H, keys.STRIP_W), dtype=np.uint8)
    window[:, :, 0] = 255                           # una colonna chiara all'estremo sinistro
    events = targets.Events(
        onset_frame=np.array([4]),
        off_frame=np.array([6]),
        key=np.array([0], dtype=np.int8),
        velocity=np.array([90], dtype=np.uint8),
    )
    strips, dense, start = data.make_example(window, events, 0, config, np.random.default_rng(0), True)

    assert start == 0
    assert strips.shape[0] == 8
    assert strips[:, 0, :, -1].max() == pytest.approx(1.0, abs=1e-3)  # la colonna e' finita a destra
    assert dense[0, 4, keys.N_KEYS - 1] == 1.0                        # e l'onset e' sull'ultimo tasto
    assert dense[0, 4, 0] == 0.0


def test_finestra_troppo_corta():
    config = data.DatasetConfig(clip_frames=64)
    window = np.zeros((10, keys.STRIP_H, keys.STRIP_W), dtype=np.uint8)
    events = targets.Events(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.uint8))
    with pytest.raises(ValueError):
        data.make_example(window, events, 0, config, np.random.default_rng(0), True)
