"""Rete: CNN per frame (una feature per tasto) + TCN temporale dilata.

Perche' questo modello e non un backbone grosso: dopo la rettifica la
informazione utile e' un bordo locale largo ~12 px (il tasto che si abbassa e il
dito che lo copre). Non serve un'immagine da ImageNet, serve precisione
temporale. Quindi:

1. `KeyStem`: comprime la larghezza in modo *esatto* (una colonna per tasto,
   `adaptive_avg_pool` da 88*KEY_PX a 88) e schiaccia l'altezza con 4 conv
   stride-2. Nessun parametro sprecato, e la griglia dei tasti resta allineata.
2. `TemporalTCN`: conv 2D dilata (dilatazione solo sul tempo, kernel 3 sull'asse
   dei tasti) sui frame: vede +-32 frame (oltre 1 s a 60 fps) mescolando i tasti
   vicini, che e' quello che serve per accordi e trilli.
3. Quattro teste: onset, stato del tasto, rilascio, velocita'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import keys

HEADS = ("onset", "frame", "offset", "velocity")


class KeyStem(nn.Module):
    """Da (B, 1, H, W) a (B, C, 88): una feature per tasto, per frame."""

    def __init__(self, channels: tuple[int, ...] = (24, 48, 64, 96), in_channels: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], (5, 3), stride=(4, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        ]
        for prev, nxt in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv2d(prev, nxt, 3, stride=(2, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(nxt),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # una colonna per tasto: dalla larghezza della striscia alle 88 colonne
        x = F.adaptive_avg_pool2d(x, (x.shape[-2], keys.N_KEYS))
        x = self.body(x)
        return x.mean(dim=-2)                       # (B, C, 88)


class TemporalBlock(nn.Module):
    """Conv 2D dilata: il tempo e' l'asse "verticale", i tasti quello orizzontale."""

    def __init__(self, channels: int, kernel_t: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel_t - 1) // 2      # padding simmetrico: video offline, il futuro si vede
        self.conv = nn.Conv2d(
            channels, channels, (kernel_t, 3), padding=(pad, 1), dilation=(dilation, 1), bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.norm(F.gelu(self.conv(x))))


class KeyNet(nn.Module):
    """Striscia rettificata -> 4 mappe di probabilita' (T, 88)."""

    def __init__(
        self,
        stem_channels: tuple[int, ...] = (24, 48, 64, 96),
        temporal_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.stem = KeyStem(stem_channels, in_channels=in_channels)
        self.project = nn.Conv2d(self.stem.out_channels, temporal_channels, 1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(temporal_channels, 3, d, dropout) for d in dilations]
        )
        self.heads = nn.ModuleDict({name: nn.Conv2d(temporal_channels, 1, 1) for name in HEADS})

    def receptive_field_frames(self) -> int:
        return 1 + sum(2 * d for d in self.dilations)

    def init_head_bias(self, priors: dict[str, float]) -> None:
        """Inizializza i bias delle teste alla frequenza attesa del positivo.

        Con l'1-2% di frame positivi per tasto, partire da bias=0 fa collassare
        la BCE su "tutto negativo" nelle prime centinaia di passi: partire dal
        log-odds del prior evita di perdere quella fase.
        """
        for name, prior in priors.items():
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            nn.init.constant_(self.heads[name].bias, float(np.log(prior / (1 - prior))))

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        """strips: (B, T, H, W) -> {'onset'|'frame'|'offset'|'velocity': (B, T, 88)}."""
        batch, length = strips.shape[0], strips.shape[1]
        flat = strips.reshape(batch * length, *strips.shape[2:])      # (B*T, C, H, W)
        feat = self.stem(flat)                                      # (B*T, C, 88)
        feat = feat.reshape(batch, length, feat.shape[1], feat.shape[2]).permute(0, 2, 1, 3)
        shared = self.blocks(self.project(feat))                    # (B, C, T, 88)
        return {name: head(shared).squeeze(1) for name, head in self.heads.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def in_channels_from_checkpoint(path: str | Path) -> int:
    """Quanti canali ha il modello salvato: si legge dalla prima convoluzione."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    key = next(name for name in state if name.endswith("stem.body.0.weight"))
    return int(state[key].shape[1])


def load_checkpoint(path: str | Path, device: str = "cpu") -> "KeyNet":
    """Ricostruisce il modello con i canali giusti e carica i pesi."""
    net = KeyNet(in_channels=in_channels_from_checkpoint(path))
    net.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    return net.to(device).eval()


def temporal_radius_for_checkpoint(path: str | Path) -> int:
    """Raggio della linea di base usato in training (0 se il modello e' a un canale).

    Il valore viene scritto accanto ai pesi in `args.json`: senza, un modello a
    due canali verrebbe alimentato con le feature sbagliate e non funzionerebbe.
    """
    if in_channels_from_checkpoint(path) == 1:
        return 0
    sidecar = Path(path).with_name("args.json")
    if sidecar.exists():
        try:
            return int(json.loads(sidecar.read_text()).get("temporal_radius", 15))
        except (ValueError, TypeError, json.JSONDecodeError):
            return 15
    return 15

