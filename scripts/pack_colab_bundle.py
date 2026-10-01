"""Crea lo zip con codice e pesi per il notebook Colab.

    python3 scripts/pack_colab_bundle.py          # -> dist/pianovam_video2midi_bundle.zip

Dentro ci sono solo le cose che servono per l'inferenza: `src/`, i pesi del modello
(con `args.json` accanto, che dice il raggio temporale usato in addestramento) e
`requirements.txt`. Non c'e' il dataset e non ci sono le cache: sono ~1,5 MB.

Serve quando il notebook Colab non puo' clonare il repo (repo non pubblicato): si
carica lo zip a mano nella cella che lo chiede.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
PESI_PREDEFINITI = "runs/m6_2ch/last.pt"


def file_del_bundle(checkpoint: Path) -> list[tuple[Path, Path]]:
    """(sorgente, nome dentro lo zip) di tutto quello che serve per l'inferenza."""
    voci = [(sorgente, Path("src") / sorgente.name)
            for sorgente in sorted((RADICE / "src").glob("*.py"))]
    voci.append((RADICE / "requirements.txt", Path("requirements.txt")))
    cartella = Path("runs") / checkpoint.parent.name
    voci.append((checkpoint, cartella / checkpoint.name))
    args = checkpoint.with_name("args.json")            # senza, il modello a 2 canali
    if args.is_file():                                  # verrebbe alimentato male
        voci.append((args, cartella / "args.json"))
    return voci


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip di src/ + pesi per il notebook Colab")
    parser.add_argument("--checkpoint", default=PESI_PREDEFINITI,
                        help="pesi da mettere nello zip (default: m6, epoca 9)")
    parser.add_argument("--out", default="dist/pianovam_video2midi_bundle.zip")
    args = parser.parse_args()

    checkpoint = RADICE / args.checkpoint
    if not checkpoint.is_file():
        raise SystemExit(f"pesi non trovati: {checkpoint}")
    destinazione = RADICE / args.out
    destinazione.parent.mkdir(parents=True, exist_ok=True)

    voci = file_del_bundle(checkpoint)
    with zipfile.ZipFile(destinazione, "w", zipfile.ZIP_DEFLATED) as archivio:
        for sorgente, nome in voci:
            archivio.write(sorgente, str(nome))

    peso = destinazione.stat().st_size / 1e6
    print(f"{destinazione} ({peso:.1f} MB), {len(voci)} file:")
    for sorgente, nome in voci:
        print(f"  {nome}  ({sorgente.stat().st_size / 1e3:.0f} KB)")
    print()
    print("In Colab: carica questo zip nella cella 1 del notebook pianovam_video2midi.ipynb")
    print("(oppure mettilo su un URL e scrivilo in BUNDLE_URL).")


if __name__ == "__main__":
    main()
