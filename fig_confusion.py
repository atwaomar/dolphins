"""Figures du rapport : matrices de confusion du CNN 1D (crop 1024), lues dans les logs d'entraînement.

Chaque ligne est normalisée par le nombre de fenêtres de l'espèce (rappel, en %).
Lancer depuis le dossier du projet : python figures/fig_confusion.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    'cnn_log_crop1024.txt': 'rapport/figures/confusion_cnn1d.png',
    'cnn_log_crop1024_groups.txt': 'rapport/figures/confusion_cnn1d_groups.png',
}


def read_confusion(path):
    """Dernière matrice de confusion du log : un en-tête de classes puis une ligne par classe."""
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    head = max(i for i, l in enumerate(lines) if l.split()[:2] == ['GG', 'GMA'])
    classes = lines[head].split()
    rows = [lines[head + 1 + k].split() for k in range(len(classes))]
    assert [r[0] for r in rows] == classes
    return classes, np.array([[int(v) for v in r[1:]] for r in rows])


plt.rcParams.update({'font.size': 10})
for log, out in RUNS.items():
    classes, C = read_confusion(log)
    P = 100 * C / C.sum(1, keepdims=True)
    acc = 100 * np.trace(C) / C.sum()
    print(f'{log} : précision {acc:.1f} %')

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(P, cmap='Blues', vmin=0, vmax=100)
    for j in range(len(classes)):
        for k in range(len(classes)):
            if P[j, k] >= 0.5:
                ax.text(k, j, f'{P[j, k]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if P[j, k] > 55 else 'black')
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Espèce prédite')
    ax.set_ylabel('Espèce réelle')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('% des fenêtres de l\'espèce réelle')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
