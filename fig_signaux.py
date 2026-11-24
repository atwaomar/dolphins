"""Figures du rapport (chapitre 2) : fenêtres brutes, une par espèce, puis plusieurs d'une même espèce.

Lancer depuis le dossier du projet : python figures/fig_signaux.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 200_000
INK = '#1f4e79'
X = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
t_ms = np.arange(X.shape[1]) / FS * 1e3
rng = np.random.default_rng(1)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9})


def trace(ax, x, titre):
    x = np.asarray(x, dtype=np.float64)
    ax.plot(t_ms, x / np.abs(x).max(), lw=0.4, color=INK)
    ax.set_title(titre, pad=2)
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


# (1) une fenêtre tirée au hasard par espèce
fig, axes = plt.subplots(2, 5, figsize=(10, 3.3), sharex=True)
for ax, c in zip(axes.flat, np.unique(y)):
    trace(ax, X[rng.choice(np.flatnonzero(y == c))], c)
for ax in axes[1]:
    ax.set_xlabel('Temps (ms)')
fig.tight_layout(h_pad=0.6, w_pad=0.8)
fig.savefig('rapport/figures/signaux_par_espece.png', dpi=200)

# (2) cinq fenêtres d'une même espèce
c = 'GG'
idx = rng.choice(np.flatnonzero(y == c), 5, replace=False)
fig, axes = plt.subplots(1, 5, figsize=(10, 1.75), sharex=True)
for k, (ax, i) in enumerate(zip(axes, idx)):
    trace(ax, X[i], f'{c}, fenêtre {k + 1}')
    ax.set_xlabel('Temps (ms)')
fig.tight_layout(w_pad=0.8)
fig.savefig('rapport/figures/signaux_meme_espece.png', dpi=200)
