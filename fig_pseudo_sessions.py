"""Figure du rapport (annexe) : répartition des pseudo-sessions entre les 5 plis du découpage par
pseudo-sessions. Une ligne par espèce, un segment par pseudo-session (largeur = nombre de fenêtres).
(a) couleur = pli dans lequel la pseudo-session sert de validation ;
(b) premier pli : validation en couleur, entraînement en gris.
Lancer depuis le dossier du projet : python figures/fig_pseudo_sessions.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import GroupKFold

g = np.load('cache/noise_groups.npy')
y = pd.read_csv('DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
classes = np.unique(y)

data = {}
for c in classes:
    idx = np.flatnonzero(y == c)
    gc = g[idx]
    fold_of = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=5).split(idx, groups=gc)):
        for grp in set(gc[va]):
            fold_of[grp] = k
    sizes = pd.Series(gc).value_counts()
    data[c] = sorted([(int(sizes[grp]), fold_of[grp]) for grp in sizes.index], key=lambda t: (t[1], -t[0]))

FOLD_COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
GREY = '#cfcdc4'
GAP = 25  # espace blanc entre deux segments, en nombre de fenêtres

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for panel, ax in enumerate(axes):
    for row, c in enumerate(classes[::-1]):
        x = 0
        for n, f in data[c]:
            if panel == 0:
                col = FOLD_COL[f]
            else:
                col = FOLD_COL[0] if f == 0 else GREY
            ax.barh(row, n, left=x, height=0.72, color=col, edgecolor='none')
            x += n + GAP
        if panel == 1:
            v = sum(n for n, f in data[c] if f == 0)
            ax.text(x + 120, row, f'{100 * v / 11312:.0f} %', va='center', fontsize=9, color='#52514e')
    ax.set_yticks(range(len(classes)), classes[::-1])
    ax.set_xlabel('Nombre de fenêtres')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 12500)
axes[0].set_title("(a) Pli où chaque pseudo-session sert de validation", fontsize=10)
axes[1].set_title('(b) Premier pli : validation et entraînement', fontsize=10)
axes[0].legend(handles=[Patch(color=FOLD_COL[k], label=f'pli {k + 1}') for k in range(5)],
               ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9)
axes[1].legend(handles=[Patch(color=FOLD_COL[0], label='validation'), Patch(color=GREY, label='entraînement')],
               ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig('rapport/figures/pseudo_sessions.png', dpi=200)
print('figure : rapport/figures/pseudo_sessions.png')
