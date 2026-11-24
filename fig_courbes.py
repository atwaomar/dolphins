"""Figure du rapport : courbes d'apprentissage du CNN 1D (fenêtre de 1024), moyenne ± écart-type
sur les 5 plis de validation croisée, pour le découpage aléatoire et par pseudo-sessions.
Lancer depuis le dossier du projet : python figures/fig_courbes.py
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STYLE = {'random': ('#1f4e79', 'découpage aléatoire'), 'groups': ('#c0504d', 'pseudo-sessions')}


def read(split):
    loss, acc = [], []
    for k in range(5):
        txt = open(f'cv_logs/cnn_crop1024_{split}_fold{k}.txt', encoding='utf-8', errors='replace').read()
        ep = re.findall(r'epoch\s+\d+\s+loss ([\d.]+)\s+val acc ([\d.]+)', txt)
        loss.append([float(a) for a, _ in ep])
        acc.append([100 * float(b) for _, b in ep])
    return np.array(loss), np.array(acc)


plt.rcParams.update({'font.size': 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
for split, (col, lab) in STYLE.items():
    loss, acc = read(split)
    e = np.arange(1, loss.shape[1] + 1)
    for ax, v in ((a1, loss), (a2, acc)):
        m, s = v.mean(0), v.std(0)
        ax.plot(e, m, color=col, lw=1.6, label=lab)
        ax.fill_between(e, m - s, m + s, color=col, alpha=0.2, lw=0)
    print(f'{split} : accuracy finale {acc[:, -1].mean():.1f} ± {acc[:, -1].std():.1f} %, '
          f'meilleure époque {acc.max(1).mean():.1f} ± {acc.max(1).std():.1f} %')
a1.set_xlabel('Époque'); a1.set_ylabel("Perte d'entraînement")
a2.set_xlabel('Époque'); a2.set_ylabel('Accuracy en validation (%)')
for ax in (a1, a2):
    ax.grid(alpha=0.3)
# une seule légende, commune aux deux panneaux
handles, labels = a2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig('rapport/figures/courbes_apprentissage.png', dpi=200)
