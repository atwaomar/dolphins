"""Figure du rapport (choix de la fréquence de coupure du passe-haut, 5 kHz).

Pour 1500 fenêtres brutes par espèce (8192 éch. à 200 kHz) :
  - clic  = 512 éch. centrés sur le pic de l'enveloppe ;
  - bruit = 512 éch. pris sur le bord le plus éloigné du clic.
(a) spectre moyen (fenêtre de Hann) du clic et du bruit, moyenné en dB sur les espèces ;
(b) part de l'énergie du bruit située sous f, pour chaque espèce.

Lancer depuis le dossier du projet : python figures/fig_coupure.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

FS, HALF, N, FC = 200_000, 256, 1500, 5_000
rng = np.random.default_rng(0)
X = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
win = np.hanning(2 * HALF)
f = np.fft.rfftfreq(2 * HALF, 1 / FS)

clic_db, bruit_db, cum_bruit = [], [], {}
for c in np.unique(y):
    idx = np.sort(rng.choice(np.flatnonzero(y == c), N, replace=False))
    xb = np.asarray(X[idx], dtype=np.float64)
    xb -= xb.mean(1, keepdims=True)
    env = uniform_filter1d(np.abs(hilbert(xb, axis=1)), 64, axis=1)
    pk = env[:, HALF:-HALF].argmax(1) + HALF
    clic = np.take_along_axis(xb, pk[:, None] + np.arange(-HALF, HALF), axis=1)
    start = np.where(pk > X.shape[1] // 2, 0, X.shape[1] - 2 * HALF)
    bruit = np.take_along_axis(xb, start[:, None] + np.arange(2 * HALF), axis=1)
    pc = (np.abs(np.fft.rfft(clic * win, axis=1)) ** 2).mean(0)
    pb = (np.abs(np.fft.rfft(bruit * win, axis=1)) ** 2).mean(0)
    ref = pc.max()                                  # même référence pour clic et bruit d'une espèce
    clic_db.append(10 * np.log10(pc / ref))
    bruit_db.append(10 * np.log10(pb / ref))
    cum_bruit[c] = 100 * np.cumsum(pb) / pb.sum()
    print(f'{c:4s} bruit sous {FC/1e3:.0f} kHz : {cum_bruit[c][f < FC][-1]:.1f} %')

clic_db, bruit_db = np.mean(clic_db, 0), np.mean(bruit_db, 0)
fk = f / 1e3
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.axvspan(0, FC / 1e3, color='0.88', lw=0)
ax.plot(fk, clic_db, color='#1f4e79', lw=1.4, label='clic')
ax.plot(fk, bruit_db, color='#c0504d', lw=1.4, label='bruit')
ax.axvline(FC / 1e3, color='k', ls='--', lw=0.9)
ax.text(FC / 1e3 + 0.8, ax.get_ylim()[1] - 3, '$f_c = 5$ kHz', va='top')
ax.set_xlim(0, 50)
ax.set_xlabel('Fréquence (kHz)')
ax.set_ylabel('Énergie relative (dB)')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig('rapport/figures/coupure_spectres.png', dpi=200)

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.axvspan(0, FC / 1e3, color='0.88', lw=0)
for c, cum in cum_bruit.items():
    ax.plot(fk, cum, lw=1.1, label=c)
ax.axvline(FC / 1e3, color='k', ls='--', lw=0.9)
ax.set_xlim(0, 15)
ax.set_ylim(80, 100.5)
ax.set_xlabel('Fréquence $f$ (kHz)')
ax.set_ylabel('Énergie du bruit sous $f$ (%)')
ax.legend(ncol=2, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig('rapport/figures/coupure_energie_bruit.png', dpi=200)
