"""Bande de fréquence utile de chaque classe : spectre du clic comparé au spectre du bruit.

Pour chaque signal (8192 éch. à 200 kHz) :
  - clic  = 512 éch. centrés sur le pic de l'enveloppe
  - bruit = 512 éch. pris sur un bord (loin du clic)
On moyenne les DSP (Welch) par classe, puis on calcule :
  - f_pic    : fréquence du maximum de la DSP du clic
  - f5 / f95 : 5 % et 95 % de l'énergie "clic - bruit" (bande utile)
  - f_snr    : première fréquence où le clic dépasse le bruit de 6 dB
La fréquence de coupure du passe-haut doit rester sous min(f5) et min(f_snr) de toutes les classes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, welch

FS = 200_000
HALF = 256
N_PER_CLASS = 2000   # sous-échantillon par classe (suffisant pour une DSP moyenne)
rng = np.random.default_rng(0)

X = np.load('./DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('./DOCC10_Ytrain.csv', index_col=0)['TARGET'].values

rows, curves = [], {}
for c in np.unique(y):
    idx = np.sort(rng.choice(np.flatnonzero(y == c), min(N_PER_CLASS, (y == c).sum()), replace=False))
    xb = np.asarray(X[idx], dtype=np.float64)
    xb -= xb.mean(1, keepdims=True)

    env = uniform_filter1d(np.abs(hilbert(xb, axis=1)), 64, axis=1)
    pk = env[:, HALF:-HALF].argmax(1) + HALF
    clic = np.take_along_axis(xb, pk[:, None] + np.arange(-HALF, HALF), axis=1)
    # bruit : le bord le plus éloigné du clic
    start = np.where(pk > X.shape[1] // 2, 0, X.shape[1] - 2 * HALF)
    bruit = np.take_along_axis(xb, start[:, None] + np.arange(2 * HALF), axis=1)

    f, p_clic = welch(clic, fs=FS, nperseg=256, axis=1)
    _, p_bruit = welch(bruit, fs=FS, nperseg=256, axis=1)
    p_clic, p_bruit = p_clic.mean(0), p_bruit.mean(0)

    utile = np.clip(p_clic - p_bruit, 0, None)
    cum = np.cumsum(utile) / utile.sum()
    snr_db = 10 * np.log10(p_clic / p_bruit)
    rows.append(dict(classe=c,
                     f_pic_kHz=f[p_clic.argmax()] / 1e3,
                     f5_kHz=f[np.searchsorted(cum, 0.05)] / 1e3,
                     f95_kHz=f[np.searchsorted(cum, 0.95)] / 1e3,
                     f_snr6dB_kHz=f[np.argmax(snr_db > 6)] / 1e3))
    curves[c] = (p_clic, p_bruit)

df = pd.DataFrame(rows).set_index('classe').round(1)
print(df.to_string())
print(f"\n=> coupure passe-haut max conseillée : {min(df.f5_kHz.min(), df.f_snr6dB_kHz.min()):.1f} kHz")

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
for ax, (c, (pc, pb)) in zip(axes.flat, curves.items()):
    ax.semilogy(f / 1e3, pc, label='clic')
    ax.semilogy(f / 1e3, pb, label='bruit', alpha=.7)
    ax.axvline(df.loc[c, 'f5_kHz'], color='k', ls='--', lw=.8)
    ax.set_title(c)
    ax.set_xlabel('kHz')
axes.flat[0].legend()
plt.tight_layout()
plt.savefig('./figures/dsp_par_classe.png', dpi=120)
print('figure : figures/dsp_par_classe.png')
