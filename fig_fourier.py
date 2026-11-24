"""Figure du rapport (section 1.4) : une fenêtre brute et le module de sa TFD.

Lancer depuis le dossier du projet : python figures/fig_fourier.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 200_000  # Hz
X = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('DOCC10_Ytrain.csv')['TARGET'].values

# Fenêtre de GG où le clic ressort le plus du bruit (pic / médiane absolue), parmi 2000 tirées au hasard
rng = np.random.default_rng(0)
cand = rng.choice(np.where(y == 'GG')[0], 2000, replace=False)
score = [np.abs(X[i]).max() / np.median(np.abs(X[i])) for i in cand]
i = int(cand[int(np.argmax(score))])
x = np.asarray(X[i], dtype=np.float64)
print('fenêtre', i, 'classe', y[i])

t_ms = np.arange(len(x)) / FS * 1e3
Xk = np.fft.rfft(x)
f_khz = np.fft.rfftfreq(len(x), 1 / FS) / 1e3
mag_db = 20 * np.log10(np.abs(Xk) + 1e-12)

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(t_ms, x, lw=0.6, color='#1f4e79')
ax.set_xlabel('Temps (ms)')
ax.set_ylabel('Amplitude')
ax.set_xlim(0, t_ms[-1])
fig.tight_layout()
fig.savefig('rapport/figures/fourier_signal.png', dpi=200)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(f_khz, mag_db, lw=0.6, color='#1f4e79')
ax.set_xlabel('Fréquence (kHz)')
ax.set_ylabel(r'$20\log_{10}|X_k|$ (dB)')
ax.set_xlim(0, FS / 2e3)
fig.tight_layout()
fig.savefig('rapport/figures/fourier_spectre.png', dpi=200)
