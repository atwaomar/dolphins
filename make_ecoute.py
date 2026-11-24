"""Fichiers d'écoute ralentis x10 : 5 fenêtres brutes tirées au hasard par espèce, mises bout à bout.

Les signaux sont échantillonnés à 200 kHz. En écrivant les mêmes échantillons dans un WAV déclaré
à 20 kHz, la lecture est 10 fois plus lente : toutes les fréquences sont divisées par 10
(0-100 kHz -> 0-10 kHz, audible). Chaque fenêtre de 41 ms dure 0,41 s, chaque fichier 2,05 s.

Commande d'origine (session du 06/10/2026), recopiée telle quelle.
Lancer depuis le dossier du projet : python ecoute/make_ecoute.py
"""
import os

import numpy as np
import pandas as pd
from scipy.io import wavfile

FS_ORIG = 200_000   # Hz
SLOW = 10           # facteur de ralentissement
N_PER_SPECIES = 5

os.makedirs('ecoute', exist_ok=True)
X = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('DOCC10_Ytrain.csv')['TARGET'].values
rng = np.random.default_rng(0)
for sp in sorted(set(y)):
    idx = np.sort(rng.choice(np.where(y == sp)[0], N_PER_SPECIES, replace=False))
    clips = [np.asarray(X[i]) for i in idx]                 # signal brut, sans filtrage
    clips = [c / np.abs(c).max() * 0.9 for c in clips]      # chaque fenêtre ramenée à 90 % de la pleine échelle
    sig = np.concatenate(clips)
    wavfile.write(f'ecoute/{sp}_ralenti_x{SLOW}.wav', FS_ORIG // SLOW, (sig * 32767).astype(np.int16))
    print(sp, idx.tolist())
