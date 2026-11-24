"""Part de l'énergie du signal filtré (passe-haut 5 kHz) contenue dans une fenêtre centrée sur le
maximum de l'enveloppe, pour plusieurs tailles de fenêtre (chapitres 2 et 3 du rapport).
Lancer depuis le dossier du projet : python energie_fenetre.py
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

FS, N = 200_000, 1000
rng = np.random.default_rng(0)
X = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
y = pd.read_csv('DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
sos = butter(4, 5_000, btype='highpass', fs=FS, output='sos')
HALF_MAX = 512
rows = []
for c in np.unique(y):
    idx = np.sort(rng.choice(np.flatnonzero(y == c), N, replace=False))
    x = sosfiltfilt(sos, np.asarray(X[idx], dtype=np.float64), axis=1)
    env = uniform_filter1d(np.abs(hilbert(x, axis=1)), 64, axis=1)
    pk = env[:, HALF_MAX:-HALF_MAX].argmax(1) + HALF_MAX
    cs = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x ** 2, axis=1)], axis=1)
    tot = cs[:, -1]
    rr = np.arange(len(x))
    r = {'espèce': c}
    for w in (256, 512, 1024, 2048):
        a = np.clip(pk - w // 2, 0, X.shape[1])
        b = np.clip(pk + w // 2, 0, X.shape[1])
        r[f'{w}'] = 100 * np.mean((cs[rr, b] - cs[rr, a]) / tot)
    rows.append(r)
df = pd.DataFrame(rows).set_index('espèce')
print(df.round(1).to_string())
print('moyenne :', df.mean().round(1).to_dict())
print('part de la fenêtre dans le signal (%) :', {w: round(100 * w / 8192, 1) for w in (256, 512, 1024, 2048)})
