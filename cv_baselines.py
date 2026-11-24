"""Validation croisée à 5 plis des modèles de référence (moyenne et écart-type de la précision).

Entrée : fenêtres prétraitées de 512 échantillons (filtre 5 kHz, recentrage, normalisation),
mises en cache par train_cnn.py --crop 512 (cache/train_crop512.npy).
Mêmes 5 plis stratifiés que train_cnn.py --fold k (StratifiedKFold, graine 0).
Lancer depuis le dossier du projet : python cv_baselines.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.load('cache/train_crop512.npy').astype(np.float32)
y = pd.read_csv('DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
logspec = np.log10(np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) + 1e-6).astype(np.float32)
print(X.shape, logspec.shape)

models = {
    'Ridge, signal temporel': (lambda: RidgeClassifier(alpha=10), X),
    'Ridge, log-spectre': (lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=10)), logspec),
    'Gradient boosting, log-spectre': (lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, random_state=0), logspec),
}
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
rows = []
for name, (make, F) in models.items():
    acc, f1 = [], []
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        pred = make().fit(F[tr], y[tr]).predict(F[va])
        acc.append(accuracy_score(y[va], pred))
        f1.append(f1_score(y[va], pred, average='macro'))
        print(f'{name} pli {k} : précision {acc[-1]:.4f}  ({time.time() - t0:.0f} s)', flush=True)
    rows.append({'modèle': name, 'précision moy.': np.mean(acc), 'écart-type': np.std(acc),
                 'F1 macro moy.': np.mean(f1), 'écart-type F1': np.std(f1)})
print()
print(pd.DataFrame(rows).set_index('modèle').round(4).to_string())
