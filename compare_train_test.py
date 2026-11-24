"""Le jeu de test vient-il des mêmes conditions d'enregistrement que l'entraînement ?

Trois vérifications, sur le bruit de fond (2048 premiers échantillons, loin du clic) et sur le clic :
  1. validation adverse : un classifieur apprend à distinguer une fenêtre de train d'une fenêtre de
     test ; AUC ~ 0,5 -> indiscernables, AUC proche de 1 -> conditions différentes ;
  2. distance au plus proche voisin : une fenêtre de test a-t-elle un « jumeau » de bruit dans le
     train aussi proche qu'une fenêtre de train en a un ?
  3. niveau du bruit (écart-type robuste) dans les deux jeux.
Lancer depuis le dossier du projet : python compare_train_test.py
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FS, EDGE, HALF, N = 200_000, 2048, 256, 20_000
rng = np.random.default_rng(0)
Xtr = np.load('DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
Xte = np.load('DOCC10_test/DOCC10_Xtest.npy', mmap_mode='r')
itr = np.sort(rng.choice(len(Xtr), N, replace=False))
ite = np.sort(rng.choice(len(Xte), N, replace=False))
sos = butter(4, 5_000, btype='highpass', fs=FS, output='sos')


def features(X, idx, chunk=2000):
    """log-spectre du bruit (bord), log-spectre du clic (filtré, recentré), niveau du bruit."""
    noise, click, level = [], [], []
    for i in range(0, len(idx), chunk):
        x = np.asarray(X[idx[i:i+chunk]], dtype=np.float64)
        e = x[:, :EDGE] - x[:, :EDGE].mean(1, keepdims=True)
        noise.append(np.log10(welch(e, fs=FS, nperseg=256, axis=1)[1] + 1e-30))
        level.append(np.log10(1.4826 * np.median(np.abs(e), axis=1) + 1e-12))
        xf = sosfiltfilt(sos, x, axis=1)
        env = uniform_filter1d(np.abs(hilbert(xf, axis=1)), 64, axis=1)
        pk = env[:, HALF:-HALF].argmax(1) + HALF
        c = np.take_along_axis(xf, pk[:, None] + np.arange(-HALF, HALF), axis=1)
        c /= np.abs(c).max(1, keepdims=True) + 1e-12
        click.append(np.log10(np.abs(np.fft.rfft(c * np.hanning(2 * HALF), axis=1)) + 1e-6))
    return np.concatenate(noise), np.concatenate(click), np.concatenate(level)


ntr, ctr, ltr = features(Xtr, itr)
nte, cte, lte = features(Xte, ite)
print(f'{N} fenêtres de train et {N} fenêtres de test')

# 1. validation adverse
y = np.r_[np.zeros(N), np.ones(N)]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nom, A, B in [('bruit de fond', ntr, nte), ('clic', ctr, cte)]:
    F = np.r_[A, B]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), F, y,
                          cv=cv, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, p)
    print(f'[1] validation adverse sur le {nom:13s} : AUC = {auc:.3f}  '
          f'(fenêtres de test reconnues avec proba > 0,9 : {np.mean(p[y == 1] > 0.9):.1%})')

# 2. plus proche voisin dans le train, sur le bruit (espace PCA standardisé)
half = N // 2
sc = StandardScaler().fit(ntr)
pca = PCA(20, random_state=0).fit(sc.transform(ntr))
Z = lambda M: pca.transform(sc.transform(M))
ref, tr_q, te_q = Z(ntr[:half]), Z(ntr[half:]), Z(nte[:half])
nn = NearestNeighbors(n_neighbors=1).fit(ref)
d_tr = nn.kneighbors(tr_q)[0][:, 0]
d_te = nn.kneighbors(te_q)[0][:, 0]
seuil = np.quantile(d_tr, 0.95)
print(f'[2] distance au plus proche bruit de train : médiane train {np.median(d_tr):.2f}, '
      f'test {np.median(d_te):.2f} ; fenêtres de test au-delà du 95e centile du train : '
      f'{np.mean(d_te > seuil):.1%} (5 % attendus si même distribution)')

# 3. niveau du bruit
q = [5, 25, 50, 75, 95]
print('[3] niveau du bruit log10(écart-type robuste), centiles', q)
print('    train :', np.round(np.percentile(ltr, q), 2))
print('    test  :', np.round(np.percentile(lte, q), 2))

# 4. même test, espèce par espèce (espèce du test = prédiction du modèle soumis), pour écarter
#    l'effet d'une simple différence de proportions d'espèces entre train et test
import pandas as pd
ytr_lab = pd.read_csv('DOCC10_Ytrain.csv')['TARGET'].values[itr]
yte_lab = pd.read_csv('soumission_DOCC10.csv')['TARGET'].values[ite]
aucs = []
for c in np.unique(ytr_lab):
    a, b = ntr[ytr_lab == c], nte[yte_lab == c]
    if len(b) < 100:
        continue
    F, yy = np.r_[a, b], np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), F, yy,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    aucs.append(roc_auc_score(yy, p))
    print(f'[4] {c:4s} bruit train ({len(a)}) contre test prédit {c} ({len(b)}) : AUC = {aucs[-1]:.3f}')
print(f'[4] AUC moyenne par espèce : {np.mean(aucs):.3f}')
