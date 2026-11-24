"""Pseudo-sessions d'enregistrement : clustering du spectre du bruit de fond (bords sans clic).

Pour chaque clic, on calcule le log-spectre (Welch) des 2048 premiers échantillons, loin du clic,
puis on fait un k-means par classe. Chaque groupe ~ des conditions d'enregistrement proches.
Résultat : cache/noise_groups.npy (un identifiant de groupe global par clic).
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FS = 200_000
EDGE = 2048
K = 10   # groupes par classe


def edge_spectra(X, chunk=4096):
    out = []
    for i in range(0, len(X), chunk):
        xb = np.asarray(X[i:i+chunk, :EDGE], dtype=np.float64)
        _, p = welch(xb, fs=FS, nperseg=256, axis=1)
        out.append(np.log10(p + 1e-30).astype(np.float32))
    return np.concatenate(out)


def main():
    X = np.load('./DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
    labels = pd.read_csv('./DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
    S = edge_spectra(X)
    print('spectres du bruit :', S.shape)

    groups = np.empty(len(labels), dtype=np.int64)
    for c_i, c in enumerate(np.unique(labels)):
        m = labels == c
        Z = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(S[m]))
        km = KMeans(K, n_init=10, random_state=0).fit(Z)
        groups[m] = c_i * K + km.labels_
        sizes = np.sort(np.bincount(km.labels_, minlength=K))[::-1]
        sil = silhouette_score(Z, km.labels_, sample_size=3000, random_state=0)
        print(f'{c:4s} tailles des groupes {sizes.tolist()}  silhouette {sil:.2f}')

    os.makedirs('./cache', exist_ok=True)
    np.save('./cache/noise_groups.npy', groups)
    print('Groupes écrits dans cache/noise_groups.npy')


if __name__ == '__main__':
    main()
