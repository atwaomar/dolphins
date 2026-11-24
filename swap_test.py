"""Test d'échange : le modèle suit-il le clic ou le bruit de fond ?

Pour chaque signal A de la validation (découpage par pseudo-sessions), on fabrique une chimère :
  - la zone du clic de A (±512 échantillons autour du pic d'enveloppe, raccords en fondu),
  - collée dans le bruit d'un signal B, ramené au même niveau de bruit que A.
Deux cas :
  - contrôle : B de la même classe que A (mesure l'effet du collage lui-même) ;
  - échange  : B d'une autre classe.
On regarde si la prédiction suit la classe de A (le clic) ou celle de B (le bruit).
Lancer : python swap_test.py --split groups   (les modèles testés ont été entraînés avec ce découpage)
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, sosfiltfilt

from train_cnn import ARGS, SEED, ClickCNN, make_split, predict, preprocess, sos, to_gpu

HALF = 512     # zone du clic : ±512 échantillons
FADE = 64      # fondu aux raccords
CHUNK = 2000


def click_positions(X):
    xb = sosfiltfilt(sos, X, axis=1)
    env = uniform_filter1d(np.abs(hilbert(xb, axis=1)), 64, axis=1)
    return env[:, HALF:-HALF].argmax(1) + HALF


def robust_std(X):
    return 1.4826 * np.median(np.abs(X), axis=1, keepdims=True) + 1e-12


def chimeras(A, B):
    """Clic de A dans le bruit de B (B remis au niveau de bruit de A)."""
    pk = click_positions(A)
    t = np.arange(A.shape[1])[None, :]
    d = np.abs(t - pk[:, None])                                # distance au clic
    m = np.clip((HALF - d) / FADE, 0, 1)                       # 1 dans la zone, rampe de FADE, 0 dehors
    m = 0.5 - 0.5 * np.cos(np.pi * m)                          # fondu en cosinus
    B = B * robust_std(A) / robust_std(B)
    return m * A + (1 - m) * B


class Arr:
    """Petit adaptateur pour réutiliser to_gpu sur un tableau en mémoire."""
    def __init__(self, X):
        self.X, self.shape = X, X.shape

    def __getitem__(self, idx):
        return self.X[idx]


def load(path):
    model = ClickCNN(10).cuda()
    model.load_state_dict(torch.load(path))
    return model.eval()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = np.load('./DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
    labels = pd.read_csv('./DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)

    assert ARGS.split == 'groups', 'lancer avec --split groups : les modèles testés ont été entraînés ainsi'
    # même découpage et même échelle globale que l'entraînement du modèle « signal complet » par groupes
    tr_idx, va_idx = make_split(y)
    sample = np.asarray(X[np.sort(np.random.choice(tr_idx, 2000, replace=False))], dtype=np.float32)
    full_scale = float(sample.std())

    models = {
        'clic seul (1024)': (load('cnn1d_best_crop1024_groups.pt'), 'click'),
        'signal complet (8192)': (load('cnn1d_best_crop8192_nofilter_norecentre_nonorm_groups.pt'), 'full'),
    }
    # tous les modèles avec échange de bruit déjà entraînés par pseudo-sessions, par probabilité p croissante
    for p in ['0.25', '0.5', '0.75', '1']:
        swap_path = f'cnn1d_best_crop8192_nofilter_norecentre_nonorm_swap{p}_groups.pt'
        if os.path.exists(swap_path):
            models[f'signal complet + échange p={p} (8192)'] = (load(swap_path), 'full')

    # donneurs de bruit : même classe (contrôle) ou autre classe (échange), pris dans la validation
    rng = np.random.default_rng(SEED)
    yv = y[va_idx]
    same, other = np.empty_like(va_idx), np.empty_like(va_idx)
    for k in range(len(classes)):
        a = np.flatnonzero(yv == k)
        b = np.flatnonzero(yv != k)
        same[a] = va_idx[rng.permutation(a)]
        other[a] = va_idx[rng.choice(b, len(a))]

    results = {}
    for i in range(0, len(va_idx), CHUNK):
        ia = va_idx[i:i+CHUNK]
        A = np.asarray(X[ia], dtype=np.float64)
        for case, donors in [('controle', same[i:i+CHUNK]), ('echange', other[i:i+CHUNK])]:
            order = np.argsort(donors)
            B = np.empty_like(A)
            B[order] = np.asarray(X[np.sort(donors)], dtype=np.float64)   # lecture triée du mmap
            C = chimeras(A, B)
            for name, (model, kind) in models.items():
                if kind == 'click':
                    Xin = preprocess(C, half=512)
                    scale = 1.0
                else:
                    Xin = C.astype(np.float32)
                    scale = full_scale
                p = predict(model, to_gpu(Arr(Xin), np.arange(len(Xin)), scale, 'cuda')).argmax(1).cpu().numpy()
                results.setdefault((name, case), []).append(p)
        print(f'{min(i+CHUNK, len(va_idx))}/{len(va_idx)}', flush=True)

    ya, yb = y[va_idx], y[other]
    print('\n=== Prédiction sur les chimères (zone du clic de A + bruit de B) ===')
    rows = []
    for name in models:
        pc = np.concatenate(results[(name, 'controle')])
        pe = np.concatenate(results[(name, 'echange')])
        rows.append({'modèle': name,
                     'contrôle : acc (B même classe)': np.mean(pc == ya),
                     'échange : suit le clic (A)': np.mean(pe == ya),
                     'échange : suit le bruit (B)': np.mean(pe == yb),
                     'échange : autre': np.mean((pe != ya) & (pe != yb))})
    print(pd.DataFrame(rows).set_index('modèle').round(3).to_string())

    for name in models:
        pc = np.concatenate(results[(name, 'controle')])
        pe = np.concatenate(results[(name, 'echange')])
        print(f'\n--- {name} : par classe ---')
        per = []
        for k, c in enumerate(classes):
            a, b = ya == k, yb == k
            per.append({'classe': c,
                        'contrôle acc': np.mean(pc[a] == k),
                        'clic de c dans autre bruit -> c': np.mean(pe[a] == k),
                        'bruit de c sous autre clic -> c': np.mean(pe[b] == k)})
        print(pd.DataFrame(per).set_index('classe').round(3).to_string())


if __name__ == '__main__':
    main()
