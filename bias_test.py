"""Test du biais d'enregistrement : le bruit de fond, sans le clic, suffit-il à prédire l'espèce ?

On entraîne le même CNN sur les 2048 premiers échantillons de chaque fenêtre brute (0 à 10 ms),
loin du clic (centré en 4096, ou vers 5600 pour PM).
Lancer : python bias_test.py [--split groups]
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report

from train_cnn import BATCH, EPOCHS, LR, SEED, SPLIT_TAG, ClickCNN, augment, make_split, predict, to_gpu

EDGE = 2048


class EdgeView:
    """Expose seulement X[:, :EDGE] d'un mmap, pour réutiliser to_gpu."""
    def __init__(self, X):
        self.X = X
        self.shape = (X.shape[0], EDGE)

    def __getitem__(self, idx):
        return self.X[idx, :EDGE]


def transient_score(X, idx, chunk=4096):
    """max|x| / écart-type robuste : grand s'il y a un clic ou un transitoire dans le bord."""
    out = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), chunk):
        xb = np.abs(np.asarray(X[idx[i:i+chunk], :EDGE], dtype=np.float32))
        out[i:i+chunk] = xb.max(1) / (1.4826 * np.median(xb, 1) + 1e-12)
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda'

    X = np.load('./DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r')
    labels = pd.read_csv('./DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)

    # même découpage que les autres runs (--split random ou groups)
    tr_idx, va_idx = make_split(y)
    ytr, yva = y[tr_idx], y[va_idx]

    Xe = EdgeView(X)
    sample = np.asarray(Xe[np.sort(np.random.choice(tr_idx, 2000, replace=False))], dtype=np.float32)
    scale = float(sample.std())
    Xtr = to_gpu(Xe, tr_idx, scale, dev)
    Xva = to_gpu(Xe, va_idx, scale, dev)
    ytr_t = torch.from_numpy(ytr).to(dev)

    model = ClickCNN(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * ((len(Xtr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    print(f'bords 0:{EDGE}{SPLIT_TAG}, {len(Xtr)} train / {len(Xva)} val, hasard = {1/len(classes):.3f}')

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i+BATCH]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(augment(Xtr[b])), ytr_t[b], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy_score(yva, predict(model, Xva).argmax(1).cpu().numpy())
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'cnn1d_best_bias_edges{SPLIT_TAG}.pt')
        print(f'epoch {epoch+1:2d}  val acc {acc:.4f}  ({time.time()-t0:.0f} s)')

    model.load_state_dict(torch.load(f'cnn1d_best_bias_edges{SPLIT_TAG}.pt'))
    pred = predict(model, Xva).argmax(1).cpu().numpy()
    print(f'\nMeilleure val acc sur les bords : {best_acc:.4f}\n')
    print(classification_report(yva, pred, target_names=classes, digits=3))

    # le bord contient-il parfois d'autres clics ? on sépare par quartile de "transitoire"
    score = transient_score(X, va_idx)
    q = np.quantile(score, [0.25, 0.5, 0.75])
    print('Précision selon la présence de transitoires dans le bord :')
    for name, mask in [('Q1 (bruit le plus pur)', score <= q[0]),
                       ('Q2', (score > q[0]) & (score <= q[1])),
                       ('Q3', (score > q[1]) & (score <= q[2])),
                       ('Q4 (transitoires forts)', score > q[2])]:
        print(f'  {name:25s} score <= {score[mask].max():6.1f}  acc {accuracy_score(yva[mask], pred[mask]):.4f}')
    print('\nPrécision par classe sur Q1 (bruit le plus pur) :')
    m = score <= q[0]
    for k, c in enumerate(classes):
        mk = m & (yva == k)
        print(f'  {c:4s} n={mk.sum():4d}  acc {accuracy_score(yva[mk], pred[mk]):.3f}')


if __name__ == '__main__':
    main()
