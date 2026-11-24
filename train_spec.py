"""CNN 2D sur le spectrogramme des clics prétraités (modèle du domaine temps-fréquence).

Même prétraitement, même découpage, mêmes augmentations et même optimisation que train_cnn.py ;
seule l'entrée du réseau change : log |STFT|^2 (fenêtre de Hann de 64, pas de 8) au lieu du signal.
Lancer : python train_spec.py --crop 1024 [--split groups]
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_cnn import (ARGS, BATCH, EPOCHS, LR, SEED, SPLIT_TAG, TAG, augment, load_preprocessed,
                       make_split, predict, to_gpu)

N_FFT, HOP = 64, 8


class Block2d(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.max_pool2d(x, 2)


class SpecCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.register_buffer('win', torch.hann_window(N_FFT))
        self.bn_in = nn.BatchNorm2d(1)
        self.features = nn.Sequential(
            Block2d(1, 32),     # 33 x 129 -> 16 x 64
            Block2d(32, 64),    # -> 8 x 32
            Block2d(64, 128),   # -> 4 x 16
            Block2d(128, 256),  # -> 2 x 8
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2*256, n_classes))

    def spectrogram(self, x):
        with torch.autocast('cuda', enabled=False):
            S = torch.stft(x[:, 0].float(), N_FFT, HOP, window=self.win, center=True, return_complex=True)
            return torch.log(S.abs() ** 2 + 1e-6)[:, None]       # (B, 1, 33 fréquences, trames)

    def forward(self, x):
        x = self.features(self.bn_in(self.spectrogram(x)))
        x = torch.cat([x.mean((-2, -1)), x.amax((-2, -1))], dim=1)
        return self.head(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda'

    Xp = load_preprocessed('train', './DOCC10_train/DOCC10_Xtrain.npy')
    labels = pd.read_csv('./DOCC10_Ytrain.csv', index_col=0)['TARGET'].values
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)
    tr_idx, va_idx = make_split(y)
    ytr, yva = y[tr_idx], y[va_idx]
    Xtr = to_gpu(Xp, tr_idx, 1.0, dev)
    Xva = to_gpu(Xp, va_idx, 1.0, dev)
    ytr = torch.from_numpy(ytr).to(dev)

    model = SpecCNN(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * ((len(Xtr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    with torch.no_grad():
        shape = tuple(model.spectrogram(Xva[:2]).shape[2:])
    print(f'spectrogramme {shape} ({TAG}{SPLIT_TAG}), {sum(p.numel() for p in model.parameters())/1e3:.0f}k paramètres, '
          f'{len(Xtr)} train / {len(Xva)} val')

    out = f'cnnspec_best_{TAG}{SPLIT_TAG}.pt'
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr), device=dev)
        total_loss = 0
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i+BATCH]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(augment(Xtr[b])), ytr[b], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(b)
        acc = accuracy_score(yva, predict(model, Xva).argmax(1).cpu().numpy())
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out)
        print(f'epoch {epoch+1:2d}  loss {total_loss/len(Xtr):.4f}  val acc {acc:.4f}  ({time.time()-t0:.0f} s)', flush=True)

    model.load_state_dict(torch.load(out))
    pred = predict(model, Xva).argmax(1).cpu().numpy()
    print(f'\nMeilleure val acc : {best_acc:.4f}\n')
    print(classification_report(yva, pred, target_names=classes, digits=3))
    print(pd.DataFrame(confusion_matrix(yva, pred), index=classes, columns=classes))


if __name__ == '__main__':
    main()
