"""CNN 1D sur les clics prétraités (filtre passe-haut, recentrage, crop configurable, normalisation)."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

FS = 200_000

# ex. : python train_cnn.py --crop 1024 --no-filter
parser = argparse.ArgumentParser()
parser.add_argument('--crop', type=int, default=512, help='taille de la fenêtre découpée')
parser.add_argument('--no-filter', action='store_true', help='pas de passe-haut 5 kHz')
parser.add_argument('--no-recentre', action='store_true', help='découpe fixe au centre (4096) au lieu du pic de l\'enveloppe')
parser.add_argument('--no-norm', action='store_true', help='pas de normalisation par le max (seulement une échelle globale)')
parser.add_argument('--noise-swap', type=float, default=0.0,
                    help="proba de coller le clic dans le bruit d'un autre signal (signal complet 8192 uniquement)")
parser.add_argument('--split', choices=['random', 'groups', 'all'], default='random',
                    help="random : 80/20 aléatoire ; groups : on retire des pseudo-sessions entières (voir make_groups.py) ; "
                         "all : entraînement sur 100 %% des données, sans validation (on garde la dernière époque)")
parser.add_argument('--fold', type=int, default=None,
                    help="validation croisée à 5 plis : numéro du pli (0 à 4) gardé en validation ; "
                         "sans cette option, découpage historique (80/20 ou 1er pli des pseudo-sessions)")
ARGS = parser.parse_args()
HALF = ARGS.crop // 2
TAG = f'crop{ARGS.crop}' + '_nofilter' * ARGS.no_filter + '_norecentre' * ARGS.no_recentre + '_nonorm' * ARGS.no_norm
SPLIT_TAG = {'random': '', 'groups': '_groups', 'all': '_all'}[ARGS.split]
SPLIT_TAG += f'_fold{ARGS.fold}' * (ARGS.fold is not None)
TAG += f'_swap{ARGS.noise_swap:g}' * (ARGS.noise_swap > 0)
CLICK_HALF = 512   # zone du clic gardée lors de l'échange de bruit (comme dans swap_test.py)
FADE = 64
EPOCHS = 30
BATCH = 256
LR = 3e-3
SEED = 0
CACHE = './cache'

sos = butter(4, 5_000, btype='highpass', fs=FS, output='sos')


def preprocess(X, half=HALF, batch=2000):
    out = np.empty((len(X), 2*half), dtype=np.float32)
    for i in range(0, len(X), batch):
        xb = np.asarray(X[i:i+batch], dtype=np.float64)
        if not ARGS.no_filter:
            xb = sosfiltfilt(sos, xb, axis=1)
        if ARGS.no_recentre:
            pk = np.full(len(xb), X.shape[1] // 2)
        else:
            env = uniform_filter1d(np.abs(hilbert(xb, axis=1)), 64, axis=1)
            pk = env[:, half:-half].argmax(1) + half
        idx = pk[:, None] + np.arange(-half, half)
        seg = np.take_along_axis(xb, idx, axis=1)
        if not ARGS.no_norm:
            seg /= np.abs(seg).max(1, keepdims=True) + 1e-9
        out[i:i+batch] = seg
    return out


def load_preprocessed(name, path):
    cached = os.path.join(CACHE, f'{name}_{TAG}.npy')
    if not os.path.exists(cached):
        os.makedirs(CACHE, exist_ok=True)
        t0 = time.time()
        np.save(cached, preprocess(np.load(path, mmap_mode='r')))
        print(f'{name}: prétraitement {time.time() - t0:.0f} s')
    # mmap : on ne charge pas tout en RAM (3,7 Go pour les fenêtres de 8192)
    return np.load(cached, mmap_mode='r')


def to_gpu(X, idx, scale, dev, chunk=4096):
    """Copie les lignes idx de X (mmap) sur le GPU en bfloat16, par morceaux, sans tout charger en RAM."""
    out = torch.empty((len(idx), 1, X.shape[1]), dtype=torch.bfloat16, device=dev)
    for i in range(0, len(idx), chunk):
        xb = np.asarray(X[idx[i:i+chunk]], dtype=np.float32) / scale
        out[i:i+chunk, 0] = torch.from_numpy(xb).to(dev, dtype=torch.bfloat16)
    return out


def make_split(y):
    """Indices train / validation triés (lecture séquentielle du mmap)."""
    if ARGS.split == 'all':
        # modèle final : toutes les données étiquetées, pas de validation
        return np.arange(len(y)), np.array([], dtype=np.int64)
    if ARGS.split == 'random' and ARGS.fold is None:
        # même découpage 80/20 que les baselines
        tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    elif ARGS.split == 'random':
        # validation croisée stratifiée à 5 plis (mêmes plis que cv_baselines.py)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        tr_idx, va_idx = folds[ARGS.fold]
    else:
        # pour chaque classe, ~20 % des pseudo-sessions partent entières en validation
        # (pli ARGS.fold d'un GroupKFold à 5 plis ; pli 0 par défaut)
        groups = np.load(os.path.join(CACHE, 'noise_groups.npy'))
        va = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            _, va_c = list(GroupKFold(n_splits=5).split(idx, groups=groups[idx]))[ARGS.fold or 0]
            va.append(idx[va_c])
        va_idx = np.concatenate(va)
        tr_idx = np.setdiff1d(np.arange(len(y)), va_idx)
    return np.sort(tr_idx), np.sort(va_idx)


class Block(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=k//2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k//2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.max_pool1d(x, 2)


class ClickCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            Block(1, 32, 9),     # 512 -> 256
            Block(32, 64, 7),    # 256 -> 128
            Block(64, 128, 5),   # 128 -> 64
            Block(128, 128, 3),  # 64 -> 32
            Block(128, 256, 3),  # 32 -> 16
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2*256, n_classes))

    def forward(self, x):
        x = self.features(x)
        # moyenne + max global : insensible à la position exacte du clic
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)
        return self.head(x)


def click_positions(X, half=CLICK_HALF, batch=2000):
    """Position du clic (pic de l'enveloppe après passe-haut) dans chaque fenêtre brute, mise en cache."""
    cached = os.path.join(CACHE, f'click_pos_half{half}.npy')
    if os.path.exists(cached):
        return np.load(cached)
    pos = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        xb = sosfiltfilt(sos, np.asarray(X[i:i+batch], dtype=np.float64), axis=1)
        env = uniform_filter1d(np.abs(hilbert(xb, axis=1)), 64, axis=1)
        pos[i:i+batch] = env[:, half:-half].argmax(1) + half
    np.save(cached, pos)
    return pos


def noise_swap(x, pos, p):
    """Avec la proba p, garde la zone du clic de x et remplace le reste par le bruit d'un autre signal
    du batch, ramené au même niveau de bruit. x : (B, 1, L), pos : (B,) positions du clic."""
    B, L = x.shape[0], x.shape[-1]
    donor = x[torch.randperm(B, device=x.device)].float()
    xf = x.float()
    rstd = lambda v: 1.4826 * v.abs().median(dim=-1, keepdim=True).values + 1e-6
    donor = donor * rstd(xf) / rstd(donor)
    d = (torch.arange(L, device=x.device)[None] - pos[:, None]).abs()[:, None, :]
    m = ((CLICK_HALF - d) / FADE).clamp(0, 1)
    m = 0.5 - 0.5 * torch.cos(torch.pi * m)                  # fondu en cosinus, comme swap_test.py
    keep = (torch.rand(B, 1, 1, device=x.device) >= p).float()
    m = torch.maximum(m, keep)                              # m = 1 partout : pas d'échange
    return m * xf + (1 - m) * donor


def augment(x):
    """x : (B, 1, L) sur GPU."""
    B = x.shape[0]
    # décalage temporel aléatoire de ±32 échantillons
    shifts = torch.randint(-32, 33, (B,), device=x.device)
    idx = (torch.arange(x.shape[-1], device=x.device)[None] - shifts[:, None]) % x.shape[-1]
    x = torch.gather(x, 2, idx[:, None, :])
    # inversion de polarité
    sign = torch.where(torch.rand(B, 1, 1, device=x.device) < 0.5, -1.0, 1.0)
    # amplitude + bruit
    scale = torch.empty(B, 1, 1, device=x.device).uniform_(0.7, 1.3)
    noise = torch.randn_like(x) * torch.empty(B, 1, 1, device=x.device).uniform_(0, 0.05)
    return x * sign * scale + noise


@torch.no_grad()
def predict(model, X, batch=None):
    model.eval()
    batch = batch or max(128, 2**20 // X.shape[-1])   # batchs plus petits pour les longues fenêtres
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i+batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # moyenne avec la version inversée (test-time augmentation)
            logits = model(xb).float().softmax(1) + model(-xb).float().softmax(1)
        out.append(logits)
    return torch.cat(out)


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
    # sans normalisation par signal, on ramène quand même les données à une échelle raisonnable
    sample = np.asarray(Xp[np.sort(np.random.choice(tr_idx, 2000, replace=False))], dtype=np.float32)
    scale = float(sample.std()) if ARGS.no_norm else 1.0
    Xtr = to_gpu(Xp, tr_idx, scale, dev)
    Xva = to_gpu(Xp, va_idx, scale, dev)
    ytr = torch.from_numpy(ytr).to(dev)
    if ARGS.noise_swap > 0:
        assert ARGS.crop == 8192 and ARGS.no_recentre, "l'échange de bruit suppose le signal complet non recentré"
        pos_tr = torch.from_numpy(click_positions(np.load('./DOCC10_train/DOCC10_Xtrain.npy', mmap_mode='r'))[tr_idx]).to(dev)

    model = ClickCNN(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * ((len(Xtr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    print(f'{TAG}{SPLIT_TAG}, {sum(p.numel() for p in model.parameters())/1e3:.0f}k paramètres, {len(Xtr)} train / {len(Xva)} val')

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr), device=dev)
        total_loss = 0
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i+BATCH]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                xb = Xtr[b]
                if ARGS.noise_swap > 0:
                    xb = noise_swap(xb, pos_tr[b], ARGS.noise_swap)
                loss = F.cross_entropy(model(augment(xb)), ytr[b], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(b)

        if len(va_idx) == 0:
            # pas de validation : on garde les poids de la dernière époque
            torch.save(model.state_dict(), f'cnn1d_best_{TAG}{SPLIT_TAG}.pt')
            print(f'epoch {epoch+1:2d}  loss {total_loss/len(Xtr):.4f}  ({time.time()-t0:.0f} s)', flush=True)
            continue
        acc = accuracy_score(yva, predict(model, Xva).argmax(1).cpu().numpy())
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'cnn1d_best_{TAG}{SPLIT_TAG}.pt')
        print(f'epoch {epoch+1:2d}  loss {total_loss/len(Xtr):.4f}  val acc {acc:.4f}  ({time.time()-t0:.0f} s)')

    model.load_state_dict(torch.load(f'cnn1d_best_{TAG}{SPLIT_TAG}.pt'))
    if len(va_idx) > 0:
        pred = predict(model, Xva).argmax(1).cpu().numpy()
        print(f'\nMeilleure val acc : {best_acc:.4f}\n')
        print(classification_report(yva, pred, target_names=classes, digits=3))
        print(pd.DataFrame(confusion_matrix(yva, pred), index=classes, columns=classes))

    # prédictions sur le test (inutiles pour les plis de validation croisée)
    if ARGS.fold is not None:
        return
    del Xtr, Xva
    Xte_np = load_preprocessed('test', './DOCC10_test/DOCC10_Xtest.npy')
    Xte = to_gpu(Xte_np, np.arange(len(Xte_np)), scale, dev)
    y_test = classes[predict(model, Xte).argmax(1).cpu().numpy()]
    pd.DataFrame({'TARGET': y_test}, index=np.arange(len(y_test))).to_csv(
        f'./DOCC10_Ytest_pred_cnn_{TAG}{SPLIT_TAG}.csv', index_label='ID')
    print(f'Prédictions test écrites dans DOCC10_Ytest_pred_cnn_{TAG}{SPLIT_TAG}.csv')


if __name__ == '__main__':
    main()
