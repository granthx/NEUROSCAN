"""
Brain Tumour Segmentation (BraTS) - 2D U-Net in PyTorch
-------------------------------------------------------
Expected raw data layout (BraTS 2020/2021 style):
    data/raw/<CASE_ID>/<CASE_ID>_flair.nii.gz
                       <CASE_ID>_t1.nii.gz
                       <CASE_ID>_t1ce.nii.gz
                       <CASE_ID>_t2.nii.gz
                       <CASE_ID>_seg.nii.gz

Install:
    pip install torch nibabel numpy tqdm matplotlib

Usage:
    python brain_tumor_segmentation.py preprocess --raw data/raw --out data/processed
    python brain_tumor_segmentation.py train --data data/processed --epochs 30
    python brain_tumor_segmentation.py predict --data data/processed --case <CASE_ID> --ckpt best_model.pt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODALITIES = ["flair", "t1", "t1ce", "t2"]
CROP = 192          # centre crop (BraTS is 240x240)
NUM_CLASSES = 4     # 0=background, 1=necrotic core, 2=edema, 3=enhancing tumour


# ----------------------------------------------------------------------------
# 1. Preprocessing: NIfTI -> normalised .npy volumes
# ----------------------------------------------------------------------------
def preprocess(raw_dir: str, out_dir: str):
    import nibabel as nib

    raw, out = Path(raw_dir), Path(out_dir)
    cases = sorted(p for p in raw.iterdir() if p.is_dir())
    for case in tqdm(cases, desc="Preprocessing"):
        cid = case.name
        dst = out / cid
        if (dst / "img.npy").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)

        chans = []
        for m in MODALITIES:
            vol = nib.load(str(case / f"{cid}_{m}.nii.gz")).get_fdata().astype(np.float32)
            brain = vol > 0
            if brain.any():  # z-score inside brain only
                mu, sd = vol[brain].mean(), vol[brain].std() + 1e-8
                vol = np.where(brain, (vol - mu) / sd, 0)
            chans.append(vol)
        img = np.stack(chans)                                  # (4, H, W, D)
        seg = nib.load(str(case / f"{cid}_seg.nii.gz")).get_fdata().astype(np.uint8)
        seg[seg == 4] = 3                                      # remap label 4 -> 3

        # reorder to (C, D, H, W) and centre-crop H, W
        img = img.transpose(0, 3, 1, 2)
        seg = seg.transpose(2, 0, 1)
        h, w = img.shape[2:]
        t, l = (h - CROP) // 2, (w - CROP) // 2
        img = img[:, :, t:t + CROP, l:l + CROP]
        seg = seg[:, t:t + CROP, l:l + CROP]

        np.save(dst / "img.npy", img.astype(np.float16))
        np.save(dst / "seg.npy", seg)


# ----------------------------------------------------------------------------
# 2. Dataset (2D slices)
# ----------------------------------------------------------------------------
class BraTSSlices(Dataset):
    def __init__(self, data_dir, cases, augment=False, empty_keep=0.2, seed=0):
        self.root, self.augment = Path(data_dir), augment
        self.vols = {}
        self.index = []
        rng = random.Random(seed)
        for cid in cases:
            seg = np.load(self.root / cid / "seg.npy", mmap_mode="r")
            img = np.load(self.root / cid / "img.npy", mmap_mode="r")
            self.vols[cid] = (img, seg)
            for z in range(seg.shape[0]):
                if not np.any(img[:, z]):          # skip blank slices
                    continue
                if seg[z].any() or rng.random() < empty_keep:
                    self.index.append((cid, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        cid, z = self.index[i]
        img, seg = self.vols[cid]
        x = torch.from_numpy(np.array(img[:, z], dtype=np.float32))
        y = torch.from_numpy(np.array(seg[z], dtype=np.int64))
        if self.augment:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            if random.random() < 0.5:
                x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return x, y


# ----------------------------------------------------------------------------
# 3. Model: 2D U-Net
# ----------------------------------------------------------------------------
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=4, n_cls=NUM_CLASSES, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([block(in_ch, f[0])] + [block(f[i], f[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([block(f[i] * 2, f[i]) for i in range(4)])
        self.head = nn.Conv2d(f[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 4:
                skips.append(x)
                x = self.pool(x)
        for i in reversed(range(4)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


# ----------------------------------------------------------------------------
# 4. Loss + metrics
# ----------------------------------------------------------------------------
def dice_loss(logits, target, eps=1e-5):
    p = F.softmax(logits, 1)
    t = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    d = (2 * inter + eps) / (p.sum(dims) + t.sum(dims) + eps)
    return 1 - d[1:].mean()                      # ignore background


class ComboLoss(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def region_dice(pred, target, eps=1e-5):
    """Dice for BraTS regions: WT (whole), TC (core), ET (enhancing)."""
    regions = {
        "WT": lambda m: m > 0,
        "TC": lambda m: (m == 1) | (m == 3),
        "ET": lambda m: m == 3,
    }
    out = {}
    for k, fn in regions.items():
        p, t = fn(pred), fn(target)
        inter = (p & t).sum().item()
        out[k] = (2 * inter + eps) / (p.sum().item() + t.sum().item() + eps)
    return out


# ----------------------------------------------------------------------------
# 5. Training / validation
# ----------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    cases = sorted(p.name for p in Path(args.data).iterdir() if (p / "img.npy").exists())
    random.Random(42).shuffle(cases)
    n_val = max(1, int(0.15 * len(cases)))
    val_cases, train_cases = cases[:n_val], cases[n_val:]
    print(f"{len(train_cases)} train cases / {len(val_cases)} val cases")

    tr = BraTSSlices(args.data, train_cases, augment=True)
    va = BraTSSlices(args.data, val_cases, augment=False)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = ComboLoss()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x, y in tqdm(tl, desc=f"Epoch {ep}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device if device != "mps" else "cpu", enabled=(device == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()

        model.eval()
        agg = {"WT": [], "TC": [], "ET": []}
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu()
                for k, v in region_dice(pred, y).items():
                    agg[k].append(v)
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        mean_dice = float(np.mean(list(m.values())))
        print(f"loss {run/len(tl):.4f} | Dice WT {m['WT']:.3f} TC {m['TC']:.3f} ET {m['ET']:.3f} | mean {mean_dice:.3f}")
        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), "best_model.pt")
            print("  saved best_model.pt")


# ----------------------------------------------------------------------------
# 6. Inference + visualisation
# ----------------------------------------------------------------------------
def predict(args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    root = Path(args.data) / args.case
    img = np.load(root / "img.npy").astype(np.float32)
    seg = np.load(root / "seg.npy")
    zs = np.where(seg.reshape(seg.shape[0], -1).any(1))[0]
    z = int(zs[len(zs) // 2]) if len(zs) else seg.shape[0] // 2

    with torch.no_grad():
        pred = model(torch.from_numpy(img[:, z]).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()

    cmap = ListedColormap([(0, 0, 0, 0), "red", "green", "yellow"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[0, z], cmap="gray"); ax[0].set_title("FLAIR")
    ax[1].imshow(img[0, z], cmap="gray"); ax[1].imshow(seg[z], cmap=cmap, vmin=0, vmax=3); ax[1].set_title("Ground truth")
    ax[2].imshow(img[0, z], cmap="gray"); ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=3); ax[2].set_title("Prediction")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150)
    print("Saved prediction.png")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess"); p.add_argument("--raw", required=True); p.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=16); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=2)
    r = sub.add_parser("predict")
    r.add_argument("--data", required=True); r.add_argument("--case", required=True); r.add_argument("--ckpt", default="best_model.pt")

    a = ap.parse_args()
    {"preprocess": lambda: preprocess(a.raw, a.out), "train": lambda: train(a), "predict": lambda: predict(a)}[a.cmd]()
