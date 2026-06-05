#!/usr/bin/env python3
# =============================================================================
# reference_pytorch.py — ROZWIĄZANIE REFERENCYJNE (KM2)
# =============================================================================
# Identyczna sieć i hiperparametry jak nasza biblioteka w Julii — punkt
# odniesienia do porównania w artykule (dokładność, czas, pamięć).
#
# Sieć (jak AWID-2026-CNN.ipynb / nasz CustomAwid):
#   Conv(1->6, 3x3, pad=1, bias=False) -> MaxPool(2)
#   Conv(6->16, 3x3, pad=1, bias=False) -> MaxPool(2)
#   Flatten -> Linear(784->84) -> ReLU -> Dropout(0.4) -> Linear(84->10)
# Hiperparametry: 3 epoki, lr=0.01, SGD (bez momentum), batch=20, CrossEntropy.
# Liczone na CPU (nasza biblioteka Julia też CPU).
#
# Instalacja (CPU):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Uruchomienie:
#   python3 reference/reference_pytorch.py           # ziarno 0
#   python3 reference/reference_pytorch.py 0 1 2 3    # kilka ziaren -> średnia ± odch.
# =============================================================================

import sys
import time
import resource
import statistics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cpu")  # CPU dla porównania z biblioteką Julia

EPOCHS = 3
LR = 0.01
BATCH = 20

# Ziarna z argumentów (domyślnie pojedyncze 0):
SEEDS = [int(a) for a in sys.argv[1:]] or [0]

# --- Dane (FashionMNIST), wczytane raz ---------------------------------------
tf = transforms.ToTensor()  # obrazy do [0,1], kształt (1, 28, 28)
train_ds = datasets.FashionMNIST(root="reference/data", train=True,  download=True, transform=tf)
test_ds  = datasets.FashionMNIST(root="reference/data", train=False, download=True, transform=tf)


def build_model():
    # Domyślna inicjalizacja Conv2d/Linear w PyTorch to Kaiming/He.
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(784, 84),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(84, 10),
    ).to(DEVICE)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).argmax(dim=1)
            correct += (pred == yb.to(DEVICE)).sum().item()
            total += yb.numel()
    return correct / total


def train_once(seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)  # powtarzalne tasowanie
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, generator=g)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False)

    model = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)   # zwykły SGD, bez momentum
    loss_fn = nn.CrossEntropyLoss()                    # uśrednia po batchu (jak nasze /B)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
    dt = time.perf_counter() - t0
    return evaluate(model, test_loader), dt


# --- Pętla po ziarnach ------------------------------------------------------
print(f"[ref PyTorch] device={DEVICE}, wątki={torch.get_num_threads()}, "
      f"epoki={EPOCHS}, lr={LR}, batch={BATCH}, ziarna={SEEDS}")

accs = []
for s in SEEDS:
    acc, dt = train_once(s)
    accs.append(acc * 100)
    print(f"ziarno {s}: dokładność = {acc*100:5.2f}%   (czas 3 epok: {dt:5.1f} s)")

if len(accs) > 1:
    print(f"\nDokładność: średnia = {statistics.mean(accs):.2f}% "
          f"± {statistics.stdev(accs):.2f} pp  (n={len(accs)} ziaren)")

peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB na Linux
print(f"Szczytowa pamięć procesu (RSS): {peak_kib/1024:.0f} MiB")
