# mlp_binary_real.py
import random
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ----------------- Utilities -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    return (preds == y.view(-1).long()).float().mean().item()

# ----------------- Dataset -----------------
def load_real_binary_dataset(seed: int = 42):
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # torch tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X, y

# ----------------- Model -----------------
class MLP(nn.Module):
    def __init__(self, in_dim=30, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Softplus(),               # ✅ best activation from random search
            nn.Linear(hidden, 1)     # output layer
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ----------------- Training -----------------
@dataclass
class TrainConfig:
    batch_size: int = 128             # ✅ match random_search
    lr: float = 0.0005                # ✅ best learning rate
    epochs: int = 100                 # ✅ same as random_search
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_acc += accuracy_from_logits(logits.detach(), yb) * xb.size(0)
        n += xb.size(0)
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb.float())
        total_loss += loss.item() * xb.size(0)
        total_acc += accuracy_from_logits(logits, yb) * xb.size(0)
        n += xb.size(0)
    return total_loss / n, total_acc / n

def main():
    set_seed(42)
    cfg = TrainConfig()

    # data
    X, y = load_real_binary_dataset(seed=42)
    dataset = TensorDataset(X, y)
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size)

    # model, loss, optimizer
    in_dim = X.shape[1]
    model = MLP(in_dim=in_dim, hidden=32).to(cfg.device)        # ✅ best hidden size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)       # ✅ best optimizer

    # training loop
    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, cfg.device)

        if va_loss < best_val_loss:
            best_val_loss, best_state = va_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:5.2f}%  "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:5.2f}%")

    # test
    if best_state:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, cfg.device)
    print(f"\nTest: loss={te_loss:.4f}, acc={te_acc*100:5.2f}%")

if __name__ == "__main__":
    main()
