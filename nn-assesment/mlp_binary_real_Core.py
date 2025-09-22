# mlp_binary_real.py
import random
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# NEW: sklearn for a real dataset + scaling
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ------------- Utilities -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Binary accuracy for BCEWithLogitsLoss: sigmoid -> threshold 0.5"""
    preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    return (preds == y.view(-1).long()).float().mean().item()

# ------------- Data (real) -------------
def load_real_binary_dataset(seed: int = 42):
    """
    Load Breast Cancer dataset.
    Split -> then fit scaler only on train -> then transform all splits.
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float32)   # [N, 30]
    y = data.target.astype(np.float32) # [N], 0/1

    # 1) split first
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    # 2) fit scaler only on training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # 3) convert to tensors
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_val,   y_val   = torch.from_numpy(X_val),   torch.from_numpy(y_val)
    X_test,  y_test  = torch.from_numpy(X_test),  torch.from_numpy(y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ------------- Model -------------
class MLP(nn.Module):
    def __init__(self, in_dim=30, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),# input layer -> hidden layer (fully connected) 
            nn.Tanh(),              # activation function (non-linear)
            nn.Linear(hidden, hidden), # hidden layer -> hidden layer
            nn.Linear(hidden, 1)  # logits layer (output layer, single neuron for binary classification)
        )
    def forward(self, x):
        return self.net(x).squeeze(1) # [B] (squeeze to remove extra dim) .net(x) -> Passes the input data x through the entire neural network defined in self.net.

# ------------- Training -------------
@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-2
    epochs: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad() # clears old gradients from the last step (otherwise they would accumulate).
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

    # 1) data (REAL instead of blobs)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_real_binary_dataset(seed=42)
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size)

    # 2) model, loss, optimizer
    in_dim = X_train.shape[1]   # 30 for this dataset
    model = MLP(in_dim=in_dim, hidden=32).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 3) train
    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, cfg.device)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:5.2f}%  "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:5.2f}%")

    # 4) test
    if best_state:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, cfg.device)
    print(f"\nTest: loss={te_loss:.4f}, acc={te_acc*100:5.2f}%")

if __name__ == "__main__":
    main()

