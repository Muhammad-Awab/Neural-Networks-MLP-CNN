import random
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
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

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_val,   y_val   = torch.from_numpy(X_val),   torch.from_numpy(y_val)
    X_test,  y_test  = torch.from_numpy(X_test),  torch.from_numpy(y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ----------------- Model Builder -----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ----------------- Training -----------------
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

# ----------------- Random Search -----------------
def random_search(num_trials=10, seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_real_binary_dataset(seed=seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=128)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

    # Search space
    activations = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU,nn.Softplus]
    optimizers = [optim.SGD, optim.Adam, optim.RMSprop, optim.AdamW]
    hidden_options = [[16], [32], [64], [128], [32, 32], [64, 32], [128, 64]]

    best_val_acc, best_cfg = 0.0, None

    for trial in range(num_trials):
        act = random.choice(activations)
        opt_cls = random.choice(optimizers)
        hidden = random.choice(hidden_options)
        lr = random.choice([1e-3, 1e-2, 5e-3])

        model = MLP(in_dim=X_train.shape[1], hidden_sizes=hidden, activation=act).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = opt_cls(model.parameters(), lr=lr)

        # Train few epochs for speed
        for epoch in range(10):
            train_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Trial {trial+1}: act={act.__name__}, opt={opt_cls.__name__}, hidden={hidden}, lr={lr}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = (act, opt_cls, hidden, lr, model.state_dict())

    # Test best
    best_act, best_opt, best_hidden, best_lr, best_state = best_cfg
    best_model = MLP(in_dim=X_train.shape[1], hidden_sizes=best_hidden, activation=best_act).to(device)
    best_model.load_state_dict(best_state)
    test_loss, test_acc = eval_epoch(best_model, test_loader, nn.BCEWithLogitsLoss(), device)

    print(f"\nBest Config -> act={best_act.__name__}, opt={best_opt.__name__}, hidden={best_hidden}, lr={best_lr}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    random_search(num_trials=15)



# mlp_binary_compare.py
# import random
# from dataclasses import dataclass
# from sklearn.model_selection import train_test_split
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler

# # ---------- Utilities ----------
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
#     preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
#     return (preds == y.view(-1).long()).float().mean().item()

# # ---------- Data ----------
# def load_real_binary_dataset(seed: int = 42):
#     data = load_breast_cancer()
#     X = data.data.astype(np.float32)
#     y = data.target.astype(np.float32)

#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, test_size=0.3, random_state=seed, stratify=y
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train).astype(np.float32)
#     X_val   = scaler.transform(X_val).astype(np.float32)
#     X_test  = scaler.transform(X_test).astype(np.float32)

#     return (torch.from_numpy(X_train), torch.from_numpy(y_train)), \
#            (torch.from_numpy(X_val),   torch.from_numpy(y_val)), \
#            (torch.from_numpy(X_test),  torch.from_numpy(y_test))

# # ---------- Model ----------
# class MLP(nn.Module):
#     def __init__(self, in_dim=30, hidden=32, activation="relu"):
#         super().__init__()
#         act_layer = {
#             "relu": nn.ReLU(),
#             "leakyrelu": nn.LeakyReLU(0.1),
#             "sigmoid": nn.Sigmoid(),
#             "tanh": nn.Tanh(),
#             "softplus": nn.Softplus(),
#         }[activation.lower()]

#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             act_layer,
#             nn.Linear(hidden, hidden),
#             act_layer,
#             nn.Linear(hidden, 1)
#         )

#     def forward(self, x):
#         return self.net(x).squeeze(1)

# # ---------- Training ----------
# @dataclass
# class TrainConfig:
#     batch_size: int = 128
#     lr: float = 1e-2
#     epochs: int = 15
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

# def train_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     total_loss, total_acc, n = 0.0, 0.0, 0
#     for xb, yb in loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb.float())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * xb.size(0)
#         total_acc += accuracy_from_logits(logits.detach(), yb) * xb.size(0)
#         n += xb.size(0)
#     return total_loss / n, total_acc / n

# @torch.no_grad()
# def eval_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss, total_acc, n = 0.0, 0.0, 0
#     for xb, yb in loader:
#         xb, yb = xb.to(device), yb.to(device)
#         logits = model(xb)
#         loss = criterion(logits, yb.float())
#         total_loss += loss.item() * xb.size(0)
#         total_acc += accuracy_from_logits(logits, yb) * xb.size(0)
#         n += xb.size(0)
#     return total_loss / n, total_acc / n

# # ---------- Experiment Runner ----------
# def run_experiment(activation, optimizer_name, train_loader, val_loader, test_loader, in_dim, cfg):
#     model = MLP(in_dim=in_dim, hidden=32, activation=activation).to(cfg.device)
#     criterion = nn.BCEWithLogitsLoss()

#     # choose optimizer
#     if optimizer_name == "sgd":
#         optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
#     elif optimizer_name == "momentum":
#         optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
#     elif optimizer_name == "nesterov":
#         optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True)
#     elif optimizer_name == "rmsprop":
#         optimizer = optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.9)
#     elif optimizer_name == "adam":
#         optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
#     elif optimizer_name == "adamw":
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer_name}")

#     # training loop
#     best_val_loss, best_state = float("inf"), None
#     for epoch in range(1, cfg.epochs + 1):
#         tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
#         va_loss, va_acc = eval_epoch(model, val_loader, criterion, cfg.device)
#         if va_loss < best_val_loss:
#             best_val_loss, best_state = va_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
#     if best_state:
#         model.load_state_dict(best_state)

#     te_loss, te_acc = eval_epoch(model, test_loader, criterion, cfg.device)
#     return te_loss, te_acc

# # ---------- Main ----------
# def main():
#     set_seed(42)
#     cfg = TrainConfig()
#     (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_real_binary_dataset()

#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
#     val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=cfg.batch_size)
#     test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size)

#     in_dim = X_train.shape[1]
#     activations = ["relu", "leakyrelu", "sigmoid", "tanh", "softplus"]
#     optimizers  = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "adamw"]

#     results = []
#     for act in activations:
#         for opt in optimizers:
#             te_loss, te_acc = run_experiment(act, opt, train_loader, val_loader, test_loader, in_dim, cfg)
#             results.append((act, opt, te_loss, te_acc))
#             print(f"Activation={act:8s} Optimizer={opt:8s} --> Test Loss={te_loss:.4f} Test Acc={te_acc*100:5.2f}%")

#     print("\n=== Final Comparison ===")
#     for act, opt, loss, acc in results:
#         print(f"{act:8s} | {opt:8s} | loss={loss:.4f} | acc={acc*100:5.2f}%")

# if __name__ == "__main__":
#     main()
