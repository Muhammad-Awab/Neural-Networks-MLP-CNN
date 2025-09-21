# mlp_binary.py
import math
import random
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# ------------- Utilities -------------
def set_seed(seed: int = 42): 
    random.seed(seed) #With the same seed, Python’s random calls will always give the same sequence.
    torch.manual_seed(seed) #Sets the seed for generating random numbers. Returns a torch.Generator object.
                            #Tells PyTorch: “Whenever you make random numbers on the CPU (your main processor), always start from the same point
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.




def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """For BCEWithLogits: threshold sigmoid(logits) at 0.5"""
    #torch.sigmoid(logits) -> Turns raw outputs (which can be any number, like -2.3 or +5.6) into probabilities between 0 and 1.
    #If probability ≥ 0.5 → predict class 1.
    #If probability < 0.5 → predict class 0.
    #view(-1) -> Reshapes the tensor to be a 1D array, making it easier to compare predictions with true labels.
    #long() -> Converts boolean values (True/False) to integers (1/0) for comparison.
    #preds == y.view(-1).long() -> Compares predicted classes with true labels, resulting in a tensor of boolean values (True for correct predictions, False for incorrect ones).
    #float().mean().item() -> Converts boolean values to floats (1.0 for True, 0.0 for False), calculates the mean accuracy, and retrieves it as a standard Python float.
    preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    return (preds == y.view(-1).long()).float().mean().item()

# ------------- Data (synthetic) -------------
def make_blobs(n_per_class=2000, dim=2, gap=3.0, std=1.0, seed=42):
    """
    Two Gaussian blobs:
      class 0 centered at (-gap/2, 0),
      class 1 centered at ( gap/2, 0)
    """
    # seed for reproducibility
    # torch.Generator().manual_seed(seed) -> Creates a random number generator that produces the same sequence of numbers every time you use the same seed.
    # This is useful for ensuring that your experiments are reproducible, meaning you can get the same results each time you run your code.
    # torch.randn((n_per_class, dim), generator=g) -> Generates random numbers that follow a normal distribution (bell curve) with a mean of 0 and a standard deviation of 1.
    # Multiply by std → makes them more spread out or tighter   
    # Add mean0 / mean1 → moves them left or right (two clusters).
    # torch.cat([...], dim=0) -> Combines the two sets of random numbers (one for each class) into a single dataset.
    # torch.zeros(n_per_class) / torch.ones(n_per_class) -> Creates labels for the two classes: 0 for the first class and 1 for the second class.
    g = torch.Generator().manual_seed(seed)

    #Concept show in image
    mean0 = torch.tensor([-gap / 2.0, 0.0])[:dim]
    mean1 = torch.tensor([+gap / 2.0, 0.0])[:dim]
    x0 = torch.randn((n_per_class, dim), generator=g) * std + mean0
    x1 = torch.randn((n_per_class, dim), generator=g) * std + mean1
    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)], dim=0)
    return X, y

# ------------- Model -------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # logits for binary class
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

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
        optimizer.zero_grad() #clears old gradients from the last step (otherwise they would accumulate).
        logits = model(xb) #forward pass through the network; outputs raw predictions (logits).
        loss = criterion(logits, yb.float()) #computes the loss between predictions and targets.
        loss.backward()  #backpropagation; computes gradients of the loss w.r.t. model parameters.
        optimizer.step() #updates model parameters using computed gradients.
      
        #loss.item() → converts tensor loss to a Python float.
        # Multiply by xb.size(0) → weight by number of samples in batch. In PyTorch (and most deep learning frameworks), the 0-th dimension is the batch dimension. rows
        # accuracy_from_logits → a function you presumably wrote to compute accuracy from logits.
        # logits.detach() → prevents gradient tracking for accuracy calculation.
        # n += xb.size(0) → keeps track of total samples processed.
        total_loss += loss.item() * xb.size(0) #loss.item() gives the scalar value of the loss for the current batch. Multiplying by xb.size(0) (the batch size) gives the total loss for that batch.
        total_acc += accuracy_from_logits(logits.detach(), yb) * xb.size(0)
        n += xb.size(0) #n is total number of samples processed so far
    return total_loss / n, total_acc / n

@torch.no_grad() # This decorator tells PyTorch not to compute gradients during the execution of this function. This is useful during evaluation because it saves memory and computation time, as we don't need gradients when we're
def eval_epoch(model, loader, criterion, device):
    model.eval() # Sets the model to evaluation mode. This is important because certain layers, like dropout and batch normalization, behave differently during training and evaluation.
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

    # 1) data 
    #Generates a synthetic 2D dataset with blobs for binary classification.
    X, y = make_blobs(n_per_class=3000, dim=2, gap=3.0, std=1.2, seed=42)
    dataset = TensorDataset(X, y)
    n_total = len(dataset)
    n_train = int(0.7 * n_total) # 70% for training
    n_val   = int(0.15 * n_total) # 15% for validation
    n_test  = n_total - n_train - n_val # 15% for testing
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size)

    # 2) model, loss, optimizer
    model = MLP(in_dim=2, hidden=32).to(cfg.device)
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
        #detach().cpu().clone() → ensures the weights are copied safely and can be saved or restored.
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
