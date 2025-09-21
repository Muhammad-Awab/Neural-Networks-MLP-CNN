# cnn_mnist.py
import random
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------- Utilities -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- Model -------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,28,28] The tensor shape is written as [B, C, H, W] = Batch, Channels, Height, Width. 28*28 is the image size of the input.
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,32,14,14] shrink the picture
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B,64,7,7] Reduce size → Fewer numbers → Faster training, less memory.
        )
        self.head = nn.Sequential(
            nn.Flatten(), # Flatten the 3D tensor to 1D tensor for the fully connected layer.
            #64 = number of channels (features the CNN learned)
            #7*7 = spatial dimensions of the feature maps
            #128 = number of neurons in the hidden layer
            #The network learns how to map those 3136 features → 128 useful combinations, before making the final prediction.
            nn.Linear(64 * 7 * 7, 128), # 64 channels, each of size 7*7
            nn.ReLU(),
            nn.Linear(128, 10)  # logits for 10 classes (0-9 digits)
        )

    def forward(self, x):
        x = self.feat(x)
        return self.head(x)

# ------------- Train / Eval -------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        # Compute accuracy
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, total_correct / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, total_correct / n

def main():
    set_seed(42)
    cfg = TrainConfig()

#Makes training faster and more stable.
# Ensures all features (pixels) have similar scale.
# Helps gradients flow better.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])


    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(cfg.device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=(cfg.device=="cuda"))

    model = SimpleCNN().to(cfg.device)
    criterion = nn.CrossEntropyLoss() #CrossEntropyLoss expects logits and handles softmax internally.
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, cfg.device)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:5.2f}%  "
              f"test_loss={te_loss:.4f}  test_acc={te_acc*100:5.2f}%")

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print(f"\nSaved best model to mnist_cnn.pt (best test acc ≈ {best_acc*100:.2f}%).")

if __name__ == "__main__":
    main()
