"""Small MNIST CNN used as an evaluation oracle for image-generation samples.

Trains once on real MNIST and is then loaded by `utils.eval_utils` to compute
class-accuracy and class-marginal KL on generated images. Not part of the flow
matching model — purely an evaluation utility.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Same normalization as the image trainer (maps to roughly [-1, 1]).
_MNIST_MEAN = (0.5,)
_MNIST_STD = (0.5,)


class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _build_loaders(data_root: str, batch_size: int, num_workers: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_classifier(data_root='./data', save_path='eval_assets/mnist_cnn.pt',
                     epochs=3, batch_size=128, lr=1e-3, num_workers=2, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, test_loader = _build_loaders(data_root, batch_size, num_workers)
    model = MnistCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"epoch {epoch}: test acc {correct/total:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Saved classifier to {save_path}")
    return model


def load_classifier(path='eval_assets/mnist_cnn.pt', device='cpu'):
    model = MnistCNN()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    train_classifier(
        data_root=args.data_root, save_path=args.save_path,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers,
    )
