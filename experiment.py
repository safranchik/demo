import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from cubyc import Run

training_data = datasets.FashionMNIST(
    root='.data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='.data',
    train=False,
    download=True,
    transform=ToTensor()
)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


def train_loop(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


def eval_loop(model, test_loader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


@Run(tags=['fashion-mnist'], remote="https://github.com/safranchik/demo.git")
def experiment_run(learning_rate: float, weight_decay: float):
    train_loader = DataLoader(training_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop(model, criterion, optimizer, train_loader, epochs=1)
    test_accuracy = eval_loop(model, test_loader)

    for i in range(5):
        yield {'test_accuracy': test_accuracy / (i+1)}


if __name__ == '__main__':

    for learning_rate in [1e-2, 1e-3, 1e-4]:
        for weight_decay in [0, 1e-5, 1e-4]:
            experiment_run(learning_rate=learning_rate, weight_decay=weight_decay)
