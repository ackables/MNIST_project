import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) Loader function (as before)
def get_mnist_dataloaders(batch_size=64, data_dir="~/mnist_data"):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=mnist_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=mnist_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, test_loader


# 2) Define a simple MLP for MNIST (flatten 28×28 → hidden layers → 10 outputs)
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden1=256, hidden2=128, num_classes=10):
        super(MLP, self).__init__()
        # Linear layers 
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        # x comes in as shape [batch_size, 1, 28, 28]
        x = x.view(x.size(0), -1)      # flatten → [batch_size, 784]
        x = F.relu(self.fc1(x))        # first hidden layer + ReLU
        x = F.tanh(self.fc2(x))        # second hidden layer + tanh
        x = self.fc3(x)                # logits for 10 classes
        return x


# 3) Training & evaluation routines
def train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    criterion = nn.CrossEntropyLoss()  # standard for classification
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        # Move data to the chosen device (CPU or CUDA)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()           # zero out gradients from previous step
        output = model(data)            # forward pass → shape [batch_size, 10]
        loss = criterion(output, target)
        loss.backward()                 # backward pass (compute gradients)
        optimizer.step()                # update weights

        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(
                f"Train Epoch: {epoch:02d} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0


def evaluate(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum losses for computing average
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # no gradient computation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)             # [batch_size, 10]
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # predicted class index
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )


# 4) Main entry: set up device, loaders, model, optimizer, and loop over epochs
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # Device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    # Instantiate model, move to device
    model = MLP().to(device)

    # Optimizer (SGD) and maybe a scheduler if you like
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)

    # (Optional) Save the trained model
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("Model saved to mnist_mlp.pth")


if __name__ == "__main__":
    main()