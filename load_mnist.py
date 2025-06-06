import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, data_dir="~/mnist_data"):
    """
    Downloads (if needed) and returns PyTorch DataLoader objects
    for MNIST train and test sets.
    """

    # 1) Define a transform: convert PIL→Tensor and normalize to [0,1] then mean‐std normalize
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),                              # → Tensor in [0,1]
        transforms.Normalize((0.1307,), (0.3081,))          # normalizing with MNIST’s mean/std
    ])

    # 2) Download/load the training set
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,                 # this is the training set
        download=True,              # download if not already in data_dir
        transform=mnist_transform   # apply above transform
    )

    # 3) Download/load the test set
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=mnist_transform
    )

    # 4) Wrap them in DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # shuffle training data each epoch
        num_workers=2,     # on macOS, 0 or 2 usually works fine
        pin_memory=False    # speeds up host→GPU transfers if you use a GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage: iterate through one batch to verify it works
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    print(f"Loaded one batch of MNIST:")
    print(f"  images.shape = {images.shape}")  # e.g., torch.Size([64, 1, 28, 28])
    print(f"  labels.shape = {labels.shape}")  # e.g., torch.Size([64])