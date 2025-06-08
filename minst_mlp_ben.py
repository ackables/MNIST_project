import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_loader(training_cycles, data_dir="./MNIST_dataset"):
    mnist_tensor = transforms.Compose(
        transforms.ToTensor
    )

    training_set = datasets.MNIST(root=data_dir, train=True, download=True, target_transform=mnist_tensor)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, target_transform=mnist_tensor)

    # sets batch size to be the number of training set elements divided by the number of training cycles
    # cast as an integer to eliminate partially filled batches
    batch_size = int((len(training_set)/training_cycles))

    print(batch_size)

    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3
    )

    return training_loader, test_loader

class MLP(nn.Module):
    def __init__(self, activations, nouts, nin=28**2):
        super(MLP, self).__init__()
        # put nin into a list and append it with nouts list
        dims = [nin]+nouts
        for n in range(len(nouts)): # for the number of items in nouts, create layers connecting number of inputs to next number in list
            self.layers = nn.ModuleList(nn.Linear(dims[n], dims[n+1]))

        # add activations list to self
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        # flatten MNIST images to flat vectors
        x = x.view(x.size(0), -1)

        # apply activation functions to each layer of the MLP
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x)) # eg F.relu(self.activations(x))



def main():
    # number of training cycles
    training_cycles = 200

    # set number of neurons at each layer of MLP
    nouts = [256, 512, 128, 10]

    # set activation functions for MLP layers
    activations = [
        F.ReLU,
        F.Tanh,
        F.Tanh,
        F.Identity
    ]

    training_loader = mnist_loader(training_cycles=training_cycles)

    model = MLP(nouts=nouts)


if __name__ == "__main__":
    main()