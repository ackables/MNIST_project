import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_norm_param_gen(data_dir="./MNIST_dataset"):
    training_set = training_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

    training_loader = DataLoader(
        training_set,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    sum = 0.0
    sum_sq = 0.0
    num_images = 0.0

    for images, _ in training_set:
        sum += images.sum()
        sum_sq += (images**2).sum()
        num_images += images.numel()

    mean = sum / num_images
    var = (sum_sq / num_images) - mean**2
    std = torch.sqrt(var)

    print(f"mean: {mean}\tstd: {std}")
    return mean, std



def mnist_loader(training_cycles, data_dir="./MNIST_dataset"):
    mean, std = mnist_norm_param_gen() # generate mean and std of MNIST dataset for normalization
    mnist_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    training_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_tensor)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=mnist_tensor)

    batch_size = 64


    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return training_loader, test_loader

class MLP(nn.Module):
    def __init__(self, activations, nouts, nin=28**2):
        super(MLP, self).__init__()
        # put nin into a list and append it with nouts list
        dims = [nin]+nouts

        # for the number of items in nouts, create layers connecting number of inputs to next number in list
        self.layers = nn.ModuleList(nn.Linear(dims[n], dims[n+1]) for n in range(len(nouts)))

        # add activations list to self
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        # flatten MNIST images to flat vectors
        x = x.view(x.size(0), -1)

        # apply activation functions to each layer of the MLP
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x)) # eg F.relu(self.layers(x))

        return x

def training_routine(model, device, training_loader, optimization_function, cycle, logging_interval = 100):
    model.train() # set model into training mode
    loss_func = nn.CrossEntropyLoss() # cross entropy loss function
    loss_total = 0.0

    for batch_num, (data, target) in enumerate(training_loader, start=1):
        # move data to GPU or CPU device
        data = data.to(device)
        target = target.to(device)

        optimization_function.zero_grad()
        output = model(data) # run batch through the MLP nn
        loss = loss_func(output, target) # compute loss comparing model output and true value
        loss.backward() # back propagation step
        optimization_function.step() # update model weights based on back propagation step

        loss_total += loss.item() # keep running total of losses at each step
        if batch_num % logging_interval == 0:
            loss_mean = loss_total / batch_num

            print(
                f"Cycle #: {cycle:03d}"
                f"[{batch_num * len(data)}/{len(training_loader.dataset)} " # how many images out of total training set size have been processed
                f"({100. * batch_num / len(training_loader):.0f}%)]\t" # what percentage of training data has been processed
                f"Loss: {loss_mean}" # average loss for this cycle
            )
            loss_total = 0.0 # set loss total back to 0 to compute average for next cycle

def evaluate(model, device, test_loader):
    model.eval() # set model to testing mode
    loss_func = nn.CrossEntropyLoss(reduction="sum")  # sum losses for computing average
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # no gradient computation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)             # [batch_size, 10]
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # predicted class index
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            print(f"output tensor: {output[0]}\ntarget: {target[0]}\t prediction: {pred[0].item()}")
            break






def main():
    # number of training cycles
    training_cycles = 30
    # set learning rate
    learning_rate = 0.005

    # set number of neurons at each layer of MLP
    nouts = [2048, 1024, 10]

    # set CPU as device
    device = torch.device("cpu")

    # set activation functions for MLP layers
    activations = [
        nn.LeakyReLU(),
        nn.LeakyReLU(),
        nn.Identity()
    ]

    # load data
    training_loader, test_loader = mnist_loader(training_cycles=training_cycles)

    # create model with activation functions and number of neurons at each layer
    model = MLP(activations=activations, nouts=nouts).to(device)

    # set optimization function used for gradient descent
    optimization_function = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.85, nesterov=True)

    # Training and evaluation loop
    for cycle in range(1, training_cycles + 1):
        training_routine(model, device, training_loader, optimization_function, cycle)
        evaluate(model, device, test_loader)

    # for k, v in model.named_parameters():
    #     print(k, v)
    # torch.save(model.state_dict(), "mnist_mlp.pth")




if __name__ == "__main__":
    main()