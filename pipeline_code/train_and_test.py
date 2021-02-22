import numpy as np
import random

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms

import tqdm
from Network import Network

import wandb  # wandb is used to monitor the network during training and evaluation

wandb.login()

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Download the dataset
def get_data(train_bool=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    dataset = datasets.MNIST(
        root="data/", download=True, train=train_bool, transform=transform
    )
    return dataset


# Make the loader
def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# Initialise the model and it's parameters
def make(config):

    # Make the data
    train = get_data(train_bool=True)
    test = get_data(train_bool=False)

    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = Network(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # combine logsoftmax and nlloss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer


def train_batch(images, labels, model, optimizer, criterion):
    images = images.to(device)
    labels = labels.to(device)

    # Forward propagation
    outputs = model(images)

    # Calculate softmax and cross entropy loss
    loss = criterion(outputs, labels)

    # Clear gradient
    optimizer.zero_grad()
    # Calculating gradient
    loss.backward()
    # Update parameters
    optimizer.step()

    # Compute accuracy for the batch
    _, predicted = torch.max(outputs.detach(), 1)
    nb_correct = (predicted == labels).sum().item()

    return nb_correct, loss


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    progressB = tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), position=0, leave=True
    )
    with torch.no_grad():  # All the operations whill have no gradient
        for _, (images, labels) in progressB:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * (correct / total)
        wandb.log({"Test Accuracy": accuracy})

        print(f"Accuracy on test set: {accuracy:.2f}")

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

    # Save the model in the pytorch format
    torch.save(model.state_dict(), "state_dict_model.pt")
    wandb.save("state_dict_model.pt")


def train(model, train_loader, test_loader, criterion, optimizer, config):

    # Tell wandb to watch the model parameters (gradients,  weights, ...)
    wandb.watch(model, criterion, log="all", log_freq=10)  # log_freq in number of steps

    nb_epochs = config.epochs

    for epoch in range(1, nb_epochs + 1):
        model.train()

        progressB = tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), position=0, leave=True
        )

        totalCorrect = 0
        total = 0

        for _, (images, labels) in progressB:

            batchCorrect, loss = train_batch(
                images, labels, model, optimizer, criterion
            )

            totalCorrect += batchCorrect
            total += len(labels)

            accuracy = 100 * (totalCorrect / total)

            wandb.log({"Train Loss": loss, "Train Accuracy": accuracy, "Epoch": epoch})
            progressB.set_description(
                f"loss: {loss.item():.2f}, accuracy: {accuracy:.2f},epoch: {epoch}/{nb_epochs}"
            )

        # We test the modele after each epoch here
        if epoch % 1 == 0:
            test(model, test_loader)


# The whole pipeline
def model_pipeline(hyperparameters):

    with wandb.init(project="pytorch-pipeline", config=hyperparameters):

        # We access hyperparameters through wandb so logging matches execution
        config = wandb.config

        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, test_loader, criterion, optimizer, config)
        # Also evaluate the model at a given frequency

    return model


if __name__ == "__main__":

    config = dict(
        epochs=10,
        classes=10,
        kernels=[16, 32, 64],
        batch_size=128,
        learning_rate=0.001,
        dataset="MNIST",
        architecture="CNN",
    )

    model = model_pipeline(config)
