import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim
import scipy.io
from torch import nn
from typing import Callable
from torch.utils.data import Dataset, DataLoader


# Get appropriate directories
master_dir = os.getcwd()


def generate_sine_data(step: float = 0.05, noise: bool = False, std: float = 0.2):
    """
    Generates training/test data for the second exercise where we try to approximate y=sin(x**2).
    :param step: The step size i.e. spacing between the different points.
    :param noise: Boolean indicating whether we train on noisy data.
    :param std: Standard deviation of the noisy data.
    :return: Tuple containing x values and corresponding function values.
    """
    # Generate x values
    x = np.arange(0, 3*np.pi, step=step)
    # Generate y values
    y = np.sin(x**2)
    # Add noise if so desired
    if noise:
        y += np.random.normal(loc=0, scale=std, size=y.shape)

    return x, y


class Trainer:

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_data: tuple[np.array, np.array],
                 test_data: tuple[np.array, np.array], loss_fn: Callable = nn.MSELoss()):

        # Save parameters as fields
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Save data, convert from Numpy array to Torch tensor
        self.train_x = torch.from_numpy(train_data[0]).float()
        self.train_y = torch.from_numpy(train_data[1]).float()
        self.test_x  = torch.from_numpy(test_data[0]).float()
        self.test_y  = torch.from_numpy(test_data[1]).float()

        # Initialize empty lists for training losses
        self.training_losses = []
        self.test_losses = []

    def train(self, nb_of_epochs: int = 100, verbose=True):

        for i in range(nb_of_epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Compute the output of the network
            train_outputs = self.model(self.train_x)
            test_outputs = self.model(self.test_x)

            # Compute the loss
            train_loss = self.loss_fn(train_outputs, self.train_y)
            test_loss = self.loss_fn(test_outputs, self.train_y)

            # Compute the gradients
            train_loss.backward()

            # Update the weights
            self.optimizer.step()

            if verbose:
                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, nb_of_epochs, train_loss.item()))

