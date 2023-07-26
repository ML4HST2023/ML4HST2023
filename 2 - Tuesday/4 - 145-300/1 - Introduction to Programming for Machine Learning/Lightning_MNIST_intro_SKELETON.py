"""
Python script for training a simple MNIST classifier using PyTorch Lightning.
"""

# Import python modules
import os
import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import warnings

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

# Disable PossibleUserWarning for the "num_workers" argument in the DataLoader
warnings.filterwarnings("ignore", message="The dataloader, train_dataloader, does not have many workers.*")
warnings.filterwarnings("ignore", message="The dataloader, test_dataloader, does not have many workers.*")


class MNISTModel(L.LightningModule):
    def __init__(self):
        """
        The __init__() method defines the network architecture, loss function, and metrics that will be used in the
        training and testing steps.
        """
        super().__init__()

        ###################################################################
        # Initialize the network architecture


        ###################################################################

        ###################################################################
        # Initialize the accuracy metric for training and testing
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
        ###################################################################

    def forward(self, x):
        """
        The forward() method defines the forward pass of the network.
        """
        h = x.view(-1, 28 * 28) # Reshape the input to a vector
        h = self.l1(h)          # Apply the first linear layer
        h = F.relu(h)           # Apply the ReLU activation function
        h = self.l2(h)          # Apply the second linear layer
        return h                # Return the output

    def training_step(self, batch, batch_nb):
        """
        The training_step() method defines the training loop. It takes a batch of data as input and returns the loss

        This hook is called every time a batch is fed to the training loop. It is used to compute the loss on the
        training data and log it to the progress bar and logger.

        For our training_step(), loss must be returned at the end of the method.
        """

        ###################################################################



        ###################################################################

        # Log the loss and accuracy and display them in the progress bar
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        """
        The test_step() method defines the testing loop. It takes a batch of data as input and returns the loss

        This hook is called every time a batch is fed to the testing loop. It is used to compute the loss on the
        testing data and log it to the progress bar and logger.
        """

        ###################################################################



        ###################################################################

        # Log the loss and accuracy and display them in the progress bar
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """
        The configure_optimizers() method defines the optimizer to be used in the training loop.

        This hook is called once at the beginning of training.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":

    ########################################
    # Initialize the model

    ########################################

    # Init a training DataLoader from the MNIST Dataset
    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Initialize a trainer
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=10)

    ########################################
    # Train the model

    ########################################

    # Init a test DataLoader from the MNIST Dataset
    test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    ########################################
    # Test the model

    ########################################
