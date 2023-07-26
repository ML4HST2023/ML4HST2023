"""
This file contains the CNN_lightning class, which is a PyTorch Lightning implementation of the CNN model.
Architecture-specific changes are made in the __init__ function.
"""

# Import Python-native modules
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# Import custom modules
from . import config


class CNN_lightning(pl.LightningModule):
    """
    A lot of this code was generated from referencing the PyTorch Lightning documentation:
    https://lightning.ai/docs/pytorch/stable/
    """

    def __init__(
        self,
        num_dummy_images,
        num_channels,
        image_width,
        image_height,
        verbose=True,
    ):
        super(CNN_lightning, self).__init__()

        # Dummy input to calculate the output shape of each layer
        self.dummy_input = torch.ones(
            num_dummy_images, num_channels, image_width, image_height
        )

        self.architecture = nn.Sequential()

        ###############################
        # Convolution Layer 1
        ###############################
        self._nice_print("Convolution Layer 1", verbose=verbose)

        # Explicit convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.architecture.add_module("conv1", self.conv1)
        output_shape = self._get_layer_output_shape(
            "conv1", self.architecture, print_shape=verbose
        )

        # ReLU activation layer
        self.relu1 = nn.ReLU()
        self.architecture.add_module("relu1", self.relu1)
        output_shape = self._get_layer_output_shape(
            "relu1", self.architecture, print_shape=verbose
        )

        # Max pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module("maxpool1", self.maxpool1)
        output_shape = self._get_layer_output_shape(
            "maxpool1", self.architecture, print_shape=verbose
        )

        # Batch normalization layer
        self.b1 = nn.BatchNorm2d(output_shape[1])
        self.architecture.add_module("b1", self.b1)
        output_shape = self._get_layer_output_shape(
            "b1", self.architecture, print_shape=verbose
        )

        # Dropout layer
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.architecture.add_module("dropout1", self.dropout1)
        output_shape = self._get_layer_output_shape(
            "dropout1", self.architecture, print_shape=verbose
        )

        ###############################
        # Convolution Layer 2
        ###############################
        self._nice_print("Convolution Layer 2", verbose=verbose)

        # Explicit convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=output_shape[1],
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.architecture.add_module("conv2", self.conv2)
        output_shape = self._get_layer_output_shape(
            "conv2", self.architecture, print_shape=verbose
        )

        # ReLU activation layer
        self.relu2 = nn.ReLU()
        self.architecture.add_module("relu2", self.relu2)
        output_shape = self._get_layer_output_shape(
            "relu2", self.architecture, print_shape=verbose
        )

        # Max pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module("maxpool2", self.maxpool2)
        output_shape = self._get_layer_output_shape(
            "maxpool2", self.architecture, print_shape=verbose
        )

        # Batch normalization layer
        self.b2 = nn.BatchNorm2d(output_shape[1])
        self.architecture.add_module("b2", self.b2)
        output_shape = self._get_layer_output_shape(
            "b2", self.architecture, print_shape=verbose
        )

        ###############################
        # Flatten Layer
        ###############################
        self._nice_print("Flatten Layer", verbose=verbose)

        # Flatten layer
        self.flatten = nn.Flatten()
        self.architecture.add_module("flatten", self.flatten)
        output_shape = self._get_layer_output_shape(
            "flatten", self.architecture, print_shape=verbose
        )

        ###############################
        # Output Layer
        ###############################
        self._nice_print("Output Layer", verbose=verbose)

        # Fully connected layer
        self.FULLY_CONNECTED_INPUTS = self._get_layer_output_shape(
            name="fc_calc", model_in=self.architecture, print_shape=False
        )[1]
        self.fc1 = nn.Linear(self.FULLY_CONNECTED_INPUTS, 1)
        self.architecture.add_module("fc1", self.fc1)
        output_shape = self._get_layer_output_shape(
            "fc1", self.architecture, print_shape=verbose
        )

        # Sigmoid activation layer (acting as output)
        self.sigmoid = nn.Sigmoid()
        self.architecture.add_module("sigmoid", self.sigmoid)
        output_shape = self._get_layer_output_shape(
            "sigmoid", self.architecture, print_shape=verbose
        )

    def forward(self, x):
        """
        Forward pass of the model
        """
        x = self.architecture(x)
        return x

    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        opt = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        return opt

    def training_step(self, batch, batch_idx):
        """
        Training step of the model (automatically called by PyTorch Lightning)
        """
        logits, y = self._common_step(batch)
        training_dict = self._calculate_loss_and_accuracy("train", logits, y)
        return training_dict["train_loss"]

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model (automatically called by PyTorch Lightning)
        """
        logits, y = self._common_step(batch)
        validation_dict = self._calculate_loss_and_accuracy("val", logits, y)
        return validation_dict["val_loss"]

    def test_step(self, batch, batch_idx):
        """
        Test step of the model (automatically called by PyTorch Lightning)
        """
        logits, y = self._common_step(batch)
        testing_dict = self._calculate_loss_and_accuracy("test", logits, y)
        return testing_dict["test_loss"]

    def _common_step(self, batch):
        """
        Common step of the model - shared by training, validation and testing
        """
        x, y = batch
        y = y.unsqueeze(1)
        y = y.float()
        logits = self.forward(x)
        return logits, y

    def _calculate_loss_and_accuracy(self, typeName, logits, y):
        """
        Calculate loss and accuracy for a given step
        """
        loss = nn.BCELoss()(logits, y)
        preds = torch.round(logits)
        acc = (preds == y).sum().item() / len(preds)
        self.log(
            f"{typeName}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{typeName}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        output_dict = {f"{typeName}_loss": loss, f"{typeName}_accuracy": acc}

        return output_dict

    def _get_layer_output_shape(self, name, model_in, print_shape=True):
        """
        Get the output shape of a given layer
        """
        if print_shape:
            print(f"Output shape after {name}: {model_in(self.dummy_input).shape}")
        return model_in(self.dummy_input).shape

    @staticmethod
    def _nice_print(string_in, verbose=True):
        """
        Print a string in a nice format
        """
        if verbose:
            border_length = len(string_in) + 4
            top_border = "*" * border_length
            bottom_border = "-" * border_length

            print(top_border)
            print(f"* {string_in} *")
            print(bottom_border)
