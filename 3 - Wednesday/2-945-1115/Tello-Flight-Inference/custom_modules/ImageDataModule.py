"""
This module is used to create a PyTorch Lightning Data Module
It is used to create a dataset and dataloader for the training, validation, and test datasets
"""

# Import Python-native modules
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import custom modules
from . import config


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_width, image_height, batch_size=64):
        super(ImageDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Transformations to be applied to our test images
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((image_width, image_height)),
                transforms.RandomRotation(degrees=config.RANDOM_ROTATION_DEGREES),
                transforms.RandomHorizontalFlip(
                    p=config.RANDOM_HORIZONTAL_FLIP_PERCENTAGE
                ),
                transforms.RandomAffine(degrees=config.RANDOM_AFFINE_DEGREES),
                transforms.ToTensor(),
            ]
        )

        # Transformations to be applied to our validation/test images
        self.transform_test = transforms.Compose(
            [
                transforms.Resize((image_width, image_height)),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        # This method is used for download or preparing data
        # It is called only once for the entire run
        # torchvision.datasets.ImageFolder() does not require this step
        return

    def setup(self, stage=None):
        # This method is used for splitting the dataset into train, validation, and test datasets
        # It is called every time the trainer is initialized or the data module is re-initialized

        # Create training dataset
        self.train_dataset = datasets.ImageFolder(
            root=self.data_dir + "/Train", transform=self.transform_train
        )
        self._nice_print(f"Length of training dataset: {len(self.train_dataset)}")

        # Create validation dataset
        self.val_dataset = datasets.ImageFolder(
            root=self.data_dir + "/Val", transform=self.transform_test
        )
        self._nice_print(f"Length of validation dataset: {len(self.val_dataset)}")

        # Create test dataset
        self.test_dataset = datasets.ImageFolder(
            root=self.data_dir + "/Test", transform=self.transform_test
        )
        self._nice_print((f"Length of test dataset: {len(self.test_dataset)}"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

    @staticmethod
    def _nice_print(string_in):
        """
        Print a string in a nice format
        """
        border_length = len(string_in) + 4
        top_border = "*" * border_length
        bottom_border = "-" * border_length

        print(top_border)
        print(f"* {string_in} *")
        print(bottom_border)
