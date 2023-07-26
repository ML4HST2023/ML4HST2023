"""
This script trains a CNN model using PyTorch Lightning for our Drone Obstacle Dataset.

The model architecture is defined in CNN_lightning.py, the data module defined in ImageDataModule.py, and the
parameters are defined in config.py.

Note that this script is HEAVILY reliant on parameters found in config.py
"""

# Import Python-native modules
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import warnings

# Import custom modules
from custom_modules.ImageDataModule import ImageDataModule
from custom_modules.CNN_lightning import CNN_lightning
from custom_modules.supplemental_functions import resize_image_dimensions
from custom_modules import config

# Disable PossibleUserWarning for the "num_workers" argument in the DataLoader
warnings.filterwarnings("ignore", message="The dataloader, .*")
warnings.filterwarnings(
    "ignore", message="Checkpoint directory .* exists and is not empty."
)

if __name__ == "__main__":
    # Kickoff the timing of the training run
    start_time = datetime.now()
    print(f"---- Training run started at: {start_time}")
    print(
        f"----> Training using device: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    # Set image dimensions
    image_width, image_height = resize_image_dimensions(
        image_width=config.IMAGE_WIDTH,
        image_height=config.IMAGE_HEIGHT,
        size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
    )

    # Create an instance of our data module
    dm = ImageDataModule(
        data_dir=config.DATA_DIR,
        image_width=image_width,
        image_height=image_height,
        batch_size=config.BATCH_SIZE,
    )

    # Create an instance of our model
    model = CNN_lightning(
        num_dummy_images=config.NUM_DUMMY_IMAGES,
        num_channels=config.NUM_CHANNELS,
        image_width=image_width,
        image_height=image_height,
    )

    # Define the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # parameter to monitor
        mode="min",  # direction to monitor (ex: 'min' for loss, 'max' for acc)
        patience=config.EARLY_STOPPING_PATIENCE,  # number of epochs to wait before stopping
        verbose=True,  # log information to the terminal
        min_delta=config.MIN_DELTA,  # minimum change in monitored value to qualify as improvement
    )

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Create an instance of our trainer, and train the model
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        min_epochs=config.MIN_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
    )
    print(f">>>> Training model for a max {config.MAX_EPOCHS} epochs")
    trainer.fit(model, datamodule=dm)

    # End the timing of the training run
    end_time = datetime.now()
    print(f"---- Training run ended at: {end_time}")
    print(f"---- Training run duration: {end_time - start_time}")

    if config.TEST_AND_SAVE_MODEL_PT:
        # Test the model

        # Kickoff the timing of the testing run
        start_time = datetime.now()
        print(
            f"\n\n---- Testing run for model '{config.TORCH_MODEL_FILENAME_EXT}' started at: {start_time}"
        )
        test_results = trainer.test(model, datamodule=dm)

        # End the timing of the testing run
        end_time = datetime.now()
        print(
            f"---- Testing run for model '{config.TORCH_MODEL_FILENAME_EXT}' ended at: {end_time}"
        )
        print(
            f"---- Testing run duration for model '{config.TORCH_MODEL_FILENAME_EXT}': {end_time - start_time}"
        )

        ###########################
        # Save model as .pt file
        ###########################

        # Check to make sure that config.TORCH_MODEL_DIRECTORY exists, otherwise create it
        if os.path.exists(config.TORCH_MODEL_DIRECTORY) is False:
            print(f">>>> Creating directory: {config.TORCH_MODEL_DIRECTORY}")
            os.makedirs(config.TORCH_MODEL_DIRECTORY)
        else:
            print(f">>>> Model directory: {config.TORCH_MODEL_DIRECTORY}")

        # Get the test accuracy (to be used in the filename)
        test_acc = test_results[0]["test_acc"]

        # Save the model
        torch_model_filename = (
            config.TORCH_MODEL_DIRECTORY
            + config.TORCH_MODEL_FILENAME
            + f"_acc_{test_acc:.4f}"
            + config.TORCH_MODEL_FILENAME_EXT
        )
        print(f">>>> Saving model to {torch_model_filename}")
        torch.save(model.state_dict(), torch_model_filename)

    if config.TEST_BEST_MODEL_CKPT:
        # Test the model ckpt
        checkpoint_path = checkpoint_callback.best_model_path

        # Kickoff the timing of the testing run
        start_time = datetime.now()
        print(
            f"\n\n---- Testing run for model '{checkpoint_path}' started at: {start_time}"
        )
        test_results = trainer.test(model, ckpt_path=checkpoint_path, datamodule=dm)

        # End the timing of the testing run
        end_time = datetime.now()
        print(f"---- Testing run for model '{checkpoint_path}' ended at: {end_time}")
        print(
            f"---- Testing run duration for model '{checkpoint_path}': {end_time - start_time}"
        )
