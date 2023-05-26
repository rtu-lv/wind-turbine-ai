#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import pickle
import sys
from os.path import exists, join

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchmetrics import R2Score

from transformer_model import FourierTransformer2D

TUNING_LOGS_DIR = "tuning_logs"

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.alya_dataset import AlyaDataset

# Used to force same split each run between validation and training
torch.manual_seed(42)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained model")
ap.add_argument("-d", "--data", type=str, required=True,
                help="path to binary stored data class")
ap.add_argument("-c", "--continue", type=float, required=False,
                help="use to continue training with a given learning rate")
ap.add_argument("-e", "--epochs", type=int, required=True,
                help="Number of epochs to train")
ap.add_argument("-t", "--trials", type=int, required=True,
                help="Number of trials for hyperparameter optimization")
ap.add_argument("-db", "--db_path", type=str, required=True,
                help="Path of alya data files")
ap.add_argument("-cpus", "--num_cpus", type=int, required=False,
                help="Number of CPUs to use")
ap.add_argument("-gpus", "--num_gpus", type=int, required=False,
                help="Number of GPUs to use")
args = vars(ap.parse_args())

# define the train and val splits
TRAIN_SPLIT = 0.8

# define training hyper-parameters
EPOCHS = args["epochs"]
NUM_SAMPLES = args["trials"]

DATA_BASE_PATH = args["db_path"]
DATA_TRAIN_SUBDIR = "surr_train"
DATA_TEST_SUBDIR = "surr_test"

if not torch.cuda.is_available():
    print("[WARN] CUDA is not available")

NUM_CPUS = multiprocessing.cpu_count() if (args["num_cpus"] is None) else args["num_cpus"]
NUM_GPUS = torch.cuda.device_count() if (args["num_gpus"] is None or not torch.cuda.is_available()) else args["num_gpus"]

print("Number of CPUs to be used: {}".format(NUM_CPUS))
print("Number of GPUs to be used: {}".format(NUM_GPUS))


class SurrogateModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]

        print("[INFO] initializing the model...")
        if args["continue"] is not None:
            self.model = torch.load(args["model"])
        else:
            self.model = FourierTransformer2D(config)

        self.num_workers = 0#multiprocessing.cpu_count()

        self.__load_data()

        self.loss_function = nn.MSELoss()

        self.train_accuracy = R2Score(num_outputs=4)
        self.val_accuracy = R2Score(num_outputs=4)
        self.test_accuracy = R2Score(num_outputs=4)

    def __load_data(self):
        # load the Alya surrogate dataset
        print("[INFO] loading the Alya Surrogate dataset...")
        data_cache_file = join(current_dir, args["data"])
        if exists(data_cache_file):
            try:
                with open(data_cache_file, 'rb') as f:
                    self.train_dataset = pickle.load(f)
                    self.test_dataset = pickle.load(f)
            except OSError as error:
                print(error)
                with open(data_cache_file, 'rb') as f:
                    self.train_dataset = pickle.load(f)
                    self.test_dataset = pickle.load(f)
        else:
            raise Exception("Alya cached data file not found")

        #self.train_dataset.transform_porosity()
        #self.test_dataset.transform_porosity()
        # self.train_dataset.plot_data()

        # calculate the train/validation split
        print("[INFO] generating the train/validation split...")
        num_train_samples = int(len(self.train_dataset) * TRAIN_SPLIT)
        num_val_samples = len(self.train_dataset) - num_train_samples

        (self.train_dataset, self.val_dataset) = random_split(self.train_dataset, [num_train_samples, num_val_samples])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # perform a forward pass and calculate the training loss
        pred = self.model(x1, x2)

        loss = self.loss_function(pred, y)
        self.log_dict({"summary/train_loss": loss, "step": self.current_epoch + 1})

        self.train_accuracy(pred, y)
        self.log_dict({"summary/train_accuracy": self.train_accuracy, "step": self.current_epoch + 1})

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # make the predictions and calculate the validation loss
        pred = self.model(x1, x2)

        loss = self.loss_function(pred, y)
        self.log_dict({"summary/validation_loss": loss, "step": self.current_epoch + 1})

        acc = self.val_accuracy(pred, y)
        self.log_dict({"summary/validation_accuracy": acc, "step": self.current_epoch + 1})

        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        pred = self.model(x1, x2)

        loss = self.loss_function(pred, y)
        self.log("summary/test_loss", loss)

        self.test_accuracy(pred, y)
        self.log("summary/test_accuracy", self.test_accuracy)

        return loss

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr)

        sch1 = lr_scheduler.CosineAnnealingLR(opt, 500)
        sch2 = lr_scheduler.LambdaLR(opt, lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))

        return [opt], [sch1, sch2]


def train_surrogate_model(model, num_epochs, num_gpus):
    callbacks = [LearningRateMonitor(logging_interval='step')]

    trainer = pl.Trainer(accelerator="gpu" if num_gpus > 0 else "cpu", devices=num_gpus if num_gpus > 0 else 1,
                         max_epochs=num_epochs, callbacks=callbacks, enable_progress_bar=True)
    trainer.fit(model=model)

    return trainer


def load_and_cache_data(data_cache_file):
    print("[INFO] loading and caching Alya data files...")

    train_dataset = AlyaDataset(os.path.join(DATA_BASE_PATH, DATA_TRAIN_SUBDIR))
    test_dataset = AlyaDataset(os.path.join(DATA_BASE_PATH, DATA_TEST_SUBDIR))
    with open(data_cache_file, 'wb') as f:
        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)


def train_and_test():
    data_cache_file = join(current_dir, args["data"])
    if not exists(data_cache_file):
        load_and_cache_data(data_cache_file)

    config = {
        "lr": 1e-4,
        "batch_size": 64
    }

    print("--- Training surrogate model ---")
    model = SurrogateModel(config)

    trainer = train_surrogate_model(model, EPOCHS, NUM_GPUS)

    print("--- Testing surrogate model ---")
    trainer.test(ckpt_path='best')

    # serialize the model to disk
    torch.save(model, args["model"])

    # Export the model in the ONNX format
    # torch.onnx.export(best_model.model, best_model.train_dataset.dataset.get_input(), "cnn_surrogate.onnx",
    #                  export_params=True,
    #                  input_names=['upstream', 'downstream'], output_names=['porosity'])


if __name__ == "__main__":
    train_and_test()


