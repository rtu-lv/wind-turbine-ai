#!/usr/bin/env python3
# set the matplotlib backend so figures can be saved in the background
import argparse
import os
import pickle
import sys
from os.path import exists, join

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torchmetrics import MeanMetric
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import multiprocessing

from model_lenet import LeNet

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.alya_dataset import AlyaDataset

matplotlib.use("agg")

# Used to force same split each run between validation and training
torch.manual_seed(42)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
                help="path to output loss/accuracy plot")
ap.add_argument("-d", "--data", type=str, required=True,
                help="path to binary stored data class")
ap.add_argument("-c", "--continue", type=float, required=False,
                help="use to continue training with a given learning rate")
ap.add_argument("-e", "--epochs", type=int, required=True,
                help="Number of epochs to train")
args = vars(ap.parse_args())

# define training hyper-parameters
if args["continue"] is not None:
    INIT_LR = args["continue"]
else:
    INIT_LR = 1e-3
EPOCHS = args["epochs"]

# define the train and val splits
TRAIN_SPLIT = 0.8

DATA_BASE_PATH = '../data/'


class ModuleSurrogate(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        print("[INFO] initializing the model...")
        if args["continue"] is not None:
            self.model = torch.load(args["model"])
        else:
            self.model = LeNet(num_channels=2)

        # --------------------------------------------------------------------------------------
        # load the Alya surrogate dataset
        print("[INFO] loading the Alya Surrogate dataset...")
        data_cache_file = join(current_dir, args["data"])
        if exists(data_cache_file):
            with open(data_cache_file, 'rb') as f:
                self.train_dataset = pickle.load(f)
                self.test_dataset = pickle.load(f)
        else:
            self.train_dataset = AlyaDataset(DATA_BASE_PATH + "surrogate_train")
            self.test_dataset = AlyaDataset(DATA_BASE_PATH + "surrogate_test")
            with open(data_cache_file, 'wb') as f:
                pickle.dump(self.train_dataset, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.test_dataset, f, pickle.HIGHEST_PROTOCOL)

        #self.train_dataset.transform_porosity()
        #self.test_dataset.transform_porosity()

        #self.train_dataset.plot_data()

        # calculate the train/validation split
        print("[INFO] generating the train/validation split...")
        num_train_samples = int(len(self.train_dataset) * TRAIN_SPLIT)
        num_val_samples = len(self.train_dataset) - num_train_samples
        # %%
        (self.train_dataset, self.val_dataset) = random_split(self.train_dataset, [num_train_samples, num_val_samples])

        self.train_loss = nn.MSELoss()
        self.val_loss = nn.MSELoss()

        self.train_accuracy = MeanMetric()
        self.val_accuracy = MeanMetric()
        self.test_accuracy = MeanMetric()

    def train_dataloader(self):
        number_of_cores = multiprocessing.cpu_count() // 4
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=number_of_cores)

    def val_dataloader(self):
        number_of_cores = multiprocessing.cpu_count() // 4
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=number_of_cores)

    def test_dataloader(self):
        number_of_cores = multiprocessing.cpu_count()
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=number_of_cores)

    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # perform a forward pass and calculate the training loss
        pred = self.model(x1, x2)

        loss = self.train_loss(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_accuracy(pred / y)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # make the predictions and calculate the validation loss
        pred = self.model(x1, x2)

        loss = self.val_loss(pred, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        self.val_accuracy(pred / y)
        self.log("validation_accuracy", self.val_accuracy, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        pred = self.model(x1, x2)

        loss = self.train_loss(pred, y)
        self.log("test_loss", loss)

        self.test_accuracy(pred / y)
        self.log("test_accuracy", self.test_accuracy)

        return loss

    def configure_optimizers(self):
        # initialize our optimizer
        opt = AdamW(self.model.parameters(), lr=INIT_LR)

        sch1 = lr_scheduler.CosineAnnealingLR(opt, 500)
        sch2 = lr_scheduler.LambdaLR(opt, lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))

        return [opt], [sch1, sch2]


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DEF_BATCH_SIZE = 50

model = ModuleSurrogate(batch_size=DEF_BATCH_SIZE)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=EPOCHS, auto_scale_batch_size="binsearch",
                     log_every_n_steps=DEF_BATCH_SIZE)
#trainer.tune(model=model)
print("Suggested batch size:{}".format(model.batch_size))

trainer.fit(model=model)
trainer.test(ckpt_path='best')

# serialize the model to disk
#torch.save(model.model, args["model"])