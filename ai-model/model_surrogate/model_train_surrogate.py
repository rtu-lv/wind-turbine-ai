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
import torchmetrics
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model_lenet import LeNet

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.alya_dataset import AlyaDataset

from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter()


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

        # calculate the train/validation split
        print("[INFO] generating the train/validation split...")
        num_train_samples = int(len(self.train_dataset) * TRAIN_SPLIT)
        num_val_samples = len(self.train_dataset) - num_train_samples
        # %%
        (self.train_dataset, self.val_dataset) = random_split(self.train_dataset, [num_train_samples, num_val_samples])

        self.lossFn = nn.MSELoss()  # <-- Mean Square error loss function

        self.train_accuracy = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.MeanMetric()
        self.test_accuracy = torchmetrics.MeanMetric()

        # initialize a dictionary to store training history
        self.H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)

    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # perform a forward pass and calculate the training loss
        pred = self.model(x1, x2)

        loss = (self.lossFn(pred[0], y[0]) + self.lossFn(pred[1], y[1]) + self.lossFn(pred[2], y[2]) + self.lossFn(pred[3], y[3])) / 4
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_accuracy((pred / y).sum().item() / self.batch_size)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True)

        #writer.add_scalar("Loss/train", loss, self.train_accuracy.)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # make the predictions and calculate the validation loss
        pred = self.model(x1, x2)

        loss = self.lossFn(pred, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        self.val_accuracy((pred / y).sum().item() / self.batch_size)
        self.log("validation_accuracy", self.val_accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        pred = self.model(x1, x2)

        loss = self.lossFn(pred, y)
        self.log("test_loss", loss)

        self.test_accuracy((pred / y).sum().item() / self.batch_size)
        self.log("test_accuracy", self.test_accuracy)

    def configure_optimizers(self):
        # initialize our optimizer and loss function
        opt = AdamW(self.model.parameters(), lr=INIT_LR)
        # opt = ASGD(model.parameters(), lr=INIT_LR, t0=500, weight_decay=0.01)

        sch1 = lr_scheduler.CosineAnnealingLR(opt, 500)
        sch2 = lr_scheduler.LambdaLR(opt, lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))

        return [opt], [sch1, sch2]


class ModelCallback(Callback):
    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.callback_metrics['train_loss'].cpu().detach().numpy()
        module.H["train_loss"].append(train_loss)

        train_accuracy = trainer.callback_metrics['train_accuracy'].cpu().detach().numpy()
        module.H["train_acc"].append(train_accuracy)

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.callback_metrics['validation_loss'].cpu().detach().numpy()
        module.H["val_loss"].append(val_loss)

        val_acc = trainer.callback_metrics['validation_accuracy'].cpu().detach().numpy()
        module.H["val_acc"].append(val_acc)


class CombinedLoss(nn.Module):
    def __init__(self, loss_function):
        super(CombinedLoss, self).__init__()
        self.lossFn = loss_function

    def forward(self, inputs, targets):
        size = inputs.size(dim=1)
        loss = 0
        for i in range(0..length):
            loss += self.lossFn(inputs[i], targets[i])
        return loss / size


def plot(model):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(45, 15))
    ax2 = ax1.twinx()
    ax1.set_ylim([0, .1])
    ax2.set_ylim([0, 2])
    ax1.plot(model.H["train_loss"], label="train_loss", color="red")
    ax1.plot(model.H["val_loss"], label="val_loss", color="orange")
    ax2.plot(model.H["train_acc"], label="train_acc", color="blue")
    ax2.plot(model.H["val_acc"], label="val_acc", color="green")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.savefig(args["plot"], bbox_inches='tight')

    # %%

    # plot the prediction vs. real Porosity
#    plt.style.use("ggplot")
#    fig, axs = plt.subplots(1, 3, figsize=(45, 15), sharey=True)
#    axs[0].set_ylim([0, 1e5])
#    for i in range(3):
#        axs[i].set_xlim([0, 1e5])
#    for i, dS in enumerate(Gdata):
#        axs[i].scatter(Gdata[dS][1], Gdata[dS][2], color="blue")
#        axs[i].plot([0, 1e5], [0, 1e5], linestyle=":", color="green")
#        axs[i].set_xlabel("Real porosity")
#        axs[i].set_title(dS)
#    axs[0].set_ylabel("Predicted porosity")
#    fig.suptitle("Predicted versus real porosity")
#    plt.savefig(args["plot"] + "_comp", bbox_inches='tight')


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DEF_BATCH_SIZE = 50

model = ModuleSurrogate(batch_size=DEF_BATCH_SIZE)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=EPOCHS, auto_scale_batch_size="binsearch",
                     callbacks=[ModelCallback()])
#trainer.tune(model=model)
print("Batch size: {}".format(model.batch_size))

trainer.fit(model=model)
trainer.test(ckpt_path='best')
writer.flush()

plot(model)

# serialize the model to disk
torch.save(model.model, args["model"])