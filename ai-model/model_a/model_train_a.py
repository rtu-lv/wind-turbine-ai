#!/usr/bin/env python3
# set the matplotlib backend so figures can be saved in the background
import matplotlib

# import the necessary packages
from model_lenet import LeNet
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
import pickle
import torchmetrics
from os.path import exists, join
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from common.alya_dataset import AlyaDataset

matplotlib.use("Agg")

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
BATCH_SIZE = 50
EPOCHS = args["epochs"]

# define the train and val splits
TRAIN_SPLIT = 0.8


class ModelA(LightningModule):
    def __init__(self):
        super().__init__()
        print("[INFO] initializing the model...")
        if args["continue"] is not None:
            self.model = torch.load(args["model"])
        else:
            self.model = LeNet(numChannels=2)

        self.lossFn = nn.MSELoss()  # <-- Mean Square error loss function

        self.train_accuracy = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.MeanMetric()

    def training_step(self, batch, batch_idx):
        x1, x2, ang, y = batch

        # perform a forward pass and calculate the training loss
        pred = self.model(x1, x2, ang)

        loss = self.lossFn(pred, y)
        self.log("train_loss", loss)

        self.train_accuracy((pred / y).sum().item())
        self.log("train_accuracy", self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, ang, y = batch

        # make the predictions and calculate the validation loss
        pred = self.model(x1, x2, ang)

        loss = self.lossFn(pred, y)
        self.log("validation_loss", loss)

        self.val_accuracy((pred / y).sum().item())
        self.log("validation_accuracy", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x1, x2, ang, y = batch

        (x1, x2, ang) = (x1.to(device), x2.to(device), ang.to(device))
        pred = self.model(x1, x2, ang)

    def configure_optimizers(self):
        # initialize our optimizer and loss function
        opt = AdamW(self.model.parameters(), lr=INIT_LR)
        # opt = ASGD(model.parameters(), lr=INIT_LR, t0=500, weight_decay=0.01)

        sch1 = lr_scheduler.CosineAnnealingLR(opt, 500)
        sch2 = lr_scheduler.LambdaLR(opt, lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))

        return [opt], [sch1, sch2]


def plot(model, H, Gdata):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(45, 15))
    ax2 = ax1.twinx()
    ax1.set_ylim([0, .1])
    ax2.set_ylim([0, 2])
    ax1.plot(H["train_loss"], label="train_loss", color="red")
    ax1.plot(H["val_loss"], label="val_loss", color="orange")
    ax2.plot(H["train_acc"], label="train_acc", color="blue")
    ax2.plot(H["val_acc"], label="val_acc", color="green")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.savefig(args["plot"], bbox_inches='tight')

    # serialize the model to disk
    torch.save(model, args["model"])

    # %%

    # plot the prediction vs. real Porosity
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 3, figsize=(45, 15), sharey=True)
    axs[0].set_ylim([0, 1e5])
    for i in range(3):
        axs[i].set_xlim([0, 1e5])
    for i, dS in enumerate(Gdata):
        axs[i].scatter(Gdata[dS][1], Gdata[dS][2], color="blue")
        axs[i].plot([0, 1e5], [0, 1e5], linestyle=":", color="green")
        axs[i].set_xlabel("Real porosity")
        axs[i].set_title(dS)
    axs[0].set_ylabel("Predicted porosity")
    fig.suptitle("Predicted versus real porosity")
    plt.savefig(args["plot"] + "_comp", bbox_inches='tight')

#--------------------------------------------------------------------------------------
# load the Alya surrogate dataset
print("[INFO] loading the Alya Surrogate dataset...")
data_dir = join(currentdir, args["data"])
if exists(data_dir):
    with open(data_dir, 'rb') as f:
        trainData = pickle.load(f)
        testData = pickle.load(f)
else:
    trainData = AlyaDataset("../../surr-train")
    testData = AlyaDataset("../../surr-test")
    with open(data_dir, 'wb') as f:
        pickle.dump(trainData, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testData, f, pickle.HIGHEST_PROTOCOL)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = len(trainData) - numTrainSamples
# %%
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples])

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ModelA()
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, log_every_n_steps=50)
trainer.fit(model=model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
trainer.test(ckpt_path='best', dataloaders=testDataLoader)
