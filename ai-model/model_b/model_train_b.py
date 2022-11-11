# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from model_cnn import aLNetB
from sklearn.metrics import mean_absolute_error
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import pickle
from os.path import exists

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from common.alya_dataset import AlyaDataset

# Used to force same split each run between validation and training
torch.manual_seed(42)

# %% Setting Training parameters and DS initialization

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

# define training hyperparameters
if args["continue"] is not None:
    INIT_LR = args["continue"]
else:
    INIT_LR = 1e-3
BATCH_SIZE = 10
EPOCHS = args["epochs"]

# define the train and val splits
TRAIN_SPLIT = 0.8

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the Alya surrogate dataset
print("[INFO] loading the Alya Surrogate dataset...")
if (exists(args["data"])):
    with open(args["data"], 'rb') as f:
        trainData = pickle.load(f)
        testData = pickle.load(f)
else:
    trainData = AlyaDataset("../../mat-train", True)
    testData = AlyaDataset("../../mat-test", True)
    with open(args["data"], 'wb') as f:
        pickle.dump(trainData, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testData, f, pickle.HIGHEST_PROTOCOL)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = len(trainData) - numTrainSamples
# %%
(trainData, valData) = random_split(trainData,
                                    [numTrainSamples, numValSamples]
                                    # , generator=torch.Generator().manual_seed(42)   # not compatible with v1.4.0 PyTorch
                                    )

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
                             batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# %% Initialize the Net model

print("[INFO] initializing the aLNet model...")
if args["continue"] is not None:
    model = torch.load(args["model"]).to(device)
else:
    model = aLNetB(numChannels=2).to(device)

# initialize our optimizer and loss function
opt = AdamW(model.parameters(), lr=INIT_LR)
sch1 = lr_scheduler.CosineAnnealingLR(opt, 500, 1e-5)
sch2 = lr_scheduler.LambdaLR(opt, \
                             lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))
lossFn = nn.MSELoss()  # <-- Mean Square error loss function
# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# %% Training part with train (and validation) DS

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss¡
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainPredNorm = 0
    valPredNorm = 0

    ePrint = True
    # loop over the training set
    for (x1, x2, ang, y) in trainDataLoader:
        # send the input to the device
        (x1, ang, y) = (x1.to(device), ang.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x1, ang)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainPredNorm += (pred / y).sum().item()
        if ePrint:
            print("Loss Funct: ", loss)
            print(trainPredNorm)
            ePrint = False

    # Scheduler step
    sch1.step()
    sch2.step()

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x1, x2, ang, y) in valDataLoader:
            # send the input to the device
            (x1, ang, y) = (x1.to(device), ang.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x1, ang)
            totalValLoss += lossFn(pred, y)
            # calculate the number of correct predictions
            valPredNorm += (pred / y).sum().item()

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainPredNorm = trainPredNorm / len(trainDataLoader.dataset)
    valPredNorm = valPredNorm / len(valDataLoader.dataset)

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainPredNorm)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valPredNorm)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainPredNorm))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valPredNorm))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

final_LR = sch2.get_last_lr()

# %% Evaluation on test DS

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

Gdata = {"Train": [trainDataLoader],
         "Validation": [valDataLoader],
         "Test": [testDataLoader]}

# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []

    # loop over the different datasets
    for dataSet in Gdata:
        # loop over the test set
        realPoro = []
        predPoro = []
        for (x1, x2, ang, y) in Gdata[dataSet][0]:
            # grab the ground truth porosity
            for y_ind in y:
                realPoro.append(y_ind.item() * 1e5)
            # send the input to the device
            (x1, ang) = (x1.to(device), ang.to(device))
            # make the predictions and add them to the list
            pred = model(x1, ang)
            if dataSet == "Test": preds.extend(pred.cpu())
            for pred_ind in pred:
                predPoro.append(pred_ind.item() * 1e5)
        Gdata[dataSet].append(realPoro)
        Gdata[dataSet].append(predPoro)

# %%
# generate a classification report
print(f'Test data mean absolute error: '
      f'{mean_absolute_error(testData.targets().cpu().numpy(), np.array(preds))}')

print("Final LR: {}".format(final_LR))

# %%

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