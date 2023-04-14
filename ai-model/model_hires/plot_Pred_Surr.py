#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:53:29 2023

@author: al
"""

import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Used to force same split each run between validation and training
torch.manual_seed(42)

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from common.alya_dataset import AlyaDataset

import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained model")
ap.add_argument("-d", "--data", type=str, required=True,
                help="path to binary stored data class")
args = vars(ap.parse_args())


def plot_PredvsReal(Gdata):
    # plot the prediction vs. real Porosity
    plt.style.use("ggplot")
    fig, axs = plt.subplots(4, 3, figsize=(45,60), sharey=True)
    for i in range(3):
        axs[0][i].set_ylim([0, 5e4])
        for j in range(4):
            axs[j][i].set_xlim([0, 5e4])
    for i, dS in enumerate(Gdata):
        for j in range(4):
            axs[j][i].scatter([ arr[j] for arr in Gdata[dS][1] ], 
                              [ arr[j] for arr in Gdata[dS][2] ], 
                              color="blue")
            axs[j][i].plot([0, 5e4], [0, 5e4], linestyle=":", color="green" )
            axs[j][i].set_xlabel("Real porosity")
            axs[j][i].set_title(dS)
    for j in range(4):
        axs[j][0].set_ylabel("Predicted porosity")
    fig.suptitle("Predicted versus real porosity")
    pname = 'PredvsSurrogate.png'
    plt.savefig(pname, bbox_inches='tight')


def plot_PoroCorrelations(Gdata):
    # plot the prediction vs. real Porosity
    plt.style.use("ggplot")
    fig, axs = plt.subplots(3, 3, figsize=(45,60), sharey=True)
    for i in range(3):
        axs[0][i].set_ylim([0, 5e4])
        axs[0][i].set_xlim([0, 5e4])
        axs[1][i].set_ylim([-5e4, 5e4])
        axs[1][i].set_xlim([0, 5e4])
        axs[2][i].set_ylim([-5e4, 5e4])
        axs[2][i].set_xlim([-5e4, 5e4])
    for i, dS in enumerate(Gdata):
        axs[0][i].scatter([ arr[0] for arr in Gdata[dS][1] ], 
                          [ arr[3] for arr in Gdata[dS][1] ], 
                          color="blue")
        axs[1][i].scatter([ arr[0] for arr in Gdata[dS][1] ], 
                          [ arr[1] for arr in Gdata[dS][1] ], 
                          color="blue")
        axs[2][i].scatter([ arr[1] for arr in Gdata[dS][1] ], 
                          [ arr[2] for arr in Gdata[dS][1] ], 
                          color="blue")        
        axs[0][i].plot([0, 5e4], [0, 5e4], linestyle=":", color="green" )
        for j in range(3):
            axs[j][i].set_title(dS)
    axs[0][0].set_ylabel("P4")
    [axs[0][i].set_xlabel("P1") for i in range(3)]
    axs[1][0].set_ylabel("P3")
    [axs[1][i].set_xlabel("P1") for i in range(3)]
    axs[2][0].set_ylabel("P3")
    [axs[2][i].set_xlabel("P2") for i in range(3)]
    fig.suptitle("Porosity tensor correlations")
    pname = 'PoroCorrelations.png'
    plt.savefig(pname, bbox_inches='tight')

    
    
def getPlotData(model):
    Gdata = { "Train" :     [trainDataLoader], 
              "Validation" :[valDataLoader],
              "Test" :      [testDataLoader] }

    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # loop over the different datasets
        for dataSet in Gdata:
            realPoro = []
            predPoro = []
            for (x1, x2, x3, x4, y) in Gdata[dataSet][0]:
                # grab the ground truth porosity
                for y_ind in y:
                    realPoro.append(y_ind.cpu().numpy() * 5e4)
                	# send the input to the device
                x1 = x1.to(device)
            		# make the predictions and add them to the list
                pred = model(x1)
                for pred_ind in pred:
                    predPoro.append(pred_ind.cpu().numpy() * 5e4)  # pred_ind.item() ???
            Gdata[dataSet].append(realPoro)
            Gdata[dataSet].append(predPoro)
            
    return Gdata


if __name__ == "__main__":
    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args["model"], map_location=device).to(device)

    data_cache_file = args["data"]
    with open(data_cache_file, 'rb') as f:
        train_dataset = pickle.load(f)
        test_dataset = pickle.load(f)

    num_train_samples = int(len(train_dataset) * 0.8)
    num_val_samples = len(train_dataset) - num_train_samples

    (train_dataset, val_dataset) = random_split(train_dataset, [num_train_samples, num_val_samples])

    trainDataLoader = DataLoader(train_dataset)
    valDataLoader   = DataLoader(val_dataset)
    testDataLoader  = DataLoader(test_dataset)
    
    Gdata = getPlotData(model)
    
    plot_PredvsReal(Gdata)
    plot_PoroCorrelations(Gdata)