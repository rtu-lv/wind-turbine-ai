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
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchmetrics import MeanAbsoluteError

from model_cnn_surrogate import SurrogateCNN

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
            self.model = SurrogateCNN(num_channels=2)

        self.num_workers = 0#multiprocessing.cpu_count()

        self.__load_data()

        self.loss_function = nn.MSELoss()

        self.train_accuracy = MeanAbsoluteError() # TODO: Restore the function created by Albert
        self.val_accuracy = MeanAbsoluteError()
        self.test_accuracy = MeanAbsoluteError()

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_accuracy(pred, y)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        # make the predictions and calculate the validation loss
        pred = self.model(x1, x2)

        loss = self.loss_function(pred, y)
        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        acc = self.val_accuracy(pred, y)
        self.log("validation_accuracy", acc, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, y = batch

        pred = self.model(x1, x2)

        loss = self.loss_function(pred, y)
        self.log("test_loss", loss)

        self.test_accuracy(pred, y)
        self.log("test_accuracy", self.test_accuracy)

        return loss

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr)

        sch1 = lr_scheduler.CosineAnnealingLR(opt, 500)
        sch2 = lr_scheduler.LambdaLR(opt, lr_lambda=(lambda ep: (ep * (1e-2 - 1) + EPOCHS) / EPOCHS))

        return [opt], [sch1, sch2]


# %%
DEF_BATCH_SIZE = 50


def train_surrogate_model(config, num_epochs, num_gpus):
    model = SurrogateModel(config)

    metrics = {"loss": "ptl/val_loss", "accuracy": "ptl/val_accuracy"}
    callbacks = [LearningRateMonitor(logging_interval='step'), TuneReportCallback(metrics, on="validation_end"),
                 TuneReportCheckpointCallback(metrics, filename="checkpoint", on="validation_end")
                ]

    trainer = pl.Trainer(accelerator="gpu" if NUM_GPUS > 0 else "cpu",
                         devices=num_gpus, max_epochs=num_epochs,
                         callbacks=callbacks, enable_progress_bar=True)
    trainer.fit(model=model)


def tune_surrogate_model(num_epochs, num_samples):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }
    trainable = tune.with_parameters(train_surrogate_model, num_epochs=num_epochs, num_gpus=NUM_GPUS)

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    resources_per_trial = {
        "cpu": NUM_CPUS,
        "gpu": NUM_GPUS
    }

    result = tune.run(
        trainable,
        resources_per_trial=resources_per_trial,
        scheduler=scheduler,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tuning_logs",
        local_dir=os.getcwd(),
        resume='AUTO+ERRORED'
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    return best_trial

def load_and_cache_data(data_cache_file):
    print("[INFO] loading and caching Alya data files...")

    train_dataset = AlyaDataset(os.path.join(DATA_BASE_PATH, DATA_TRAIN_SUBDIR))
    test_dataset = AlyaDataset(os.path.join(DATA_BASE_PATH, DATA_TEST_SUBDIR))
    with open(data_cache_file, 'wb') as f:
        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)


def tune_and_test():
    data_cache_file = join(current_dir, args["data"])
    if not exists(data_cache_file):
        load_and_cache_data(data_cache_file)

    best_trial = tune_surrogate_model(EPOCHS, NUM_SAMPLES)
    best_trained_model = SurrogateModel(best_trial.config)
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

    print("--- Testing surrogate model ---")
    trainer = pl.Trainer(accelerator="gpu" if NUM_GPUS > 0 else "cpu", devices=NUM_GPUS)
    trainer.test(model=best_trained_model, ckpt_path=os.path.join(best_checkpoint_dir, "checkpoint"))

    # serialize the model to disk
    # torch.save(model.model, args["model"])
    # torch.onnx.export(best_model.model, best_model.train_dataset.dataset.get_input(), "cnn_surrogate.onnx",
    #                  export_params=True,
    #                  input_names=['upstream', 'downstream'], output_names=['porosity'])


if __name__ == "__main__":
    tune_and_test()


