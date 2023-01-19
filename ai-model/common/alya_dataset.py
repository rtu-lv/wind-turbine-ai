import os
import re
import sys
from os import listdir
from os.path import isfile, join
import time

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# %% Read dataset from Alya "results" folder and store it in Torch Tensor format

# get vector fields (3 channels: vx, vy, p) from an alya results file
def get_fields(file):
    dataUS = []  # Init Upstream data list
    dataDS = []  # Init Downstream data list

    # get Upstream and downstream data from file
    UPS_flag = False
    DWS_flag = False

    with open(file, 'r') as f:  # Open file with field data
        for line in f:
            if line.split()[0] == "UPSTREAM":  # Search for Upstream data start
                UPS_flag = True  # Flag start of Upstream data
            elif UPS_flag:  # Read Upstream data
                # Start of Downstream data
                if line.split()[0] == "DOWNSTREAM":
                    UPS_flag = False
                    DWS_flag = True
                else:
                    # Append [Xcoord, Ycoord, Vx, Vy, P] from each line to dataUS
                    dataUS.append([float(d) for d in line.split()])
            elif DWS_flag:
                # Append [Xcoord, Ycoord, Vx, Vy, P] from each line to dataUS
                dataDS.append([float(d) for d in line.split()])

    # Convert to numpy arrays
    np_dataUS = np.array(dataUS)
    np_dataDS = np.array(dataDS)

    for i, d_array in enumerate([np_dataUS, np_dataDS]):
        # Get list of coordinates
        x_list = sorted([*set(d_array[:, [0]].flatten())])  # get a list of x coords set
        y_list = sorted([*set(d_array[:, [1]].flatten())])  # and also from y

        # dimensions of the field
        n_x, n_y = (len(x_list), len(y_list))

        # 2 channels field initialization: vx, vy
        # and 1 channel for press
        v_field = np.zeros((2, n_x, n_y))
        p_field = np.zeros((n_x, n_y))

        for d in d_array:
            x, y = (d[0], d[1])  # coordinates
            vx, vy = (d[2], d[3])  # velocity components
            p = d[4]  # pressure
            idx_x = x_list.index(x)  # x and y coordinates indices
            idx_y = y_list.index(y)
            # Velocity field component X
            v_field[0, idx_x, idx_y] = vx
            # Velocity field component Y
            v_field[1, idx_x, idx_y] = vy
            # Pressure field values
            p_field[idx_x, idx_y] = p
            if (i == 0):
                v_UPS = v_field[:]
                p_UPS = p_field[:]
            else:
                v_DWS = v_field[:]
                p_DWS = p_field[:]

    return v_UPS, v_DWS, p_UPS, p_DWS


# Parser for getting simulation variables from file
# v_dict -> {"vIn" : [float], "ang" : [float], "por" : [float, float, float, float]}
def parse_run_variables(file):
    with open(file, 'r') as f:  # Open file with field data
        lines = f.readlines()  # read all the lines

    v_dict = {'vIn': None,  # Store values in dictionary
              'ang': None,
              'por': None}

    count = 0  # counter to stop parsing lines
    for l in lines:
        l_split = l.split(':')  # separate leading word
        if ("ANGLE" in l):
            v_dict["ang"] = float(l_split[1])
            count = count + 1
        elif ("VELOCITY" in l):
            v_dict["vIn"] = float(l_split[1])
            count = count + 1
        elif ("POROSITY" in l):
            # strip trailing spaces and split with '[', ',' and ']'
            porMat = re.split('[\[,\]]', l_split[1].strip())
            # remove empty values resulting from regexp and return list
            porMat = list(filter(None, porMat))
            # store as list of floats in the dict
            v_dict["Por"] = [float(p) for p in porMat]
            count = count + 1

        # match l_split[0].strip():       # Switch with leading word
        #     case "ANGLE":
        #         v_dict["ang"] = float(l_split[1])
        #         count = count + 1
        #     case "VELOCITY":
        #         v_dict["vIn"] = float(l_split[1])
        #         count = count + 1
        #     case "POROSITY":
        #         # strip trailing spaces and split with '[', ',' and ']'
        #         porMat = re.split('[\[,\]]', l_split[1].strip())
        #         # remove empty values resulting from regexp and return list
        #         porMat = list(filter(None, porMat))
        #         # store as list of floats in the dict
        #         v_dict["Por"] = [float(p) for p in porMat]
        #         count = count + 1

        if count == 3: break  # Only 3 lines have to be read

    return v_dict


def load_file(dataset, folder, f):
    # read run variables 'vIn', 'ang' and '[Por]' into a dictionary
    run_vars = parse_run_variables(join(folder, f))

    # store run variables (for debugging purposes)
    dataset.vIn.append(run_vars["vIn"])
    dataset.ang.append(run_vars["ang"])

    # get Upwind and Downwind velocity and pressure fields
    x1, x2, x3, x4 = get_fields(join(folder, f))

    return x1, x2, x3, x4, run_vars


def load_files(dataset, folder, files):
    data_list = list()
    # load each file
    for f in files:
        x1, x2, x3, x4, run_vars = load_file(dataset, folder, f)
        data_list.append((x1, x2, x3, x4, run_vars))
    return data_list


class AlyaDataset(Dataset):
    """ Read the files in folder and get the data in torch dataset format"""

    POR_NORM = 1e6

    # -------
    # TODO : Modification pending, not needed right now for cnn_a training
    # -------
    # Function to get an estimated porosity from the UpWind and DownWind fields
    def estimate_porosity(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device = {device}")

        currentdir = os.path.dirname(os.path.realpath(__file__))
        parentdir = os.path.dirname(currentdir) + "/model_surrogate"
        print(parentdir)
        sys.path.append(parentdir)

        model = torch.load("../model_surrogate/cnnA", map_location=device).to(device)
        model.eval()
        tU = self.x1_data
        tD = self.x2_data
        tA = self.a_data
        mask = torch.ones(self.y_data.numel(), dtype=torch.bool)
        for i, y in enumerate(self.y_data):
            # switch off autograd
            with torch.no_grad():
                # send the input to the device and make predictions on it
                (tensU, tensD, tensA) = (tU[i:i + 1].to(device),
                                         tD[i:i + 1].to(device),
                                         tA[i:i + 1].to(device))
                self.y_data[i][0] = model(tensU, tensD, tensA).item()
                print(f"y_data = {self.y_data[i]}")
            if 1e-4 > self.y_data[i][0] or self.y_data[i][0] > 8e-1:
                mask[i] = False
        self.y_data = self.y_data[mask]
        self.a_data = self.a_data[mask]
        self.x1_data = self.x1_data[mask]
        self.x2_data = self.x2_data[mask]

    # -------

    # Initialization function needs a folder containing the results files
    # from Alya
    def __init__(self, folder, remote_data, use_cnn_a=False):
        file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
        tmp_x1 = []  # Input variable x1 == V field Upwind
        tmp_x2 = []  # Input variable x2 == V field Downwind
        tmp_x3 = []  # Input variable x3 == Press field Upwind
        tmp_x4 = []  # Input variable x4 == Press field Downwind
        tmp_y = []  # Target variable y == [Por]
        self.vIn = []  # simulation V modulus set
        self.ang = []  # simulation V angle set

        start_time = time.time()

        n_workers = os.cpu_count() if remote_data else 2
        chunk_size = round(len(file_list) / n_workers)

        with ThreadPoolExecutor(n_workers) as executor:
            futures = list()
            # split the load operations into chunks
            for i in range(0, len(file_list), chunk_size):
                # select a chunk of filenames
                filepaths = file_list[i:(i + chunk_size)]

                # submit the task
                future = executor.submit(load_files, self, folder, filepaths)
                futures.append(future)

            for future in as_completed(futures):
                data_list = future.result()

                for x1, x2, x3, x4, run_vars in data_list:
                    # store in temporal lists
                    tmp_x1.append(x1)
                    tmp_x2.append(x2)
                    tmp_x3.append(x3)
                    tmp_x4.append(x4)
                    # store porosity matrix in temporal list
                    tmp_y.append([p / self.POR_NORM for p in run_vars["Por"]])

        elapsed_time = time.time() - start_time
        print("Alya files loaded in %f seconds" % elapsed_time)

        # Convert list data into pyTorch tensor file format
        # for [Por] = [p1, p2, p3, p4]
        # Get 2D array [[y1...y4][y2...y5]...[yn...yn+3]]
        arr_y = np.array(tmp_y).reshape(-1, 4)
        self.y_data = torch.tensor(arr_y, dtype=torch.float32)  # tensor

        # get X and Y dims for Upwind fields
        n_x, n_y = (tmp_x1[0].shape[1], tmp_x1[0].shape[2])
        # for tmp_x1 = [ [ [ [vx_1...vx_ny]_1 ... [vx_1...vy_ny]_nx ]
        #                  [ [vy_1...vy_ny]_1 ... [vy_1...vy_ny]_nx ] ] ... ]
        arr_x1 = np.array(tmp_x1).reshape(-1, 2, n_x, n_y)
        self.x1_data = torch.tensor(arr_x1, dtype=torch.float32)  # tensor

        # Convert Upwind Pressure field
        # for tmp_x3 = [ [ [P_1...P_ny]_1 ... [P_1...P_ny]_nx ] ... ]
        arr_x3 = np.array(tmp_x3).reshape(-1, n_x, n_y)
        self.x3_data = torch.tensor(arr_x3, dtype=torch.float32)  # tensor

        # repeat for Downwind velocity and pressure fields
        n_x, n_y = (tmp_x2[0].shape[1], tmp_x2[0].shape[2])
        arr_x2 = np.array(tmp_x2).reshape(-1, 2, n_x, n_y)
        self.x2_data = torch.tensor(arr_x2, dtype=torch.float32)  # tensor
        arr_x4 = np.array(tmp_x4).reshape(-1, n_x, n_y)
        self.x4_data = torch.tensor(arr_x4, dtype=torch.float32)

        if use_cnn_a:
            self.estimate_porosity()


    def transform_porosity(self):
        self.y_data = torch.log(self.y_data)

        # scaler = MinMaxScaler()
        # for i in range(self.y_data.size(dim=1)):
        #    v = self.y_data[:, i].reshape(-1, 1)
        #    scaled_column = scaler.fit_transform(v)
        #    self.y_data[:, i] = torch.tensor(scaled_column[:,0], dtype=torch.float32)]

    def plot_data(self):
        pd_df = pd.DataFrame(self[:][4].numpy())
        pd_df.columns = ['0', '1', '2', '3']

        pd_df_log = np.log(pd_df)
        pd_df_sqrt = np.sqrt(pd_df)

        plt.style.use("ggplot")
        figure, axis = plt.subplots(2, 2, figsize=(45, 30))
        axis[0, 0].plot(pd_df['0'])
        axis[0, 1].plot(pd_df['1'])
        axis[1, 0].plot(pd_df['2'])
        axis[1, 1].plot(pd_df['3'])
        plt.savefig(fname='por_plots.png')

        figure, axis = plt.subplots(2, 2, figsize=(45, 30))
        axis[0, 0].hist(pd_df['0'], bins=50)
        axis[0, 1].hist(pd_df['1'], bins=50)
        axis[1, 0].hist(pd_df['2'], bins=50)
        axis[1, 1].hist(pd_df['3'], bins=50)
        plt.savefig(fname='por_histograms.png')

        figure, axis = plt.subplots(2, 2, figsize=(45, 30))
        axis[0, 0].hist(pd_df_log['0'], bins=50)
        axis[0, 1].hist(pd_df_log['1'], bins=50)
        axis[1, 0].hist(pd_df_log['2'], bins=50)
        axis[1, 1].hist(pd_df_log['3'], bins=50)
        plt.savefig(fname='por_histograms_log.png')

        figure, axis = plt.subplots(2, 2, figsize=(45, 30))
        axis[0, 0].hist(pd_df_sqrt['0'], bins=50)
        axis[0, 1].hist(pd_df_sqrt['1'], bins=50)
        axis[1, 0].hist(pd_df_sqrt['2'], bins=50)
        axis[1, 1].hist(pd_df_sqrt['3'], bins=50)
        plt.savefig(fname='por_histograms_sqrt.png')

    def get_input(self):
        return self.x1_data, self.x2_data

    # return size of the dataset
    def __len__(self):
        return len(self.y_data)

    # function to get an item from the dataset
    def __getitem__(self, idx):
        v_uw_field = self.x1_data[idx, :]
        v_dw_field = self.x2_data[idx, :]
        p_uw_field = self.x3_data[idx, :]
        p_dw_field = self.x4_data[idx, :]
        poro = self.y_data[idx, :]
        return v_uw_field, v_dw_field, p_uw_field, p_dw_field, poro

    # function to get targets for testing
    def targets(self):
        return self.y_data[:, 0]
