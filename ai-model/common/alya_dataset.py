import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import os, sys
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# current_dir = os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
from model_surrogate.convolutional_network import ConvolutionalNetwork


# %% Read dataset from Alya "results" folder and store it in Torch Tensor format

# get vector fields (3 channels: vx, vy, p) from an alya results file
def get_fields(lines, time_steps):
    dataUS = []  # Init Upstream data list
    dataDS = []  # Init Downstream data list

    # get Upstream and downstream data from file
    UPS_flag = False
    DWS_flag = False
    for i, l in enumerate(lines):
        if (l.split()[0] == "UPSTREAM"):  # Search for Upstream data start
            UPS_flag = True  # Flag start of Upstream data
        elif (UPS_flag):  # Read Upstream data
            # Start of Downstream data
            if (l.split()[0] == "DOWNSTREAM"):
                UPS_flag = False
                DWS_flag = True
            else:
                # Append [Xcoord, Ycoord, Vx, Vy, P] from each line to dataUS
                dataUS.append([float(d) for d in l.split()])
        elif (DWS_flag):
            # Append [Xcoord, Ycoord, Vx, Vy, P] from each line to dataUS
            dataDS.append([float(d) for d in l.split()])

    # Convert to numpy arrays
    np_dataUS = np.array(dataUS)
    np_dataDS = np.array(dataDS)

    cols = np.shape(np_dataUS)[1]
    actual_time_steps = cols // 5
    redundant_cols = cols % 5

    if redundant_cols > 0:
        cols_to_delete = list(range(cols - redundant_cols, cols))

        np_dataUS = np.delete(np_dataUS, cols_to_delete, axis=1)
        np_dataDS = np.delete(np_dataDS, cols_to_delete, axis=1)

    np_data_us_t = np.hsplit(np_dataUS, actual_time_steps)
    np_data_ds_t = np.hsplit(np_dataDS, actual_time_steps)

    v_UPS = []
    v_DWS = []
    p_UPS = []
    p_DWS = []

    for data_us_t, data_ds_t in zip(np_data_us_t, np_data_ds_t):
        v_ups_item = []
        v_dws_item = []
        p_ups_item = []
        p_dws_item = []

        for i, d_array in enumerate([data_us_t, data_ds_t]):
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
                if i == 0:
                    v_ups_item = v_field[:]
                    p_ups_item = p_field[:]
                else:
                    v_dws_item = v_field[:]
                    p_dws_item = p_field[:]

        v_UPS.append(v_ups_item)
        v_DWS.append(v_dws_item)
        p_UPS.append(p_ups_item)
        p_DWS.append(p_dws_item)

    return v_UPS, v_DWS, p_UPS, p_DWS


# Parser for getting simulation variables from file
# v_dict -> {"vIn" : [float], "ang" : [float], "por" : [float, float, float, float], "time_steps" : [int] }
def parse_run_variables(file):
    with open(file, 'r') as f:  # Open file with field data
        lines = f.readlines()  # read all the lines

    v_dict = {
        'vIn': None,  # Store values in dictionary
        'ang': None,
        'por': None,
        'time_steps': None  # Store number of time steps
    }

    prev_l_split = []
    for l in lines:
        l_split = l.split(':')  # separate leading word
        if "ANGLE" in l:
            v_dict["ang"] = float(l_split[1])
        elif "VELOCITY" in l:
            v_dict["vIn"] = float(l_split[1])
        elif "POROSITY" in l:
            # strip trailing spaces and split with '[', ',' and ']'
            porMat = re.split('[\[,\]]', l_split[1].strip())
            # remove empty values resulting from regexp and return list
            porMat = list(filter(None, porMat))
            # store as list of floats in the dict
            v_dict["Por"] = [float(p) for p in porMat]
        elif "Time Steps Values" in l:
            v_dict["time_steps"] = int(l_split[1])

        if "Time Steps Values" in prev_l_split:
            v_dict["time_step_values"] = l.strip().split('\t')

        prev_l_split = l_split

        if "UPSTREAM FIELDS" in l:
            break  # Only the first lines have to be read

    return v_dict, lines


class AlyaDataset(Dataset):
    """ Read the files in folder and get the data in torch dataset format"""

    POR_NORM = 5e4

    # Function to get an estimated porosity from the UpWind and DownWind fields
    def estimate_porosity(self, surrCNN):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"device = {device}")

        currentdir = os.path.dirname(os.path.realpath(__file__))
        modeldir = os.path.dirname(currentdir) + "/model_surrogate"
        # sys.path.append(parentdir)

        model = torch.load(f"{modeldir}/{surrCNN}", map_location=device).to(device)
        model.eval()
        tVU = self.x1_data
        tVD = self.x2_data
        tPU = self.x3_data
        tPD = self.x4_data
        with torch.no_grad():
            # send the input to the device and make predictions on it
            (tensVU, tensVD) = (tVU.to(device), tVD.to(device))
            y_list = model(tensVU, tensVD).tolist()
            return y_list

    # Initialization function needs a folder containing the results files
    # from Alya
    def __init__(self, folder, use_cnn_a=False, surrCNN=""):
        file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
        tmp_x1 = []  # Input variable x1 == V field Upwind
        tmp_x2 = []  # Input variable x2 == V field Downwind
        tmp_x3 = []  # Input variable x3 == Press field Upwind
        tmp_x4 = []  # Input variable x4 == Press field Downwind
        tmp_y = []  # Target variable y == [Por]
        self.vIn = []  # simulation V modulus set
        self.ang = []  # simulation V angle set

        print("Starting loading of Alya files...")

        # Process all the files inside folder
        for i, f in enumerate(file_list):
            # read run variables 'vIn', 'ang' and '[Por]' into a dictionary
            # when reading high resolution model files, porosities have 0 value
            run_vars, lines = parse_run_variables(join(folder, f))

            # store run variables (for debugging purposes)
            self.vIn.append(run_vars["vIn"])
            self.ang.append(run_vars["ang"])

            # get Upwind and Downwind velocity and pressure fields
            # v_ups, v_dws, p_ups, p_dws
            x1, x2, x3, x4 = get_fields(lines, run_vars["time_steps"])
            # store in temporal lists
            tmp_x1.append(x1)
            tmp_x2.append(x2)
            tmp_x3.append(x3)
            tmp_x4.append(x4)
            # store porosity matrix in temporal list
            tmp_y.append([p / self.POR_NORM for p in run_vars["Por"]])

            if i % 100 == 0:
                print(f"Loaded {i} Alya files")

        print("Finished loading of Alya files")

        # get X and Y dims for Upwind fields
        n_x, n_y = (tmp_x1[0][0].shape[1], tmp_x1[0][0].shape[2])

        # for tmp_x1 = [ [ [ [vx_1...vx_ny]_1 ... [vx_1...vy_ny]_nx ]
        #                  [ [vy_1...vy_ny]_1 ... [vy_1...vy_ny]_nx ] ] ... ]
        arr_x1 = np.array(tmp_x1).reshape(1, -1, 2, n_x, n_y)
        self.x1_data = torch.tensor(arr_x1, dtype=torch.float32)  # tensor

        # Convert Upwind Pressure field
        # for tmp_x3 = [ [ [P_1...P_ny]_1 ... [P_1...P_ny]_nx ] ... ]
        arr_x3 = np.array(tmp_x3).reshape(1, -1, n_x, n_y)
        self.x3_data = torch.tensor(arr_x3, dtype=torch.float32)  # tensor

        # repeat for Downwind velocity and pressure fields
        n_x, n_y = (tmp_x2[0][0].shape[1], tmp_x2[0][0].shape[2])
        arr_x2 = np.array(tmp_x2).reshape(1, -1, 2, n_x, n_y)
        self.x2_data = torch.tensor(arr_x2, dtype=torch.float32)  # tensor
        arr_x4 = np.array(tmp_x4).reshape(1, -1, n_x, n_y)
        self.x4_data = torch.tensor(arr_x4, dtype=torch.float32)

        # Convert list data into pyTorch tensor file format
        # for [Por] = [p1, p2, p3, p4]
        # Get 2D array [[y1...y4][y2...y5]...[yn...yn+3]]
        if use_cnn_a:
            # self.estimate_porosity()
            tmp_y = self.estimate_porosity(surrCNN)
        npor = len(tmp_y[0])
        arr_y = np.array(tmp_y).reshape(-1, npor)
        self.y_data = torch.tensor(arr_y, dtype=torch.float32)  # tensor   

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
        time_idx = 0

        v_uw_field = self.x1_data[time_idx, idx, :]
        v_dw_field = self.x2_data[time_idx, idx, :]
        p_uw_field = self.x3_data[time_idx, idx, :]
        p_dw_field = self.x4_data[time_idx, idx, :]
        poro = self.y_data[idx, :]
        return v_uw_field, v_dw_field, p_uw_field, p_dw_field, poro

    # function to get targets for testing
    def targets(self):
        return self.y_data[:, 0]
