import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import os, sys


# %% Read dataset from Alya "results" folder and store it in Torch Tensor format

# get vector fields (2 channels: mod, ang) from an alya results file
def get_vector_field(file):
    f = open(file, 'r')  # Open file with field data
    lines = f.readlines()  # read all the lines
    f.close()  # Close file
    data = []  # Init data list
    for l in lines[1:]:  # Discard header line
        # Append [Xcoord, Ycoord, Vx, Vy] from each line to data
        data.append([float(d) for d in l.split()])
    np_data = np.array(data)  # Convert to numpy array
    x_list = sorted([*set(np_data[:, [0]].flatten())])  # get a list of x coords set
    y_list = sorted([*set(np_data[:, [1]].flatten())])  # and also from y
    n_x, n_y = (len(x_list), len(y_list))  # dimensions of the field
    v_field = np.zeros((2, n_x, n_y))  # 2 channels field initialization: vx and vy
    for d in np_data:
        x, y = (d[0], d[1])  # coordinates
        vx, vy = (d[2], d[3])  # velocity components
        idx_x = x_list.index(x)  # x and y coordinates indices
        idx_y = y_list.index(y)
        # ch1: Velocity field component X
        v_field[0, idx_x, idx_y] = vx
        # ch2: Velocity field component Y
        v_field[1, idx_x, idx_y] = vy
    return v_field


class AlyaDataset(Dataset):
    """ Read the files in folder and get the data in torch dataset format"""

    POR_NORM = 1e5
    ANG_NORM = 45  # Divider to normalize angle value

    # Function to get an estimated porosity from the UpWind and DownWind fields
    def estimate_porosity(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device = {device}")

        currentdir = os.path.dirname(os.path.realpath(__file__))
        parentdir = os.path.dirname(currentdir) + "/model_a"
        print(parentdir)
        sys.path.append(parentdir)

        model = torch.load("../model_a/cnnA", map_location=device).to(device)
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

    # Initialization function needs a folder (train and test DS separation by folders)
    def __init__(self, folder, use_cnn_a=False):
        file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
        tmp_x1 = []
        tmp_x2 = []
        tmp_a = []
        tmp_y = []
        self.vIn = []
        # Process all the files inside folder
        for f in file_list:
            # Split filename 'simtype'-'Num'-'U/DW'-'ang'-'vel'-'poro.txt'
            name_split = f.split('-')
            if name_split[2] == "UPW":
                # save vIn
                self.vIn.append(float(name_split[4]))
                # store angle
                ang = float(name_split[3])
                # angle value normalized
                tmp_a.append(ang / self.ANG_NORM)  # list with ang values

                # UpWind data is inside the file being processed
                tmp_x1.append(get_vector_field(join(folder, f)))
                # get corresponding DownW filename from the UpWind file name
                file_name_dw = (name_split[0] + '-' + name_split[1] + '-' + "DOWNW" +
                                '-' + name_split[3] + '-' + name_split[4] +
                                '-' + name_split[5])
                tmp_x2.append(get_vector_field(join(folder, file_name_dw)))

                # take out '.txt' from porosity value and normalize it
                por = float(name_split[-1][:-4])
                tmp_y.append(por / self.POR_NORM)  # list with Y values

        # Convert array data into pyTorch tensor file format
        arr_y = np.array(tmp_y).reshape(-1, 1)  # Get 2D array [[y1][y2]...[yn]]
        self.y_data = torch.tensor(arr_y, dtype=torch.float32)  # store as tensor

        arr_a = np.array(tmp_a).reshape(-1, 1)  # Get 2D array [[a1][a2]...[an]]
        self.a_data = torch.tensor(arr_a, dtype=torch.float32)  # store as tensor

        n_x, n_y = (tmp_x1[0].shape[1], tmp_x1[0].shape[2])  # get X and Y dims for X1
        arr_x1 = np.array(tmp_x1).reshape(-1, 2, n_x, n_y)
        self.x1_data = torch.tensor(arr_x1, dtype=torch.float32)  # store as tensor

        n_x, n_y = (tmp_x2[0].shape[1], tmp_x2[0].shape[2])  # get X and Y dims for X2
        arr_x2 = np.array(tmp_x2).reshape(-1, 2, n_x, n_y)
        self.x2_data = torch.tensor(arr_x2, dtype=torch.float32)  # store as tensor

        if use_cnn_a:
            self.estimate_porosity()

    # return size of the dataset
    def __len__(self):
        return len(self.y_data)

    # function to get an item from the dataset
    def __getitem__(self, idx):
        uw_field = self.x1_data[idx, :]
        dw_field = self.x2_data[idx, :]
        ang = self.a_data[idx, :]
        poro = self.y_data[idx, :]
        return uw_field, dw_field, ang, poro

    # function to get targets for testing
    def targets(self):
        return self.y_data[:, 0]
