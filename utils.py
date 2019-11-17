import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing

def load_data(filepath=r"data.csv", isTraining=True, days=7, input_size=1):
    data = np.array(pd.read_csv(filepath,encoding="utf-8"))[:,1:input_size+1]
    if isTraining:
        data = data[:int(data.shape[0]*0.7), :]
    else:
        data = data[int(data.shape[0]*0.7):, :]
    scale = preprocessing.StandardScaler()
    data = scale.fit_transform(data)
    x_data, y_data = [], []
    for index in range(data.shape[0]-days):
        x_data.append(data[index:(index+days), :])
        y_data.append(data[index+days,0])
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)
    return torch.Tensor(x_data), torch.Tensor(y_data)