# I will make the neural network the f(x) = sin(x)
# author: aadit

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# f_x
def f_x(x):
    return np.sin(x)

# Build dataset
def build_dataset():
    data = []
    for i in range(1,300):
        data.append((i, f_x(i), 1)) # 1 stands for "correct value"
    for j in range(300, 600):
        data.append((j, np.cos(j), 0)) # 0 stands for "incorrect value"
    
    df = 
    return data

