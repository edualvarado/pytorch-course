import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/iris.csv')
# print(df.head())
# print(df.shape)

# 1. Method
from sklearn.model_selection import train_test_split

features = df.drop('target', axis=1).values
label = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train).reshape(-1, 1) # reshape to one column vector
y_test = torch.LongTensor(y_test).reshape(-1, 1) # reshape to one column vector

# 2. Method with Pytorch
from torch.utils.data import TensorDataset, DataLoader

features2 = df.drop('target', axis=1).values
label2 = df['target'].values

# Create a TensorDataset that contains the features and labels in the form of tensors
iris = TensorDataset(torch.FloatTensor(features2), torch.LongTensor(label2))
# for i in iris:
#     print(i)

# DataLoader is a data loader wrapper that wraps a dataset and provides iterators over the dataset
iris_loader = DataLoader(iris, batch_size=50, shuffle=True)
# for i_batch, sample_batch in enumerate(iris_loader):
#     print(i_batch, sample_batch)