import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# MNIST -> 28x28 images of hand-written digits 0-9

transform = transforms.ToTensor()

if not (os.path.exists('../Data/MNIST')):
    train_data = datasets.MNIST(root='../Data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='../Data/', train=False, download=True, transform=transform)
else:
    train_data = datasets.MNIST(root='../Data/', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='../Data/', train=False, download=False, transform=transform)

print(train_data.data.shape)
print(train_data[0])
print(type(train_data[0]))

image, label = train_data[0]
print(image.shape)
print(label)

# plt.imshow(image.reshape((28,28))) #viridis
plt.imshow(image.reshape((28, 28)), cmap='gist_yarg')  # viridis

# ---

torch.manual_seed(101)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

from torchvision.utils import make_grid

np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))  # Formatting

# First batch to picture
for images, labels in train_loader:
    break

# print(images.shape)

# Print the first 12 labels
print("Labels:", labels[:12].numpy())

# Print the first 12 images
im = make_grid(images[:12], nrow=12)
plt.figure(figsize=(10, 4))
# We need to transpose the images from CWH to WHC
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()


# ---

# First, ANN

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120, 84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Multiclass classification


torch.manual_seed(101)

model = MultilayerPerceptron()
print(model)

# Each connection is a weight, each neuron has a bias -> ANN has 105,214 parameters!
for param in model.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to [100, 784]
images.view(100, -1)  # Grab those dimensions and combine them (-1)

# ---

import time

start_time = time.time()

epochs = 2

# Trackers
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train,y_train) in enumerate(train_loader):

        # Start with batch 1
        b+=1

        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred, y_train)

        # Metrics
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(b % 200 == 0):
            accuracy = trn_corr.item()*100/(100*b)
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item():.4f} accuracy: {accuracy}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500,-1))

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr = (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time / 60} mins')

# ---

train_losses = [loss.detach().numpy() for loss in train_losses]
test_losses = [loss.detach().numpy() for loss in test_losses]

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test/Validation Loss')
plt.legend()

print(train_correct) # How many where correct for each epoch
train_acc = [t/600 for t in train_correct]
test_acc = [t/100 for t in train_correct]
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test/Validation Accuracy')
plt.show()

# New unseen data

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0

    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test),-1))
        predicted = torch.max(y_val.data, 1)[1]
        correct += (predicted == y_test).sum()

100*correct.item()/len(X_test) # Accuracy

confusion_matrix(predicted.view(-1),y_test.view(-1))