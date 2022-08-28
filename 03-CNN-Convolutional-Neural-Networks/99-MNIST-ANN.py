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
# First, we try with an ANN

# We need transforms to convert the images to tensors
transform = transforms.ToTensor()

# If the data is not available, we download it
if not (os.path.exists('../Data/MNIST')):
    train_data = datasets.MNIST(root='../Data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='../Data/', train=False, download=True, transform=transform)
else:
    train_data = datasets.MNIST(root='../Data/', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='../Data/', train=False, download=False, transform=transform)

# Look to the data
print(f'Train Shape: {train_data.data.shape}, Test Shape: {test_data.data.shape}')
# print(f'One sample: {train_data[0]} of type {type(train_data[0])}')

# Show one image
image, label = train_data[0]
plt.imshow(image.reshape((28, 28)), cmap='gist_yarg')

# ---

torch.manual_seed(101)

# Create the dataloaders with the train and test data sets and the batch size
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

# Formatting
from torchvision.utils import make_grid

np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))

# Show first batch to picture
for images, labels in train_loader:
    break

# Print the first 12 labels
print("Labels:", labels[:12].numpy())

# Print the first 12 images
im = make_grid(images[:12], nrow=12)

plt.figure(figsize=(10, 4))
# We need to transpose the images from CWH to WHC
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()


# ---

# Design the ANN
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_size=784, out_size=10, layers=[120, 84]):
        super().__init__()

        # Create the layers of the model
        self.fc1 = nn.Linear(in_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_size)

    def forward(self, x):
        # Pass the input through the activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # We need to use softmax to get the probabilities of multiclass classification
        return F.log_softmax(x, dim=1)  # dim=1 -> columns


torch.manual_seed(101)

# Create the model
model = MultilayerPerceptron()
print(model)

# Each connection has a weight, each neuron has a bias -> ANN has 105,214 parameters!
for param in model.parameters():
    print(param.numel())

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert from [100, 1, 28, 28] to [100, 784]
images.view(100, -1)  # Grab those dimensions and combine them (-1)

# ---

import time

start_time = time.time()

epochs = 4

# Trackers
train_losses = []  # For losses
test_losses = []
train_correct = []  # For accuracy
test_correct = []

for epoch in range(epochs):
    # Number of corrects in the epoch
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):

        # We are now in one batch of batch_size images and labels

        # Start with batch 1
        b = b + 1

        # Forward pass
        y_pred = model(X_train.view(100, -1))
        loss = criterion(y_pred, y_train)

        # Metrics
        predicted = torch.max(y_pred.data, 1)[1]  # Get the index of the max probability
        batch_corr = (predicted == y_train).sum()  # Count the number of corrects
        trn_corr += batch_corr

        optimizer.zero_grad()  # Reset the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

        # If the batch_size is 100, and the total size is 60000, we have 600 batches
        # This will print 6 times
        if (b % 100 == 0):
            accuracy = trn_corr.item() * 100 / (100 * b)  # Accuracy in the epoch
            print(f'Epoch: {epoch} Batch: {b} Loss: {loss.item():.4f} accuracy: {accuracy}')

    # Append the loss and accuracy of the epoch
    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Test the model
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500, -1))

            # Metrics
            predicted = torch.max(y_val.data, 1)[1]  # Get the index of the max probability
            tst_corr = (predicted == y_test).sum()  # Count the number of corrects

        # We are only testing the data, not changing the network
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time / 60} minutes')

# ---

# Detach the losses
train_losses = [loss.detach().numpy() for loss in train_losses]
test_losses = [loss.detach().numpy() for loss in test_losses]

# Plot the losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test/Validation Loss')
plt.legend()
plt.show()

print(train_correct) # How many where correct for each epoch?
train_acc = [t/600 for t in train_correct] # Divide by the number of batches
test_acc = [t/100 for t in train_correct] # Divide by the number of batches
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test/Validation Accuracy')
plt.legend()
plt.show()

# ---

# New unseen data
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:

        # Forward pass
        y_val = model(X_test.view(len(X_test), -1))

        # Metrics
        predicted = torch.max(y_val.data, 1)[1]
        correct += (predicted == y_test).sum()

# Accuracy
print(100 * correct.item()/len(X_test))

