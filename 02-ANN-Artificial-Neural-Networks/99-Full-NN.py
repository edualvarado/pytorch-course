import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # How many layers?
        # Input Layer (4 features) --> h1 (n Neurons) --> h2 (n Neurons) --> Output Layer (3 classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(32)

model = Model()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/iris.csv')
# print(df.head())
# print(df.tail())
# print(df.shape)


# ---

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
# fig.tight_layout()
#
# plots = [(0,1),(2,3),(0,2),(1,3)]
# colors = ['b', 'r', 'g']
# labels = ['Iris setosa','Iris virginica','Iris versicolor']
#
# for i, ax in enumerate(axes.flat):
#     for j in range(3):
#         x = df.columns[plots[i][0]]
#         y = df.columns[plots[i][1]]
#         ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
#         ax.set(xlabel=x, ylabel=y)
#
# fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
#plt.show()

# ---

X = df.drop('target', axis=1).values # .values to convert in numpy arrays
y = df['target'].values

# from torch.utils.data import TensorDataset, DataLoader
#
# TensorDataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
# DataLoader = DataLoader(TensorDataset, batch_size=50, shuffle=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# Convert to torch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# print(y_train)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

criterion = nn.CrossEntropyLoss() # Multi-class Classification Problem
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model.parameters)

# Epoch = 1 one through all the training data (seen all training data once)
epoch = 500
losses = []

for i in range(epoch):
    # Forward pass
    y_pred = model(X_train)

    # Calculate loss or error
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'Epoch: {i} | Loss: {losses[i]}')

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# New list of losses deatached from the graph
losses = [loss.detach().numpy() for loss in losses]

plt.plot(range(epoch), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

# No backpropagation, it turns it off to speed up computation, as it only needs to infer
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

print(f'Test Loss: {loss}')

correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1}.) {y_val} | {y_test[i]}')
        if y_val.argmax() == y_test[i]:
            correct += 1

print(f'Accuracy: {correct/len(X_test)*100}%')

torch.save(model.state_dict(), 'my_iris_model.pt')
# This assumes that you have still the original model class.
# To save the entire model, you can use torch.save(model, 'my_iris_model.pt')

new_model = Model() # W or B?
new_model.load_state_dict(torch.load('my_iris_model.pt'))

new_model.eval()

# Now, we classify new data

mistery_iris = torch.tensor([[5.6, 3.7, 2.2, 0.5]])

with torch.no_grad():
    y_pred = new_model.forward(mistery_iris)
    print(y_pred)
    print(y_pred.argmax())