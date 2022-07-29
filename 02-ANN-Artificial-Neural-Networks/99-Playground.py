import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

"""
x = torch.tensor(2.0, requires_grad=True)
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1

y.backward() # Compute the gradient of y with respect to x

print(y)
print(x.grad) # Plugin x value in first derivative: x.grad is the gradient of y with respect to x

x = torch.tensor([[1., 2., 3.],[3., 2., 1.]], requires_grad=True)
print(x)

y = 3*x + 2
print(y)

z = 2*y**2
print(z)

out = z.mean()
print(out)
#
# # Backpropagation to find the gradient of x with respect to the output layer
# out.backward()
# print(x.grad)
"""

# Create a 1D tensor of 50 elements
X = torch.linspace(1, 50, 50).reshape(-1, 1)
# print(X)

# Create a 2D tensor of 50x1 elements for the error
torch.manual_seed(71)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
# print(e)

# Underlying function (true function) to be approximated
y = 2*X+1 + e
# print(y.shape)

plt.scatter(X.numpy(), y.numpy())

torch.manual_seed(59)

# model = nn.Linear(in_features=1, out_features=1)
# print(model.weight)
# print(model.bias)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

torch.manual_seed(59)

model = Model(in_features=1, out_features=1)
print(model.linear.weight.shape) # Linear layer weights randomly initialized
print(model.linear.bias.shape) # Linear layer bias randomly initialized

# for name, param in model.named_parameters():
#     print(name, '\t', param.shape, '\t', param.item())

x = torch.tensor([[2.0]])
# print(model.forward(x))

x1 = np.linspace(0,50.,50)

w1 = 0.1059
b1 = 0.9637

y1 = w1*x1 + b1
# print(y1)

plt.plot(x1, y1, 'red')

# Loss
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Epoch = entire pass though the whole dataset

epochs = 50
losses = []
for i in range(epochs):
    i = i+1
    # Predict in forward pass
    y_pred = model.forward(X)
    # Calculate our loss (Error)
    loss = criterion(y_pred, y)
    # Record the error
    losses.append(loss)
    print(f"epoch{i} loss {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}")
    # Reset gradients to avoid accumulation
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    optimizer.step()

# New list of losses deatached from the graph
losses = [loss.detach().numpy() for loss in losses]

plt.plot(X, y_pred.detach().numpy(), 'blue')
plt.plot(range(epochs), losses, 'green')
plt.ylabel('MSE LOSS')
plt.xlabel('Epoch')
plt.show()