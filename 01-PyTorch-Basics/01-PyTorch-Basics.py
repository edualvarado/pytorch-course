import numpy as np
import numpy.random
import torch

print(torch.__version__)

"""
# PART ONE - Tensor Basics

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

# Convert to tensor
x = torch.from_numpy(arr)
print(x)
print(type(x))
print(x.dtype)

# Convert to tensor
torch.as_tensor(arr)

arr2d = np.arange(0.0, 12.0).reshape(4, 3)
print(arr2d)

x2 = torch.from_numpy(arr2d)
print(x2)

# Change numpy
arr[0] = 99
print(arr)

# Affects the tensor!
print(x)

# Most common function to convert numpy to tensor if you do not want a direct link but copy, so they do not share the
# same memory spot

my_arr = np.arange(0, 10)
my_tensor = torch.tensor(my_arr)  # no link
my_other_tensor = torch.from_numpy(my_arr)  # link (a modification will change the tensor)

print(my_tensor)
print(my_other_tensor)

my_arr[0] = 99
print(my_tensor)
print(my_other_tensor)

# PART TWO - Tensor Basics

"""
# torch.Tensor(10) will return an uninitialized FloatTensor with 10 values, while torch.tensor(10) will return
# a LongTensor containing a single value (10).
# I would recommend to use the second approach (lowercase t) or any other factory method instead of
# creating uninitialized tensors via torch.Tensor or torch.FloatTensor.
# To explicitly create a tensor with uninitialized memory, you could still use torch.empty.
"""

new_arr = np.array([1, 2, 3, 4, 5])
print(new_arr.dtype)
print(torch.tensor(new_arr))  # infers the type of the tensor from the original object
print(torch.tensor(new_arr, dtype=torch.float32))  # explicitly specify the type
print(torch.Tensor(new_arr))  # check type of the tensor

# Initialize tensors or placeholders (empty) - block of memory allocated according to the shape of the tensor
# The shape of the tensor is the number of elements in each dimension

print(torch.empty(2, 2))  # 2x2 tensor with uninitialized memory

print(torch.zeros(4, 3, dtype=torch.float32))  # 4x3 tensor with zeros
# Careful with the dtype!

print(torch.ones(2, 2, dtype=torch.float32))  # 2x2 tensor with ones

print(torch.arange(0, 10, 2))  # 0 to 9

print(torch.arange(0, 18, 2).reshape(3, 3))

print(torch.linspace(0, 10, steps=3))  # 0 to 10 with 3 steps

# Convert list to tensor
my_tensor = torch.tensor([1, 2, 3])
print(my_tensor.dtype)

# Other dtype
torch.tensor([1, 2, 3], dtype=torch.float32)

# Convert type
my_tensor = my_tensor.type(torch.float32)
print(my_tensor.dtype)

# Random tensors from uniform distribution from 0 and 1
print(torch.rand(4, 2))  # random tensor with 2x2 elements

# Normal distribution
print(torch.randn(4, 2))  # random tensor with 2x2 elements with mean 0 and standard deviation 1

# Random tensors (high exclusive)
print(torch.randint(0, 10, (3, 3)))  # random tensor with 3x3 elements with values between 0 and 10


x = torch.zeros(2, 5)
print(x)
print(x.size())
print(x.shape)

print(torch.rand_like(x))  # random tensor with the same shape as x
print(torch.randn_like(x))  # random tensor with the same shape as x with mean 0 and standard deviation 1

# Set a seed
torch.manual_seed(42)
print(torch.rand(2, 3))
"""

# PART ONE - Tensor Operations

"""
x = torch.arange(6).reshape(3, 2)
print(x)
print(x[1, 1])
print(x[1, 1].dtype)
print(type(x[1, 1]))

print(x[:, 1])  # Does not retain shape
print(x[:, 1:])  # Retains shape

x = torch.arange(10)
# Does not change the tensor
print(x.view(2, 5))
print(x)
# Does not change the tensor
print(x.reshape(2, 5))
print(x)

# Way to change the original by reassignment
x = x.reshape(2, 5)
print(x)

print("Now")
x = torch.arange(10)
print(x)
print(x.shape)

z = x.view(2, 5)
# Now z is linked to x!
print(z)

x[0] = 99
print("After changing number")
print(x)
print(z)

x = torch.arange(10)
print(x)
print(x.shape)

# Infer what the second dimension should be
print(x.view(2, -1))

print("Sum")
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(a + b)  # elementwise operation
print(torch.add(a, b))  # inplace operation

# a = a + b

# All functions have underscore in the end
print(a.mul(b))
print(a)
print(a.mul_(b))  # This will reassign a to the result of the operation!!!!!!!!!
print(a)

# PART TWO - Tensor Operations

print("Part two")
a = torch.tensor([1., 2., 3.])
print(a)
print(b)
print(a * b)

# Dot product
print(a.dot(b))
a = torch.tensor([[0, 2, 4], [1, 3, 5]])
b = torch.tensor([[6, 7], [8, 9], [10, 11]])
print(a.shape)
print(b.shape)
print(a.mm(b))
print(a @ b)

x = torch.tensor([2., 3., 4., 5.])
print(x.norm())

print(x.numel())
print(len(x))
print(len(a))
print(a.numel())
"""

print("Exercise")

torch.manual_seed(42)
numpy.random.seed(42)

arr = np.random.randint(0, 5, 6)
print(arr)

x = torch.tensor(arr)
print(x)

x = x.type(torch.int64)
print(type(x))

x = x.reshape(3, 2)
print(x)

print(x[:, 1:])

print(x**2)

y = torch.randint(0,5,(2,3))
print(y)

print(x.mm(y))