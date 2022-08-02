import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To know more about embeddings
# https://www.quora.com/What-does-PyTorch-Embedding-do

# Read the data
df = pd.read_csv('../Data/NYCTaxiFares.csv')

# print(df.head())
# print(df['fare_amount'].describe())
# print(df.info())

def haversine(df, lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles

    # Convert decimal degrees to radians
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[lon2] - df[lon1])

    # haversine formula
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r  # In km


# Feature Engineering
# -------------------

# 1. Add a new column 'dist_km' to the dataframe
# Haversine formula to calculate distance between two points with latitude and longitude
df['dist_km'] = haversine(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

# 2. Add new columns for hour, AM/PM and day of week
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] >= 12, 'pm', 'am')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")

# 3. Classify which are categorical and which are continuous. Categorical will need to go through the embedding layer
cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']
y_col = ['fare_class']  # For Classification

# print(df.dtypes)

# 4. Convert to category, that is from categorical to numerical code
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
# df['Hour'].head()

# We can access the category names with Series.cat.categories or just the codes with Series.cat.codes.
# df['AMorPM'] = df['AMorPM'].cat.categories
# df['AMorPM'] = df['AMorPM'].cat.codes
# df['AMorPM'] = df['AMorPM'].cat.codes.values # To convert to numpy array

# hr = df['Hour'].cat.codes.values
# ampm = df['AMorPM'].cat.codes.values
# wkdy = df['Weekday'].cat.codes.values
# cats = np.stack([hr, ampm, wkdy], axis=1) # As columns

# 5. Stack categorical values into one tensor
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats = torch.tensor(cats, dtype=torch.int64)

# 6. Stack continuous values into one tensor
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float32)

# 7. Stack label values into one tensor
# y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1)
y = torch.tensor(df[y_col].values).flatten() # For Classification

print(cats.shape)
print(conts.shape)
print(y.shape)

# 8. For categories, instead of using one-hot encoding, we use embeddings.
# We can use the same embedding layer for all the categorical variables.
# The size of the dense vector is a hyperparameter.
cat_szs = [len(df[col].cat.categories) for col in cat_cols]     # [24, 2, 7]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]    # [(24,12), (2,1), (7,4)]

# ------------------------------------------------------------

# Test
# a. Extract the first four rows of the categorical data
catz = cats[:4]  # [[4, 0, 1], [...], [...], [...]]

# b. List of embedding layers
selfembeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
print(f'Embedding Layer {selfembeds}')   # (0) Embedding(24,12) (1) Embedding(2,1) (2) Embedding(7,4)

# c. Forward pass in the test subset tensor (later done for the entire cats tensor)
# You take the embedding layer and passing it to the categorical data as single row data
embeddingz = []
print(f'catz: {catz}')
for i,emb in enumerate(selfembeds):
    print(f'Embedding Layer {i} | {emb} - we pass {catz[:, i]}')
    embeddingz.append(emb(catz[:, i]))

# d. Concatenate the embedding layers
z = torch.cat(embeddingz, dim=1)

# e. Create dropout layer
selfembdrop = nn.Dropout(0.1)
z = selfembdrop(z)

# ------------------------------------------------------------

# 9. Create the Tabular Model
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):

        # emb_szs: embedding sizes (hyperparameter)
        # n_cont: number of continuous features
        # out_sz: number of output features
        # layers: list with neurons for each layer: [n1, n2, n3, ...]
        # p: dropout probability

        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs]) # nf: embedding size
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)   # Normalization layer

        layerlist = []  # To store layers
        n_embs = sum([nf for ni, nf in emb_szs])    # Number of embedding layers (entries of all dense vectors)
        n_in = n_cont + n_embs  # Sum number of embeddings and continuous features

        # Create layers
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))    # Number of inputs: Sum of embedded and continuous features
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i

        # Create output layer
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Mount all layers in a sequential model
        # If you have a model with lots of layers, you can create a list first and then
        # use the * operator to expand the list into positional arguments (*args, **kwargs)
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):

        # x_cat: categorical features
        # x_cont: continuous features

        # Pass the data through the embedding layers
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))

        # Concatenate the embedding layers
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        # Normalize continuous features
        x_cont = self.bn_cont(x_cont)

        # Prepare all data together
        x = torch.cat([x, x_cont], dim=1)
        x = self.layers(x)
        return x


torch.manual_seed(33)

#model = TabularModel(emb_szs, conts.shape[1], 1, [256, 128], 0.4)
model = TabularModel(emb_szs, conts.shape[1], 2, [256, 128], 0.4) # For classification
# Output size = n° classes (2 binary)
print(model)

#criterion = nn.MSELoss() # np.sqrt(MSE) --> RMSE
criterion = nn.CrossEntropyLoss() # For classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(batch_size*0.2)

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# Data shuffled already
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]

cont_train = conts[:batch_size-test_size]
cont_test = conts[batch_size-test_size:batch_size]

y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

# len(cat_train)
# len(cat_test)

print(cat_train.shape)
print(cont_train.shape)
print(cat_test.shape)
print(cont_test.shape)

import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, cont_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss)

    if(i % 5 == 0):
        print(f'Epoch {i} | Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

duration = time.time() - start_time
print(f'Duration: {duration/60} minutes')
# In laptop: 1.01 minutes

# New list of losses deatached from the graph
losses = [loss.detach().numpy() for loss in losses]
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    y_val = model(cat_test, cont_test)
    # loss = torch.sqrt(criterion(y_val, y_test))
    # print(f'Validation Loss: {loss}')
    loss = criterion(y_val, y_test) # For Classification
    print(f'CE Loss: {loss}')

# for i in range(10):
#     print(f'Prediction: {y_val[i].item():8.2f} | Actual: {y_test[i].item():8.2f}')

# For Classification
rows = 50
correct = 0
print(f'{"MODEL OUTPUT":26} ARGMAX  Y_TEST')
for i in range(rows):
    print(f'{str(y_val[i]):26} {y_val[i].argmax():^7}{y_test[i]:^7}')
    if y_val[i].argmax().item() == y_test[i]:
        correct += 1
print(f'\n{correct} out of {rows} = {100*correct/rows:.2f}% correct')

#torch.save(model.state_dict(), 'MyTaxiModel.pt')
torch.save(model.state_dict(), 'MyTaxiModelClassification.pt') # For Classification

print("Done")
