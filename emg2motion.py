import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/m1.csv')

# Extract EMG voltage and corresponding time data
emg_data = torch.tensor(df['EMG Voltage (V)'].values.reshape(-1, 1)).float()
time_data = torch.tensor(df['Time (s)'].values.reshape(-1, 1)).float()

# Normalize EMG voltage to the range [0, 115]
min_voltage = emg_data.min()
max_voltage = emg_data.max()
# print(min_voltage)
print(max_voltage)
num_bins = 10
bin_size = 115 / num_bins
bin_edges = np.arange(0, 115 + bin_size, bin_size)

def bin_data(emg_data):
    emg_binned = []
    min_voltage = emg_data.min()
    max_voltage = emg_data.max()
    for point in emg_data:
        degree_data = 115 * (point - min_voltage) / (max_voltage - min_voltage)
        for i in range(len(bin_edges)):
            if degree_data < bin_edges[i] and i > 0:
                degree_data = bin_edges[i-1]
                break
            elif degree_data < bin_edges[i] and i == 0:
                degree_data = bin_edges[0]
                break
        emg_binned.append(degree_data)
    return torch.tensor([[value] for value in emg_binned]).float()

binned_data = bin_data(emg_data)
# print(binned_data.min())
# print(emg_data)

# Assign each angle to a bin and use the midpoint of the bin as the label
# midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate midpoints of bins
# binned_indices = np.digitize(degree_data, bin_edges) - 1  # Get bin indices
# binned_indices = np.clip(binned_indices, 0, num_bins - 1)  # Ensure indices are within bounds
# binned_degree_data = torch.tensor(midpoints[binned_indices]).float().reshape(-1, 1)

# Split the normalized data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(emg_data, binned_data, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Build the neural network
# class ScaledLinearActivation(nn.Module):
#     def __init__(self, scale=1):
#         super(ScaledLinearActivation, self).__init__()
#         self.scale = scale

#     def forward(self, x):
#         return torch.relu(x) * self.scale

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.scaled_linear = ScaledLinearActivation(scale=115)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))*115
        # x = self.scaled_linear(self.fc3(x))
        return x

model = Net()

# Compile the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print average loss every 10 batches
        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch + 1}, Batch {i}, Average Loss: {running_loss / i:.4f}')

    # Print loss for each epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')
    if running_loss/len(train_loader) < 0.0005:
        print('Stopping at epoch {} due to low error'.format(epoch))
        break

# Evaluate the model
total_loss = 0
total_mae = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        mae = torch.mean(torch.abs(outputs - targets))
        total_loss += loss.item()
        total_mae += mae.item()

print(f'Loss: {total_loss / len(test_loader)}, Mean Absolute Error: {total_mae / len(test_loader)}')

# new_df = pd.read_csv('data/new_emg_data.csv')
new_df = df
new_emg_data = torch.tensor(new_df['EMG Voltage (V)'].values.reshape(-1, 1)).float()
normalized_new_emg_data = bin_data(new_emg_data)

with torch.no_grad():
    predicted_angles = model(normalized_new_emg_data)
    predicted_angles = predicted_angles.flatten().numpy()

# Plot Voltage vs. Angle
plt.plot(new_emg_data.numpy(), predicted_angles, 'o', label='Predicted Angles')
plt.xlabel('EMG Voltage (V)')
plt.ylabel('Predicted Angle (degrees)')
plt.title('EMG Voltage vs. Predicted Angle')
plt.legend()
plt.show()
