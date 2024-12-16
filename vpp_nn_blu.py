import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

def preprocess_data(filenames, window_size):
    all_vpp_data = []
    all_degree_data = []

    for filename, degree in filenames:
        df = pd.read_csv(filename)
        emg_data = df['EMG Voltage (V)'].values
        # print("Degree {} size {}".format(degree, len(emg_data)))

        # Calculate Vpp
        vpp_data = [emg_data[i:i+window_size].max() - emg_data[i:i+window_size].min() for i in range(0, len(emg_data), window_size)]
        vpp_data = np.array(vpp_data)
        if degree == 0:
            vpp_data = vpp_data[:len(vpp_data)//2]
        # Plot raw data
        # plt.plot(vpp_data)
        # plt.title(filename)
        # plt.show()

        all_vpp_data.extend(vpp_data)
        all_degree_data.extend([degree] * len(vpp_data))

    return torch.tensor(all_vpp_data).float().view(-1, 1), torch.tensor(all_degree_data).float().view(-1, 1)

# List of filenames and their corresponding degrees
# filenames = [
#     ('data/0_degree.csv', 0),
#     ('data/45_degree.csv', 45),
#     ('data/90_degree.csv', 90),
#     ('data/135_degree.csv', 135),
#     ('data/180_degree.csv', 180)
# ]
filenames = [
    ('datab/0.csv', 0),
    ('datab/45.csv', 45),
    ('datab/90.csv', 90),
    ('datab/135.csv', 135),
    ('data/0_degree.csv', 180) # old set for no flexion
]

window_size = 1000 
X, y = preprocess_data(filenames, window_size)

torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Build the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.clamp(x, min=0, max=180) 
        return x
    
early_stopping = EarlyStopping(patience=30, min_delta=0.001)
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4) # 2e-4
criterion = nn.MSELoss()

for epoch in range(4000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train) 
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    early_stopping(loss.item())
    if early_stopping.early_stop:
        print("Early stopping triggered at epoch:", epoch)
        break

# Evaluate the model
with torch.no_grad():
    predictions = model(X_test)
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Degrees')
    plt.ylabel('Predicted Degrees')
    plt.title('Actual vs Predicted Degrees of Flexion')
    plt.show()

def predict_degrees_from_csv(filename, window_size, model):
    df = pd.read_csv(filename)
    new_emg_data = df['EMG Voltage (V)'].values

    new_vpp_data = [new_emg_data[i:i+window_size].max() - new_emg_data[i:i+window_size].min() for i in range(0, len(new_emg_data), window_size)]
    new_vpp_data = torch.tensor(new_vpp_data).float().view(-1, 1)

    with torch.no_grad():
        predicted_degrees = model(new_vpp_data)
        return predicted_degrees.numpy().flatten()

test2_csv = 'datab/testb.csv' 
predicted_degrees = predict_degrees_from_csv(test2_csv, window_size, model)


filename = '/Users/dcunhrya/Documents/MEDesign/EMG_nn/datab/testb.csv'
df = pd.read_csv(filename)
gt = df['GT'].values

plt.plot(gt, 'r', label='Ground Truth')
plt.plot(predicted_degrees, label='Neural Network')
plt.legend()
plt.xlabel('Time Window')
plt.ylabel('Predicted Degree of Flexion')
plt.title('Predicted Degrees of Flexion Over Time')
plt.show()

save_dict = {"nn_predict": predicted_degrees, "gt": gt}

# with open('ryan_nn.pkl','wb') as f:
#     pkl.dump(save_dict, f)

# exit()

gt_time = np.linspace(0, len(gt) - 1, num=len(gt))  # Original time points for gt
predicted_time = np.linspace(0, len(gt) - 1, num=len(predicted_degrees))  # New time points for interpolation
interpolator = interp1d(gt_time, gt, kind='linear')  # Linear interpolator
gt_resampled = interpolator(predicted_time)  # Interpolated gt

# Calculate residuals using the resampled gt
# residuals = gt_resampled - predicted_degrees
# plt.figure(figsize=(10, 5))
# plt.plot(residuals, label='Residuals')
# plt.xlabel('Time Window')
# plt.ylabel('Residual')
# plt.title('Residuals of Predicted Degrees of Flexion')
# plt.axhline(y=0, color='black', linestyle='--')
# plt.legend()
# plt.show()

# Calculate R-squared value
# r_squared = r2_score(gt, predicted_degrees)
# print(f'R-squared value: {r_squared}')