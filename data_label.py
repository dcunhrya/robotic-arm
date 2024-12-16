import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

filename = 'datab/testb.csv'
df = pd.read_csv(filename)
emg_data = df['EMG Voltage (V)'].values
time = df['Time (s)'].values
gt = df['GT'].values

plt.plot(time, gt, 'r')
plt.show()
