# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("sensor-fault-detection.csv", sep=';')

data.columns = data.columns.str.strip().str.lower()
print("Columns:", data.columns)

data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

data = data.dropna(subset=['timestamp'])

if 'sensorid' in data.columns:
    sensor_id = 1
    data = data[data['sensorid'] == sensor_id]
    print(f"Filtered for sensor ID: {sensor_id}")

data.set_index('timestamp', inplace=True)
data = data.sort_index()

data = data[~data.index.duplicated(keep='first')]

data = data.resample('D').mean().interpolate()

col = 'value'
print("Using column:", col)

data['value_diff'] = data[col] - data[col].shift(1)

result = seasonal_decompose(data[col].dropna(), model='additive', period=30)
data['value_sea_diff'] = result.resid

min_val = data[col].min()
if min_val <= 0:
    shift_val = abs(min_val) + 1.0
    data['value_log'] = np.log(data[col] + shift_val)
    print(f"Values shifted by {shift_val} to enable log transformation.")
else:
    data['value_log'] = np.log(data[col])

data['value_log_diff'] = data['value_log'] - data['value_log'].shift(1)

result2 = seasonal_decompose(data['value_log_diff'].dropna(), model='additive', period=30)
data['value_log_seasonal_diff'] = result2.resid

plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data[col])
plt.title('Original Sensor Value')

plt.subplot(6, 1, 2)
plt.plot(data['value_diff'])
plt.title('Regular Differencing')

plt.subplot(6, 1, 3)
plt.plot(data['value_sea_diff'])
plt.title('Seasonal Adjustment (Residuals)')

plt.subplot(6, 1, 4)
plt.plot(data['value_log'])
plt.title('Log Transformation')

plt.subplot(6, 1, 5)
plt.plot(data['value_log_diff'])
plt.title('Log + Differencing')

plt.subplot(6, 1, 6)
plt.plot(data['value_log_seasonal_diff'])
plt.title('Final Stationary (Log + Seasonal Diff)')

plt.tight_layout()
plt.show()
```

### OUTPUT:

<img width="700" height="77" alt="image" src="https://github.com/user-attachments/assets/e862c0fd-2245-4d4a-aee6-f6e2def6a017" />

REGULAR DIFFERENCING:

<img width="1227" height="210" alt="image" src="https://github.com/user-attachments/assets/76a53a53-67bc-47c5-ae8e-c96e9443eb8b" />

SEASONAL ADJUSTMENT:

<img width="1257" height="202" alt="image" src="https://github.com/user-attachments/assets/3a4ea2fa-26f9-4398-ad52-7984d7783f5e" />

LOG TRANSFORMATION:

<img width="1211" height="205" alt="image" src="https://github.com/user-attachments/assets/b41c4a0c-3fff-4760-8256-f8e2e09468c2" />

### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
