import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/synthetic_test_data.csv')

# Column names based on the dataset
time_column = 'Time'  # Assuming you have a time column, if not, you might need to create it or modify the plots
ax_column = 'Ax'
ay_column = 'Ay'
az_column = 'Az'
gx_column = 'Gx'
gy_column = 'Gy'
gz_column = 'Gz'

# If you don't have a time column, you might need to create an index for plotting
if time_column not in data.columns:
    time = range(len(data))  # Using row indices as time if no time column is present
else:
    time = data[time_column]

# Extract data
ax = data[ax_column]
ay = data[ay_column]
az = data[az_column]
gx = data[gx_column]
gy = data[gy_column]
gz = data[gz_column]

# Create subplots for each sensor
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Plot accelerometer data
axs[0].plot(time, ax, label='Ax', color='r')
axs[0].plot(time, ay, label='Ay', color='g')
axs[0].plot(time, az, label='Az', color='b')
axs[0].set_ylabel('Acceleration (m/s²)')
axs[0].legend()
axs[0].set_title('Accelerometer Data')

# Plot gyroscope data
axs[1].plot(time, gx, label='Gx', color='r')
axs[1].plot(time, gy, label='Gy', color='g')
axs[1].plot(time, gz, label='Gz', color='b')
axs[1].set_ylabel('Gyroscope (rad/s)')
axs[1].legend()
axs[1].set_title('Gyroscope Data')

# Plot combined data
axs[2].plot(time, ax, label='Ax', color='r', alpha=0.5)
axs[2].plot(time, ay, label='Ay', color='g', alpha=0.5)
axs[2].plot(time, az, label='Az', color='b', alpha=0.5)
axs[2].plot(time, gx, label='Gx', color='r', linestyle='--')
axs[2].plot(time, gy, label='Gy', color='g', linestyle='--')
axs[2].plot(time, gz, label='Gz', color='b', linestyle='--')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Combined Data')
axs[2].legend()
axs[2].set_title('Combined Accelerometer and Gyroscope Data')

# Display the plots
plt.tight_layout()
plt.show()
