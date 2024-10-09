

# Earthquake Detection using Phyphox and Streamlit

## Project Overview

This project demonstrates the use of **Phyphox**, an open-source sensor data application, to detect earthquake-like vibrations using live accelerometer data. The data is fetched from a Phyphox server running on a mobile device, and a **Streamlit** app dynamically visualizes the sensor readings. If the detected acceleration magnitude exceeds a certain threshold for a defined duration, the app predicts the occurrence of an earthquake.

## Features
- **Live Data Fetching**: Collects real-time sensor data (accelerometer X, Y, Z axes) from the Phyphox app.
- **Magnitude Calculation**: Computes the total acceleration magnitude from the X, Y, Z axes.
- **Earthquake Prediction**: Detects a potential earthquake if the magnitude exceeds a threshold for more than 5 seconds.
- **Dynamic Visualization**: Displays live data updates for sensor readings and magnitude.
- **Customizable Settings**: You can adjust the earthquake magnitude threshold and detection duration.
- **Reference Information**: The app also includes a reference table for earthquake classifications based on the Richter scale and PGA (Peak Ground Acceleration).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [How to Run](#how-to-run)
4. [Code Explanation](#code-explanation)
   - [Main Python Script](#main-python-script)
   - [Earthquake Detection Logic](#earthquake-detection-logic)
   - [Dynamic Data Visualization](#dynamic-data-visualization)
5. [Features in Detail](#features-in-detail)
6. [Data Flow: Mobile Phone to Laptop](#data-flow)
   - [How Phyphox Works](#how-phyphox-works)
   - [Communication with Laptop via Server](#communication-with-laptop-via-server)

---

## Prerequisites

Before running the project, ensure you have the following installed:

- **Phyphox App**: Download and install the Phyphox app on your mobile phone from [Google Play](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox&hl=en_IN&pli=1) or [Apple App Store](https://apps.apple.com/app/phyphox/id1121278712).
- **Python**: Ensure you have Python installed on your laptop.
- **Streamlit**: Install Streamlit by running the command `pip install streamlit`.

---

## Installation

1. Clone or download the repository to your local machine.
2. Install the necessary Python libraries:
   ```bash
   pip install requests streamlit
   ```

---

## How to Run

1. **Open the Phyphox app** on your mobile device.
2. In the app, select **Remote Access** from the menu and start the server. A URL will be displayed (e.g., `http://192.168.x.x:8080`).
3. **Copy the URL** and update it in the Python script (`url` variable).
4. Run the Streamlit application:
   ```bash
   streamlit run earthquake_detection.py
   ```
5. Visit the URL provided by Streamlit (usually `http://localhost:8501`) in your browser to view the live earthquake detection interface.

---

## Code Explanation

### Main Python Script

1. **Sensor Data Fetching**: 
   - The Phyphox app is set up to act as a server on your mobile device, and it streams real-time sensor data.
   - The script makes HTTP GET requests to the Phyphox server's URL to retrieve the latest accelerometer readings (`accX`, `accY`, `accZ`).

2. **Magnitude Calculation**:
   - The total magnitude of the acceleration vector is calculated using the formula:
     \[
     \text{Magnitude} = \sqrt{{\text{accX}}^2 + {\text{accY}}^2 + {\text{accZ}}^2}
     \]
   - This magnitude helps determine whether an earthquake is happening based on the acceleration.

3. **Earthquake Detection**:
   - If the calculated magnitude exceeds a predefined threshold (`10g` in this case) for a continuous period of 5 seconds, the app detects an earthquake.
   - If the magnitude stays below the threshold, the timer resets.

### Earthquake Detection Logic

```python
def calculate_magnitude(accX, accY, accZ):
    return math.sqrt(accX**2 + accY**2 + accZ**2)
```
- This function computes the magnitude of the acceleration vector using X, Y, Z axis readings.
- The threshold is set to `10g`, but it can be adjusted according to the desired sensitivity.

### Dynamic Data Visualization

- The sensor readings and earthquake prediction are dynamically displayed on the Streamlit interface using placeholders.
- Sensor data (X, Y, Z axes) and the calculated magnitude are updated in real-time every second.

---

## Features in Detail

1. **Richter Scale Table**: The app displays a reference table that categorizes different earthquake magnitudes and their corresponding potential damages based on the Richter scale and Peak Ground Acceleration (PGA).
   
2. **Earthquake Threshold Settings**: 
   - Users can adjust the threshold for magnitude detection, allowing for sensitivity customization based on the region and device capabilities.

3. **Mobile to Laptop Communication**: 
   - The mobile phone's accelerometer data is streamed in real-time to the Streamlit app running on the laptop.

---

## Data Flow: Mobile Phone to Laptop

### How Phyphox Works

- **Phyphox (Physical Phone Experiments)** is a free mobile app that allows users to access and analyze data from the sensors on their phones.
- It provides a **remote access feature**, allowing the phone to act as a server, sharing sensor data over the local network.

### Communication with Laptop via Server

1. **Setting up the Phyphox App**:
   - Open the Phyphox app on your mobile device.
   - Select the **Accelerometer** experiment from the available experiments.
   - Enable **Remote Access** from the app menu.
   - A URL, typically in the format `http://192.168.x.x:8080`, will be generated. This is the address of the server running on the mobile device, where the accelerometer data can be accessed.
  
2. **Fetching Data on Laptop**:
   - In the Python script, you need to set the `url` variable to match the URL generated by the Phyphox app.
   - The script sends an HTTP GET request to this URL every second to retrieve the accelerometer data (`accX`, `accY`, and `accZ`).
   - This data is fetched in JSON format and parsed to extract individual axis readings, which are then used to calculate the total magnitude.

### Communication Flow:

| Step | Action |
|------|--------|
| **1** | Phyphox app on the mobile phone starts a server. |
| **2** | The phone’s accelerometer data is continuously collected and made available through the server’s URL. |
| **3** | The laptop, running the Streamlit app, sends a GET request to this URL at regular intervals (every 1 second). |
| **4** | The server responds with the current accelerometer data in JSON format. |
| **5** | The laptop processes this data, calculates the magnitude, and checks for earthquake-like movements. |
| **6** | Results are displayed on the Streamlit interface in real-time. |

---

## Conclusion

This project integrates the Phyphox app with a Streamlit-based Python application to detect and visualize earthquakes in real-time. Using live accelerometer data, the app calculates the total magnitude of the device's movement and alerts users if an earthquake-like event is detected.
