

# Earthquake Detection System 🌍🚨

![Earthquake](img.jpg) 


## Welcome! 👋

Welcome to the Earthquake Detection System project! 🎉 This initiative aims to leverage machine learning to detect earthquakes using data from gyroscopes and accelerometers. Whether you're a researcher, developer, or just curious about the intersection of technology and natural disaster detection, this project is for you. Dive in to explore how we've built and fine-tuned a system to keep our communities safe from the tremors of Earth!

## What’s Inside? 🧩

### 1. Data Generation 🛠️

We’ve crafted a dataset that's both realistic and synthetic to train and test our models. Here’s what you’ll find:
- **Accelerometer Data (Ax, Ay, Az)**: Tracks acceleration in the X, Y, and Z axes.
- **Gyroscope Data (Gx, Gy, Gz)**: Measures rotational changes in the X, Y, and Z axes.
- **Labels**: Simple binary indicators - `0` for normal conditions and `1` for earthquakes.

#### How We Generate Data
- **Normal Conditions**: Smooth, steady data with minimal noise.
- **Earthquake Conditions**: Features significant spikes and variations to mimic real earthquakes.



### 2. Model Training 🤖

We’ve experimented with several powerful models to find the best fit for our task:

#### Random Forest 🌳
- **What It Is**: A collection of decision trees working together to make predictions.
- **Why It’s Great**: Handles diverse data well and avoids overfitting.

#### XGBoost 🏆
- **What It Is**: An advanced boosting technique that’s both fast and flexible.
- **Why It’s Great**: Delivers exceptional performance and scales effortlessly.

#### Support Vector Machine (SVM) 🚀
- **What It Is**: Finds the best boundary between different classes.
- **Why It’s Great**: Perfect for high-dimensional data and complex decision boundaries.

#### Long Short-Term Memory (LSTM) ⏳
- **What It Is**: A type of neural network designed to understand sequences.
- **Why It’s Great**: Excellent for capturing patterns over time, ideal for time-series data.



### 3. Evaluating Our Models 📊

We don’t just train models - we thoroughly evaluate them using:
- **Accuracy**: How often our model gets it right.
- **Precision**: How many of the predicted positives are truly positive.
- **Recall**: How many of the actual positives our model managed to catch.
- **F1 Score**: A balanced measure combining precision and recall.

### 4. Ensuring Robustness 🔍

We use cross-validation to make sure our models are reliable:
- **K-Fold Cross-Validation**: Splits data into multiple folds to ensure robust training.
- **Stratified K-Fold Cross-Validation**: Maintains class balance in each fold for more accurate results.

### 5. Visualizing Performance 📈

See how well our models perform through visualizations:
- **Training and Validation Accuracy**: Tracks accuracy improvements over epochs.
- **Training and Validation Loss**: Shows how loss decreases during training.



## Flowchart

Here’s a simplified flowchart of our process:

1. **Data Generation**
   - Create synthetic and realistic sensor data for both normal and earthquake conditions.

2. **Data Preprocessing**
   - Scale features and split data into training and test sets.

3. **Model Training**
   - Train various models (Random Forest, XGBoost, SVM, LSTM) using the training data.

4. **Model Evaluation**
   - Evaluate models on test data, focusing on key metrics like Accuracy, Precision, Recall, and F1 Score.

5. **Cross-Validation**
   - Apply K-Fold and Stratified K-Fold cross-validation to ensure our models are robust and generalizable.

6. **Visualization**
   - Plot training and validation metrics to monitor and understand model performance.

## Getting Started 🚀

### Installation 🛠️

1. **Clone the Repository**

    ```bash
    git clone https://github.com/adityajhakumar/earthquake-detection.git
    cd earthquake-detection
    ```

2. **Install Dependencies**

    Set up a virtual environment and install required packages:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Requirements**

    Our `requirements.txt` includes:
    
    ```plaintext
    numpy
    pandas
    scikit-learn
    keras
    matplotlib
    ```

### Usage

- **Generate Data**: Run `generate_data.py` to create your dataset:

    ```bash
    python generate_data.py
    ```

- **Train and Evaluate Models**: Use `train_evaluate_models.py`:

    ```bash
    python train_evaluate_models.py
    ```

- **Cross-Validation**: Perform cross-validation with `cross_validation.py`:

    ```bash
    python cross_validation.py
    ```

## Evaluation Metrics

We use the following metrics to evaluate our models:
- **Accuracy**: Measures overall correctness.
- **Precision**: Evaluates the accuracy of positive predictions.
- **Recall**: Assesses how well positive cases are detected.
- **F1 Score**: Balances precision and recall.
- ![image](https://github.com/user-attachments/assets/d52151eb-e298-4183-92eb-36cc6383e90d)
- <img width="370" alt="image" src="https://github.com/user-attachments/assets/11a93ae7-fdc0-4ef0-8c03-50f5a2cbf8c8">



## License 📝

This project is licensed under the Apache License 2.0. Check out the [LICENSE](LICENSE) file for details.

## Contact 📧

Have questions or feedback? Feel free to reach out to me, [Aditya Kumar Jha](mailto:your.email@example.com). I’m here to help!

---

Feel free to update any placeholders or links as needed. This version of the README provides a friendly and engaging overview of the project while ensuring that users have all the necessary information to get started.
