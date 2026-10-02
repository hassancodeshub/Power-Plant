# PowerPlant Output Prediction Using Artificial Neural Networks

A PyTorch-based Artificial Neural Network (ANN) regression project for predicting the electrical output of a power plant from environmental and operational measurements.

## 📌 Project Overview

This project uses an Artificial Neural Network to predict **Electrical Energy Output (PE)** from four input features:

* **AT** — Ambient Temperature
* **V** — Exhaust Vacuum
* **AP** — Ambient Pressure
* **RH** — Relative Humidity

The model is implemented using **PyTorch** and trained as a regression model using **Mean Squared Error (MSE)** loss and the **Adam optimizer**.

The dataset contains **9,568 observations**. The notebook uses an 80/20 train-test split, resulting in:

* **7,654 training samples**
* **1,914 test samples**

## 🎯 Objective

The objective of this project is to learn the relationship between the four input measurements and the plant's electrical energy output.

The prediction pipeline is:

```text
Ambient Temperature
        │
Exhaust Vacuum
        │
Ambient Pressure ──► ANN ──► Predicted Electrical Output
        │
Relative Humidity
```

## 📊 Dataset

The project uses:

```text
powerplant_data.csv
```

The dataset contains the following columns:

| Feature | Description              | Role   |
| ------- | ------------------------ | ------ |
| `AT`    | Ambient Temperature      | Input  |
| `V`     | Exhaust Vacuum           | Input  |
| `AP`    | Ambient Pressure         | Input  |
| `RH`    | Relative Humidity        | Input  |
| `PE`    | Electrical Energy Output | Target |

The notebook checks for missing values, and the recorded result shows **zero missing values in all five columns**.

## 🧠 Machine Learning Approach

The project follows these main steps:

1. Load the dataset using Pandas.
2. Separate input features from the target.
3. Split the data into training and testing sets.
4. Standardize the input features.
5. Convert the data into PyTorch tensors.
6. Create `TensorDataset` and `DataLoader` objects.
7. Define the ANN architecture.
8. Train the model using Adam and MSE loss.
9. Save the model state when the recorded test-loader loss improves.
10. Load the saved model.
11. Evaluate the model using MSE and R².
12. Compare predicted and actual values.

## 🔄 Data Preprocessing

### Train-Test Split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Therefore:

```text
Training data: 80%
Testing data:  20%
```

The fixed `random_state=42` makes the split reproducible.

### Feature Scaling

The input features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

## 🏗️ ANN Architecture

The neural network is implemented using PyTorch's `nn.Sequential`.

```text
Input: 4 features
        │
        ▼
Linear(4 → 6)
        │
      ReLU
        │
        ▼
Linear(6 → 6)
        │
      ReLU
        │
        ▼
Linear(6 → 1)
        │
        ▼
Electrical Output
```

### Architecture Details

| Layer          | Configuration    |
| -------------- | ---------------- |
| Input          | 4 features       |
| Hidden Layer 1 | 6 neurons + ReLU |
| Hidden Layer 2 | 6 neurons + ReLU |
| Output Layer   | 1 neuron         |
| Task           | Regression       |

The output layer does not use an activation function, allowing the network to produce a continuous numerical prediction.

## ⚙️ Training Configuration

The model uses:

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

Training configuration:

| Parameter        |              Value |
| ---------------- | -----------------: |
| Epochs           |                100 |
| Batch size       |                 32 |
| Optimizer        |               Adam |
| Loss function    | Mean Squared Error |
| Training samples |              7,654 |
| Test samples     |              1,914 |

The training data is shuffled through the `DataLoader`:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

The test loader is not shuffled.

## 📉 Training Progress

The recorded training run shows the loss decreasing substantially during the early epochs.

For example:

```text
Epoch 1:
Train Loss ≈ 160972.03
Test-loader Loss ≈ 134447.48

Epoch 20:
Train Loss ≈ 102.25
Test-loader Loss ≈ 74.99

Epoch 40:
Train Loss ≈ 21.16
Test-loader Loss ≈ 19.48

Epoch 100:
Train Loss ≈ 21.52
Test-loader Loss ≈ 18.99
```

The model therefore learned most of the relationship during the early stages of training, with the loss becoming relatively stable later in training.

## 📈 Evaluation Results

After loading the saved model, the notebook reports:

```text
Training MSE : 20.670753479003906
Testing MSE  : 19.000024795532227
R² Score     : 0.9335998115810005
```

### Results Summary

| Metric       |  Result |
| ------------ | ------: |
| Training MSE | 20.6708 |
| Testing MSE  | 19.0000 |
| Test R²      |  0.9336 |

The recorded **R² score of approximately 0.934** indicates that the model explains a substantial portion of the variation in the target values within the test set used in the notebook.

## 🔍 Prediction Comparison

The notebook also creates a table comparing model predictions with the corresponding actual values.

Example predictions from the recorded output:

| Predicted | Actual |
| --------: | -----: |
|    435.92 | 433.27 |
|    436.92 | 438.16 |
|    461.30 | 458.42 |
|    475.92 | 480.82 |
|    435.68 | 441.41 |

The complete comparison contains **1,914 test observations**.

## 💾 Model Saving

During training, the notebook attempts to save the model whenever the test-loader loss improves:

```python
torch.save(model.state_dict(), "best_model.pt")
```

The saved state is subsequently loaded using:

```python
model.load_state_dict(
    torch.load("best_model.pt")
)
```

The recorded notebook confirms that the saved state was successfully loaded.

## 🛠️ Technologies Used

* **Python 3.13.5**
* **PyTorch**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**

### Main Libraries

```python
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
```

## 📁 Project Structure

A simple project structure can be organized as:

```text
PowerPlant-Output-Prediction-ANN/
│
├── ann.ipynb
├── powerplant_data.csv
├── best_model.pt
└── README.md
```

Where:

* `ann.ipynb` — Complete data preprocessing, ANN training, and evaluation workflow
* `powerplant_data.csv` — Dataset
* `best_model.pt` — Saved PyTorch model parameters
* `README.md` — Project documentation

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd PowerPlant-Output-Prediction-ANN
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib torch
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
ann.ipynb
```

### 4. Run the notebook

Run the cells sequentially to:

* Load the dataset
* Preprocess the data
* Train the ANN
* Save the model
* Load the saved model
* Evaluate predictions
* Compare predicted and actual output

## 🧪 Model Workflow

```text
                 ┌──────────────────────┐
                 │ powerplant_data.csv  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Separate X and y     │
                 │ X = AT,V,AP,RH       │
                 │ y = PE               │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Train/Test Split     │
                 │ 80% / 20%            │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ StandardScaler       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ PyTorch Tensors      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Artificial Neural    │
                 │ Network              │
                 │                      │
                 │ 4 → 6 → 6 → 1       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Prediction           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ MSE + R² Evaluation  │
                 └──────────────────────┘
```

## ⚠️ Notes on the Current Notebook

The notebook stores the loss calculated on `test_loader` in variables named `val_losses` and `epoch_val_loss`. In the current implementation, the **test set is therefore being used during training to monitor loss and select the saved model state**. It is not a separate validation set.

For a stricter machine-learning evaluation workflow, the dataset could instead be divided into:

```text
Training Set
     ↓
Model Training

Validation Set
     ↓
Model Selection / Hyperparameter Tuning

Test Set
     ↓
Final Evaluation
```

This would keep the test set completely independent until the final evaluation.

Also, the current checkpoint condition contains:

```python
if epoch_val_loss < best_val_loss:
    best_loss = epoch_val_loss
```

The variable initialized as `best_val_loss` is not updated inside the condition. A corrected version would be:

```python
if epoch_val_loss < best_val_loss:
    best_val_loss = epoch_val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

These notes describe the current notebook implementation and are not changes to the reported results.

## 📌 Results at a Glance

```text
Project:        PowerPlant Output Prediction Using Artificial Neural Networks
Task:           Regression
Framework:      PyTorch
Input Features: 4
Hidden Layers:  2
Neurons:        6 → 6
Epochs:         100
Batch Size:     32

Training MSE:   20.6708
Testing MSE:    19.0000
Test R²:        0.9336
```
---

**Project Title:**

# PowerPlant Output Prediction Using Artificial Neural Networks
