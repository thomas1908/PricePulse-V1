<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
</div>
<hr>

# PricePulse-V1

PricePulse is an artificial intelligence model developed to predict cryptocurrency market movements using LSTM (Long Short-Term Memory) recurrent neural networks. The model leverages historical data, advanced technical indicators, and resampling techniques to make reliable predictions about market trends.

## Description

The PricePulse model analyzes cryptocurrency price time series and uses technical indicators to predict whether the price of a cryptocurrency will rise or fall in the near future. This project implements a bidirectional LSTM neural network to learn the complex and nonlinear relationships between these data points.

### Objectives:
- Predict the direction of cryptocurrency prices (up or down).
- Use popular technical indicators such as RSI, MACD, EMA, and others to enhance predictions.
- Provide a reliable tool for traders wishing to automate and improve their investment strategies.

## Features

- **Market Trend Prediction**: Predict whether the cryptocurrency price will increase or decrease based on historical data.
- **Use of Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands, etc.
- **Training on Historical Data**: Use cryptocurrency historical data to train the model.
- **Resampling for a Balanced Dataset**: Use SMOTE to address class imbalance issues.
- **Performance Evaluation**: Classification report, confusion matrix, ROC curve, and AUC to assess the model's results.

## Overview

### Model Architecture

#### Data Preprocessing:
- Extract technical indicators (RSI, MACD, ATR, etc.).
- Clean and normalize data.
- Create a binary target based on price movement.

#### LSTM Model:
- Bidirectional LSTM architecture to capture temporal dependencies in the data series.
- Dropout to prevent overfitting.

#### Training and Validation:
- Resample training data using SMOTE.
- Train the model with cross-validation and learning rate reduction.

#### Evaluation:
- Predict classes (price increase or decrease).
- Calculate performance metrics such as accuracy, confusion matrix, and AUC.

## Installation

### Prerequisites
Before you begin, you need to install the following libraries:

```bash
pip install -r requirements.txt
```

#### Main Libraries:
- **TensorFlow**: For training the LSTM model.
- **Pandas**: For data processing.
- **NumPy**: For numerical operations.
- **TA-Lib**: To compute technical indicators.
- **Scikit-learn**: For data management and evaluation tools.
- **Imbalanced-learn**: For SMOTE resampling.

### Clone the Repository
Clone this repository to start working with the project:

```bash
git clone https://github.com/thomas1908/PricePulse-V1.git
cd PricePulse
```

## Data Preparation

Download or generate your own cryptocurrency historical data (e.g., Bitcoin, Ethereum) in CSV format, including columns like `timestamp`, `open`, `close`, `high`, `low`, and `volume`.

Place your data in the `model/` folder.

### `historical_data.csv` File Format:
| timestamp            | open  | high  | low   | close | volume |
|----------------------|-------|-------|-------|-------|--------|
| 2025-01-01 00:00:00  | 30000 | 30500 | 29500 | 30050 | 1200   |

## Usage

### 1. Data Preparation
The `historical_data.csv` file should be formatted as follows:

```csv
timestamp, open, high, low, close, volume
2025-01-01 00:00:00, 30000, 30500, 29500, 30050, 1200
```

### 2. Running the Model
To train the model, run the main script `model_training.py`:

```bash
python model/model_training.py
```

This will execute the following steps:
- Load the data.
- Apply technical indicators.
- Train the LSTM model.
- Evaluate the model using metrics like accuracy, confusion matrix, and AUC.

### 3. Save and Load the Model
The trained model is saved in the `model/trading_model_nn.h5` file. You can also load this model to make predictions on new data:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/trading_model_nn.h5')

# Make predictions
predictions = model.predict(new_data)
```

## Model Evaluation

Once training is complete, the model is evaluated using test data. Key metrics include:

- **Accuracy**: The rate of correct predictions.
- **Classification Report**: Precision, recall, and F1-score.
- **Confusion Matrix**: Visualization of predictions vs. actual values.
- **ROC Curve and AUC**: Measures the model's quality.

## Contribute

We welcome contributions to this project! To contribute, please follow these steps:

1. Fork this repository.
2. Create a branch for your feature (`git checkout -b my-feature`).
3. Commit your changes (`git commit -am 'Added a feature'`).
4. Push to the branch (`git push origin my-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
