import pandas as pd
import numpy as np
import os
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'], encoding='ISO-8859-1')
    return df


def add_new_indicators(df):
    # Average volume over a period
    df['Volume_SMA'] = talib.SMA(df['volume'], timeperiod=20)
    
    # On-Balance Volume (OBV)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    
    # Chande Momentum Oscillator (CMO)
    df['CMO'] = talib.CMO(df['close'], timeperiod=14)
    
    # Stochastic RSI
    df['Stoch_RSI'], _ = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
    
    # Parabolic SAR
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    
    # Average True Range (ATR)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df
    
def add_indicators(df):
    # Basic indicators
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['SMA'] = talib.SMA(df['close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['close'], timeperiod=30)
    df['Bollinger_upper'], df['Bollinger_middle'], df['Bollinger_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    df = add_new_indicators(df)

    return df

# Data preprocessing (handling missing values, normalization)
def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df = df.drop(columns=['timestamp'])
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def resample_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def create_lstm_model(input_shape):
    from tensorflow.keras.layers import Bidirectional

    model = Sequential()
    model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    y_prob = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")

def save_model(model, file_path):
    model.save(file_path)  # Save the model in Keras format (.h5)
    print("Model saved successfully!")

def load_model(file_path):
    from tensorflow.keras.models import load_model
    return load_model(file_path)

def main():
    current_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(current_dir, 'historical_data.csv')
    
    df = load_data(data_file_path)
    df = add_indicators(df)
    
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)
    
    model = create_lstm_model((X_train_resampled.shape[1], 1))
    
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                     patience=3,
                                     verbose=1, 
                                     factor=0.5,
                                     min_lr=0.0001)
    
    model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=128, validation_split=0.2, callbacks=[lr_reduction])
    
    evaluate_model(model, X_test, y_test)
    
    model_file_path = os.path.join(current_dir, 'trading_model_nn.h5')
    save_model(model, model_file_path)

if __name__ == "__main__":
    main()
