import pandas as pd
import numpy as np
import os
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE  # Pour rééchantillonner les classes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Charger les données
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'], encoding='ISO-8859-1')
    return df

# Ajout de nouveaux indicateurs techniques
def add_new_indicators(df):
    # Volume moyen sur une période
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

# Ajout des indicateurs techniques existants et nouveaux
def add_indicators(df):
    # Indicateurs de base
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['SMA'] = talib.SMA(df['close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['close'], timeperiod=30)
    df['Bollinger_upper'], df['Bollinger_middle'], df['Bollinger_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Nouveaux indicateurs ajoutés
    df = add_new_indicators(df)

    return df

# Prétraitement des données (gestion des valeurs manquantes, normalisation)
def preprocess_data(df):
    # Imputation des valeurs manquantes
    df.fillna(df.mean(), inplace=True)
    
    # Créer une colonne cible (prédire si le prix augmentera ou non)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Supprimer la colonne timestamp et toute colonne inutile
    df = df.drop(columns=['timestamp'])
    
    # Séparer les caractéristiques et la cible
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Rééchantillonnage des classes pour résoudre le déséquilibre
def resample_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

# Création du modèle LSTM
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

# Évaluer le modèle
def evaluate_model(model, X_test, y_test):
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en classes

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    # Rapport de classification
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculer la courbe ROC et l'AUC
    y_prob = model.predict(X_test)  # Récupérer les probabilités
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")

# Sauvegarder le modèle
def save_model(model, file_path):
    model.save(file_path)  # Sauvegarder le modèle au format Keras (.h5)
    print("Modèle sauvegardé avec succès !")

# Charger le modèle
def load_model(file_path):
    from tensorflow.keras.models import load_model
    return load_model(file_path)

# Fonction principale pour charger, prétraiter, entraîner et évaluer
def main():
    current_dir = os.path.dirname(__file__)  # Répertoire du script actuel
    data_file_path = os.path.join(current_dir, 'historical_data.csv')
    
    # Charger et préparer les données
    df = load_data(data_file_path)
    df = add_indicators(df)
    
    # Prétraitement des données
    X, y = preprocess_data(df)
    
    # Diviser en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Rééchantillonner les données d'entraînement
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)
    
    # Créer et entraîner le modèle LSTM
    model = create_lstm_model((X_train_resampled.shape[1], 1))  # Adapter la forme des données d'entrée pour LSTM
    
    # Reshaper les données d'entrée pour LSTM (nécessite des données 3D)
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                     patience=3,  # Nombre d'époques sans amélioration avant réduction
                                     verbose=1, 
                                     factor=0.5,  # Réduction du taux d'apprentissage de moitié
                                     min_lr=0.0001)
    
    model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=128, validation_split=0.2, callbacks=[lr_reduction])
    
    # Évaluer le modèle
    evaluate_model(model, X_test, y_test)
    
    # Sauvegarder le modèle
    model_file_path = os.path.join(current_dir, 'trading_model_nn.h5')
    save_model(model, model_file_path)

if __name__ == "__main__":
    main()
