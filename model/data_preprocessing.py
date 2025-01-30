import pandas as pd
import talib

def add_indicators(df):
    # Calcul des indicateurs techniques
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['EMA'] = talib.EMA(df['close'], timeperiod=14)
    df['moving_average'] = df['price'].rolling(window=5).mean()
    df['momentum'] = df['price'].diff()
    
    # Ajout d'autres indicateurs techniques si nécessaire
    # Par exemple, Bollinger Bands, Moving Averages, etc.
    
    df = df.dropna()  # Supprimer les lignes avec des valeurs manquantes
    return df
