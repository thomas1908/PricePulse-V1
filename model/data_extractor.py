import ccxt
import pandas as pd
import datetime
import os
import time

# Initialisation de l'exchange, ici Binance
exchange = ccxt.binance()

# Récupérer les 365 derniers jours de données horaires pour le Bitcoin (BTC/USDT)
symbol = 'BTC/USDT'
timeframe = '1h'  # Intervalle d'1 heure
since = exchange.parse8601((datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat())

# Initialiser la liste pour stocker les données
data = []

# Limite de 500 bougies par requête
limit = 500

# Effectuer les requêtes en plusieurs fois pour récupérer toutes les données
while True:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    
    # Si aucune donnée n'est retournée, on arrête la boucle
    if not ohlcv:
        break

    # Ajouter les données à la liste
    for ohlc in ohlcv:
        timestamp = datetime.datetime.utcfromtimestamp(ohlc[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        open_price = ohlc[1]
        high_price = ohlc[2]
        low_price = ohlc[3]
        close_price = ohlc[4]
        volume = ohlc[5]

        data.append([timestamp, open_price, high_price, low_price, close_price, volume])

    # Mettre à jour "since" pour récupérer les prochaines bougies
    since = ohlcv[-1][0] + 1  # Le timestamp de la dernière bougie + 1 milliseconde pour éviter un doublon

    # Pause pour éviter de dépasser les limites de l'API (optionnel, à ajuster selon les besoins)
    time.sleep(1)

# Créer un DataFrame et l'enregistrer en .csv
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

# Sauvegarder en fichier CSV
current_dir = os.path.dirname(__file__)  # Répertoire du script actuel
data_file_path = os.path.join(current_dir, 'historical_data.csv')
df.to_csv(data_file_path, index=False)

print("Données sauvegardées dans 'historical_data.csv'")
