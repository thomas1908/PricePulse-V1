import ccxt
import pandas as pd
import datetime
import os
import time

exchange = ccxt.binance()

symbol = 'BTC/USDT'
timeframe = '1h'
since = exchange.parse8601((datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat())

data = []

limit = 500

while True:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    
    if not ohlcv:
        break

    for ohlc in ohlcv:
        timestamp = datetime.datetime.utcfromtimestamp(ohlc[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        open_price = ohlc[1]
        high_price = ohlc[2]
        low_price = ohlc[3]
        close_price = ohlc[4]
        volume = ohlc[5]

        data.append([timestamp, open_price, high_price, low_price, close_price, volume])

    since = ohlcv[-1][0] + 1

    time.sleep(1)


df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])


current_dir = os.path.dirname(__file__)
data_file_path = os.path.join(current_dir, 'historical_data.csv')
df.to_csv(data_file_path, index=False)

print("Données sauvegardées dans 'historical_data.csv'")
