import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ステップ1: データ読み込みと前処理
file_path = 'BTCUSDT_20231224000_2024011110_240.csv'  # ここに適切なファイルパスを入力
btc_data = pd.read_csv(file_path)
btc_data['date'] = pd.to_datetime(btc_data['date'])

# ステップ2: データ分割
split_indices = np.array_split(btc_data.index, 3)

# ステップ3: 最高値と最低値の抽出
max_values, min_values = [], []
max_indices, min_indices = [], []
for period_indices in split_indices:
    period_data = btc_data.loc[period_indices]
    max_index = period_data['high'].idxmax()
    min_index = period_data['low'].idxmin()
    max_values.append(btc_data.loc[max_index, 'high'])  # 修正: locを使用
    min_values.append(btc_data.loc[min_index, 'low'])   # 修正: locを使用
    max_indices.append(max_index)
    min_indices.append(min_index)
# ステップ4: レジスタンスラインとサポートラインの計算
resistance_slope, resistance_intercept, _, _, _ = linregress(max_indices, max_values)
support_slope, support_intercept, _, _, _ = linregress(min_indices, min_values)

# ステップ5: グラフ描画
plt.figure(figsize=(14, 7))
plt.plot(btc_data['date'], btc_data['close'], label='Close Price', color='blue')
plt.plot(btc_data['date'], [resistance_slope * x + resistance_intercept for x in range(len(btc_data))], label='Resistance Line', color='red')
plt.plot(btc_data['date'], [support_slope * x + support_intercept for x in range(len(btc_data))], label='Support Line', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('BTC Price with Support and Resistance Lines')
plt.legend()
plt.grid(True)
plt.show()
