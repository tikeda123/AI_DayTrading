# 完全なPythonプログラムを提示します。

import pandas as pd
import matplotlib.pyplot as plt

# データの読み込みと準備
file_path = '/mnt/data/BTCUSDT_20231224000_2024011110_240.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 急反転ポイントを見つけるための関数
def find_reversal_points(df, window=5):
    reversal_points = []
    for i in range(window, len(df) - window):
        if df['close'][i] > df['close'][i - window] and df['close'][i] > df['close'][i + window]:
            reversal_points.append((df.index[i], df['close'][i], 'resistance'))
        elif df['close'][i] < df['close'][i - window] and df['close'][i] < df['close'][i + window]:
            reversal_points.append((df.index[i], df['close'][i], 'support'))
    return reversal_points

# 最も顕著なサポートポイントとレジスタンスポイントを選択する関数
def find_prominent_reversal_points(reversal_points):
    support_points = [point for point in reversal_points if point[2] == 'support']
    resistance_points = [point for point in reversal_points if point[2] == 'resistance']
    prominent_support = min(support_points, key=lambda x: x[1]) if support_points else None
    prominent_resistance = max(resistance_points, key=lambda x: x[1]) if resistance_points else None
    return prominent_support, prominent_resistance

# 急反転ポイントを見つける
reversal_points = find_reversal_points(data)

# 最も顕著なサポートポイントとレジスタンスポイントを選択
prominent_support, prominent_resistance = find_prominent_reversal_points(reversal_points)

# チャートにプロット
plt.figure(figsize=(15, 7))
plt.plot(data['close'], label='Close Price', alpha=0.5)
if prominent_support:
    plt.axhline(y=prominent_support[1], color='green', linestyle='--', label=f'Support Line at {prominent_support[1]}')
    plt.scatter(prominent_support[0], prominent_support[1], color='green', marker='^', label='Prominent Support Point')
if prominent_resistance:
    plt.axhline(y=prominent_resistance[1], color='red', linestyle='--', label=f'Resistance Line at {prominent_resistance[1]}')
    plt.scatter(prominent_resistance[0], prominent_resistance[1], color='red', marker='v', label='Prominent Resistance Point')
plt.title('BTC/USDT Price Chart with Prominent Support and Resistance Lines')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.show()
