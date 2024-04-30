import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルの読み込み
file_path = 'oi_test01_upper_60.csv'  # ファイルパスを指定
data = pd.read_csv(file_path)

# Min-Max正規化の関数
def normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized = (column - min_val) / (max_val - min_val)
    return normalized

# 'funding_rate' と 'close' の正規化
data['normalized_funding_rate'] = normalize(data['funding_rate'])
data['normalized_close'] = normalize(data['close'])

# グラフの描画
plt.figure(figsize=(14, 7))

# 正規化された 'close' のプロット
plt.plot(data['date'], data['normalized_close'], label='Normalized Close', color='blue')

# 正規化された 'funding_rate' のプロット
plt.plot(data['date'], data['normalized_funding_rate'], label='Normalized Funding Rate', color='red')

# グラフのタイトルとラベル
plt.title('Normalized Close and Funding Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Value')

# 凡例
plt.legend()

# x軸の日付表示を回転
plt.xticks(rotation=45)

# グラフの表示
plt.tight_layout()
plt.show()
