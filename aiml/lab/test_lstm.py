import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

# データの読み込み
#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230101000_20240121000_30_price_lower_mlts.csv'
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20220101000_20240112000_60_price_upper_mlts.csv'



# ファイルの読み込み
data = pd.read_csv(file_path)

# Selecting relevant columns for features
#feature_columns = ['close', 'oi', 'volume_ma', 'upper2', 'middle', 'lower2','macdhist','turnover','ema']
#feature_columns = ['close', 'oi', 'volume_ma', 'macdhist','turnover','ema']

data['upper_diff'] = data['upper2'] - data['close']
data['lower_diff'] = data['lower2']- data['close']
data['middle_diff'] = data['middle']- data['close']
data['ema_diff'] = data['ema']- data['close']
data['sma_diff'] = data['sma']- data['close']
data['close_diff'] = data['close'] - data['entry_price']
data['rsi_sell'] = data['rsi'] - 70
data['rsi_buy'] = data['rsi'] - 30
data['dmi_diff'] = data['p_di'] - data['m_di']
data['macd_diff'] = data['macd']- data['macdsignal']


#feature_columns = ['close',  'upper_diff', 'lower_diff', 'middle_diff','volume_ma','macdhist','turnover','ema','sma']
#feature_columns = ['close',  'upper_diff', 'lower_diff', 'middle_diff','volume']
#feature_columns = ['close',  'upper2', 'lower2', 'middle','volume','macdhist','turnover','ema','sma','p_di','m_di','adx','rsi']

#feature_columns = ['close','close_diff','upper_diff','lower_diff','middle_diff','ema_diff','sma_diff','ema','sma']

#feature_columns = ['close','rsi_sell','rsi_buy','macdhist','close_diff']
#feature_columns = ['close','dmi_diff','rsi_sell','rsi_buy']
feature_columns = ['close','rsi_sell','rsi_buy','dmi_diff','macd_diff','close_diff','oi','macdhist']

#feature_columns = ['close','dmi_diff']


# Filtering rows based on the condition specified
filtered_data = data[(data['bb_direction'].isin(['upper', 'lower'])) & (data['bb_profit'] != 0)]

# Initializing list to hold sequences and corresponding targets
sequences = []
targets = []

for i in range(len(filtered_data)):
    end_index = filtered_data.index[i]
    start_index = end_index - 7

    # データセットの範囲内であることを確認
    if end_index > len(data):
        break

    sequence = data.loc[start_index:end_index, feature_columns].values
    target = data.loc[end_index, 'bb_profit'] > 0  # True if bb_profit > 0, False otherwise
    sequences.append(sequence)
    targets.append(target)
    """
    # 対象の行が起点行であることを確認（'bb_direction'が'upper'/'lower'、'bb_profit'が0以外）
    if data.loc[end_index, 'bb_direction'] in ['upper', 'lower'] and data.loc[end_index, 'bb_profit'] != 0:
        # シーケンスと対応するターゲットを抽出
        sequence = data.loc[start_index:end_index, feature_columns].values
        target = data.loc[end_index, 'bb_profit'] > 0  # True if bb_profit > 0, False otherwise
        sequences.append(sequence)
        targets.append(target)
     """
# numpy配列に変換
sequences = np.array(sequences)
targets = np.array(targets)



# 少数派クラスのサンプルを特定
minority_class = 1 if np.sum(targets) < len(targets) / 2 else 0
minority_samples = sequences[targets == minority_class]

# 多数派クラスのサンプル数を計算
majority_count = len(sequences) - len(minority_samples)

# 少数派クラスのサンプルを複製
num_to_augment = majority_count - len(minority_samples)
augmented_minority_samples = np.repeat(minority_samples, num_to_augment // len(minority_samples), axis=0)

# ターゲットも同様に複製
augmented_targets = np.full(len(augmented_minority_samples), minority_class)

# 元のデータセットと結合
sequences = np.concatenate([sequences, augmented_minority_samples], axis=0)
targets = np.concatenate([targets, augmented_targets])





# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sequences = np.array([scaler.fit_transform(seq) for seq in sequences])


# データの前処理
# ここで'scaled_sequences'と'targets'を使用
# LSTMモデルの定義
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='ReLU', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='ReLU'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# モデルの作成
input_shape = (scaled_sequences.shape[1], scaled_sequences.shape[2])  # (8, 6)
lstm_model = create_lstm_model(input_shape)

# モデルの概要表示
lstm_model.summary()

# データの分割（トレーニングデータとテストデータ）
# ここでは簡単のため、すべてのデータをトレーニングに使用しますが、実際には適切な分割が必要です
x_train = scaled_sequences
y_train = targets

x_train, x_test, y_train, y_test = train_test_split(scaled_sequences, targets, test_size=0.3, random_state=52)


# モデルのトレーニング
lstm_model.fit(x_train, y_train, epochs=140, batch_size=32)


y_pred = lstm_model.predict(x_test)
# 予測値の平均と標準偏差の計算

# 新しいしきい値を使用して予測値を二値化
y_pred_binary = (y_pred > 0.5).astype(int)


# 精度の計算
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")

# 分類レポートの表示
report = classification_report(y_test, y_pred_binary)
print(report)

# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred_binary)
class_report = classification_report(y_test, y_pred_binary)

# 混同行列と分類レポートの表示
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

