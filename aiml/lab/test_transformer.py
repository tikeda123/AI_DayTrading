import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

# ... [データの読み込みと前処理のコード] ...
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# データの読み込み

file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20220101000_20240112000_60_price_lower_mlts.csv'

data = pd.read_csv(file_path)

# Selecting relevant columns for features
"""
feature_columns = ['close', 'oi', 'volume', 'upper2', 'middle', 
                   'lower2','p_close','macdhist','turnover','ema',
                   'high','low','rsi']

feature_columns = ['close', 'oi', 'volume', 'upper2', 'middle', 
                   'lower2','p_close','macdhist','turnover','ema','sma']

"""

#feature_columns = ['close','sma','ema', 'oi', 'volume']

#feature_columns = ['close',  'upper2', 'lower2', 'middle','volume_ma','macdhist','turnover','ema','sma','oi']

data['upper_diff'] = data['upper2'] - data['close']
data['lower_diff'] = data['lower2']- data['close']
data['middle_diff'] = data['middle']- data['close']
data['ema_diff'] = data['ema']- data['close']
data['sma_diff'] = data['sma']- data['close']
data['close_diff'] = data['close'] - data['entry_price']
data['volume_diff'] = data['volume'] - data['entry_volume']
data['rsi_sell'] = data['rsi'] - 70
data['rsi_buy'] = data['rsi'] - 30
data['dmi_diff'] = data['p_di'] - data['m_di']
data['macd_diff'] = data['macd']- data['macdsignal']

feature_columns = ['close','rsi_sell','rsi_buy','dmi_diff','macd_diff','oi','macdhist','funding_rate']

#feature_columns = ['close',  'upper_diff', 'lower_diff', 'middle_diff','volume_ma','macdhist','turnover','ema','sma']
#feature_columns = ['close',  'upper_diff', 'lower_diff', 'middle_diff','volume']
#feature_columns = ['close',  'upper2', 'lower2', 'middle','volume_ma','macdhist','turnover','ema','sma']

#feature_columns = ['close','close_diff','upper_diff','lower_diff','middle_diff','ema_diff','sma_diff','ema','sma']

#feature_columns = ['close','close_diff']

#feature_columns = ['close',  'upper_diff', 'lower_diff', 'middle_diff','volume','macdhist','turnover','rsi_sell','rsi_buy']

# Filtering rows based on the condition specified
filtered_data = data[(data['bb_direction'].isin(['upper', 'lower'])) & (data['bb_profit'] != 0)]

sequences = []
targets = []

for i in range(len(filtered_data)):
    end_index = filtered_data.index[i]
    start_index = end_index - 7

    # Check if within the range of the dataset
    if end_index > len(data):
        break

    # Extract sequence and corresponding target
    sequence = data.loc[start_index:end_index, feature_columns].values
    target = data.loc[end_index, 'bb_profit'] > 0  # True if bb_profit > 0, False otherwise
    sequences.append(sequence)
    targets.append(target)

# Convert to numpy arrays
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


# 位置エンコーディング関数の定義
def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding

# Transformerブロックの定義
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)  # 自己注意
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# モデルの構築
# モデルの構築
def create_transformer_model(input_shape, num_heads, dff, rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding[:, :input_shape[0], :]

    x = TransformerBlock(input_shape[1], num_heads, dff, rate)(x)

    x = Dense(20, activation='relu')(x)
    x = Dense(10, activation='relu')(x)

    # Flatten層を追加して出力を1次元化
    x = tf.keras.layers.Flatten()(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



# 'scaled_sequences'は前処理されたシーケンスデータ
input_shape = (scaled_sequences.shape[1], scaled_sequences.shape[2])


# モデルの作成
num_heads = 16
dff = 512

input_shape = (scaled_sequences.shape[1], scaled_sequences.shape[2])
transformer_model = create_transformer_model(input_shape, num_heads, dff)

# モデルのコンパイル
transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# モデルの概要表示
transformer_model.summary()


# データの分割
x_train, x_test, y_train, y_test = train_test_split(scaled_sequences, targets, test_size=0.2, random_state=76)

# モデルのトレーニング
history = transformer_model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))

# モデルの評価
y_pred = transformer_model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)  # しきい値を0.5として、予測値を二値化

# 精度の計算
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 分類レポートの表示
report = classification_report(y_test, y_pred)
print(report)

# 混同行列の表示
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


