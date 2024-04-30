import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Flatten
from tensorflow.keras.optimizers import Adam



def create_learnning_data(file_path):
    data = pd.read_csv(file_path)
    # 必要な特徴量を選択
    #feature_columns = ['close', 'volume','volume_ma_diff','volume_ma', 'sma','sma_diff','ema','ema_diff','macdhist', 'rsi_buy','rsi_sell', 'oi', 'funding_rate','dmi_diff','macd_diff']
    feature_columns = ['close', 'bol_diff','volume_ma', 'sma','ema','ema_diff','macdhist', 'rsi','dmi_diff']

    # データの準備
    sequences = []
    targets = []

    for i in range(len(data) - 9):
        sequence = data.loc[i:i+7, feature_columns].values  # 7日間のデータ
        target = int(data.loc[i+8, 'bol_diff'] > data.loc[i+7, 'bol_diff'])  # 8日目のcloseが7日目より高ければ1、そうでなければ0
        sequences.append(sequence)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

# 特徴量のスケーリング
    scaler = MinMaxScaler()
    scaled_sequences = np.array([scaler.fit_transform(seq) for seq in sequences])
    return scaled_sequences, targets
# データの分割

def split_data(scaled_sequences, targets,test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(scaled_sequences, targets, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

# Transformerブロックの定義
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = Sequential([Dense(dff, activation='relu'), Dense(d_model)])
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
def create_transformer_model(input_shape, num_heads, dff, rate=0.1):
    inputs = Input(shape=input_shape)
    transformer_block = TransformerBlock(input_shape[-1], num_heads, dff, rate)
    x = transformer_block(inputs)

    x = Flatten()(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230601000_60_price_ml.csv'

scaled_sequences, targets = create_learnning_data(file_path)
x_train, x_test, y_train, y_test = split_data(scaled_sequences, targets)

input_shape = (scaled_sequences.shape[1], scaled_sequences.shape[2])
num_heads = 16
dff = 1024
model = create_transformer_model(input_shape, num_heads, dff)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# モデルの概要表示
model.summary()

# モデルのトレーニング
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# モデルの評価
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

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


#学習したモデルから新しいデータを予測する


new_file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230701000_20240201000_60_price_ml.csv'

scaled_sequences_new, targets_new = create_learnning_data(new_file_path)
x_train_new, x_test_new, y_train_new, y_test_new = split_data(scaled_sequences_new, targets_new,test_size=0.8, random_state=12)



# モデルの評価
y_pred_new = model.predict(x_test_new)
y_pred_new = (y_pred_new > 0.5).astype(int)

# 精度の計算
accuracy_new = accuracy_score(y_test_new, y_pred_new)
print(f"Accuracy: {accuracy_new}")

# 分類レポートの表示
report_new = classification_report(y_test_new, y_pred_new)
print(report_new)

# 混同行列の表示
conf_matrix_new = confusion_matrix(y_test_new, y_pred_new)
print("Confusion Matrix:")
print(conf_matrix_new)

