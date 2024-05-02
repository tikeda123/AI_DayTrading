
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.regularizers import l2


def step_decay(epoch):
    """学習率を段階的に減少させる関数です。

    Args:
        epoch (int): 現在のエポック数。

    Returns:
        float: 新しい学習率。
    """
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 20.0
    lr = initial_lr * (drop ** np.floor((1+epoch)/epochs_drop))
    return lr

class TransformerBlock(tf.keras.layers.Layer):
    """Transformerモデルのブロックを表すクラスです。

    Args:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンション機構のヘッド数。
        dff (int): フィードフォワードネットワークの次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。

    Attributes:
        mha (MultiHeadAttention): マルチヘッドアテンション層。
        ffn (Sequential): フィードフォワードネットワーク層。
        layernorm1 (LayerNormalization): 最初のレイヤー正規化層。
        layernorm2 (LayerNormalization): 二番目のレイヤー正規化層。
        dropout1 (Dropout): 最初のドロップアウト層。
        dropout2 (Dropout): 二番目のドロップアウト層。
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, l2_reg=0.01):
        """モデルの実行を行います。

        Args:
            x (Tensor): 入力テンソル。
            training (bool): トレーニングモードかどうか。

        Returns:
            Tensor: 出力テンソル。
        """
        super().__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        # L2正規化をDense層に適用
        self.ffn = Sequential([
            Dense(dff, activation='relu', kernel_regularizer=l2(l2_reg)),
            Dense(d_model, kernel_regularizer=l2(l2_reg))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):
        """モデルの実行を行います。

        Args:
            x (Tensor): 入力テンソル。
            training (bool): トレーニングモードかどうか。

        Returns:
            Tensor: 出力テンソル。
        """
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)