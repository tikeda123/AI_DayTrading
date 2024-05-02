import os,sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
import joblib


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger_db import TradingLogger
from common.config_manager import ConfigManager
from common.constants import *
# ハイパーパラメータの設定
PARAM_learning_rate = 0.001
PARAM_epochs = 1100

N_SPLITS=4
POSITIVE_THRESHOLD = 100

def time_series_data(data: pd.DataFrame, ftime_steps,feature_columns)->tuple:
    """
    指定されたデータフレームから時系列データのシーケンスとターゲットを生成します。

    Args:
        data (pd.DataFrame): 時系列データが含まれるデータフレーム。
        ftime_steps (int): 生成するシーケンスのタイムステップ数。
        feature_columns (list): シーケンス生成に使用する特徴量のカラム名のリスト。

    Returns:
        tuple: 生成されたシーケンスとターゲットのnumpy配列のタプル。
    """

    filtered_data = data[(data[COLUMN_BB_DIRECTION].isin([BB_DIRECTION_UPPER, BB_DIRECTION_LOWER])) & (data[COLUMN_BB_PROFIT] != 0)]
        # シーケンスとターゲットの生成
    sequences, targets = [], []

    for i in range(len(filtered_data)):
        end_index = filtered_data.index[i]
        start_index = end_index - ftime_steps
        if end_index > len(data):
            break

        sequence = data.loc[start_index:end_index, feature_columns].values
        target = data.loc[end_index, COLUMN_BB_PROFIT] > POSITIVE_THRESHOLD
        sequences.append(sequence)
        targets.append(target)

        # スケーリング
    sequences = np.array(sequences)
    targets = np.array(targets)
    return sequences, targets


def step_decay(epoch)->float:
    """
    トレーニングのエポック数に応じて学習率を段階的に減衰させる関数。

    Args:
        epoch (int): 現在のエポック数。

    Returns:
        float: 計算された新しい学習率。
    """
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 20.0
    lr = initial_lr * (drop ** np.floor((1+epoch)/epochs_drop))
    return lr


class TransformerBlock(tf.keras.layers.Layer):
    """
    TransformerブロックのカスタムKerasレイヤークラスです。マルチヘッドアテンションとフィードフォワードネットワークを含みます。

    Attributes:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンションヘッドの数。
        dff (int): フィードフォワードネットワークの内部層の次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。

    Args:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンションヘッドの数。
        dff (int): フィードフォワードネットワークの内部層の次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, l2_reg=0.01):
        """
        TransformerBlockのインスタンスを初期化します。
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
        """
        レイヤーの論理を実装します。マルチヘッドアテンションとフィードフォワードネットワークを適用します。

        Args:
            x (Tensor): 入力テンソル。
            training (bool, optional): トレーニングモードかどうか。デフォルトはFalse。

        Returns:
            Tensor: 出力テンソル。
        """
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Transformer_PredictionModel:
    """
    Transformerを使用した予測モデルクラスです。設定管理、ログ記録、データの前処理、モデルのトレーニングと評価を行います。

    Attributes:
        __config_manager (ConfigManager): 設定管理オブジェクト。
        __logger (TradingLogger): ログ記録オブジェクト。
        __model (tf.keras.Model): トレーニングされたモデル。
        __scaler (MinMaxScaler): 特徴量のスケーリングに使用するオブジェクト。
        __pca (PCA): 主成分分析に使用するオブジェクト。
        __feature_columns (list): モデルの入力に使用する特徴量のカラム名のリスト。

    Args:
        config_manager (ConfigManager): 設定管理オブジェクト。
        trading_logger (TradingLogger): ログ記録オブジェクト。
    """

    def __init__(self):
        """
        Transformer_PredictionModelクラスのインスタンスを初期化します。
        """
        self.__config_manager = ConfigManager()
        self.__logger = TradingLogger()
        self.__initialize_paths()
        self.__model = None
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__pca = PCA()

    def get_feature_columns(self):
        """
        モデルに使用する特徴量カラムのリストを返します。

        Returns:
            list: 特徴量カラムのリスト。
        """
        return self.__feature_columns

    def __initialize_paths(self):
        """
        モデルとデータファイルのパスを初期化します。
        """
        self.__datapath = parent_dir + '/' + self.__config_manager.get('AIML', 'DATAPATH')
        self.__feature_columns = self.__config_manager.get('AIML', 'FEATURE_COLUMNS')

    def load_data_from_csv(self, learning_datafile: str):
        """
        指定されたCSVファイルからデータを読み込み、pandas DataFrameとして返します。

        このメソッドは、学習データファイルのパスを組み立て、pandasを使用してCSVファイルを読み込みます。
        ファイルが存在しない場合、FileNotFoundErrorをキャッチし、ログに記録した後に例外を再発生させます。

        Args:
            learning_datafile (str): 読み込む学習データのCSVファイル名。このファイル名は、内部で保持されているデータパスに追加されます。

        Returns:
            pd.DataFrame: 読み込まれたデータを含むDataFrame。

        Raises:
            FileNotFoundError: 指定されたCSVファイルが見つからない場合に発生します。
        """
        data_path = self.__datapath + learning_datafile
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError as e:
            self.__logger.log_system_message(f"FileNotFoundError: {e}")
            raise
        return df

    def apply_pca(self, data, n_components=None):
        """
        データセットに対してPCA(主成分分析)を適用します。

        Args:
            data (np.array): PCAを適用するデータセット。
            n_components (int, optional): 保持する主成分の数。デフォルトでは特徴量の数の半分を使用。

        Returns:
            np.array: PCA適用後のデータセット。
        """
        if n_components is None:
            # 特徴量の数の半分を使用
            n_components = max(1, data.shape[2] // 2)
        self.__pca = PCA(n_components=n_components)
        # データの形状を変更（PCAが2D入力を要求するため）
        reshaped_data = data.reshape(-1, data.shape[2])
        # PCAを適用
        pca_data = self.__pca.fit_transform(reshaped_data)
        # データの形状を元に戻す
        pca_data = pca_data.reshape(-1, data.shape[1], n_components)
        return pca_data

    def load_and_prepare_data(self, learning_datafile, test_size=0.8, random_state=None, augmentation_enabled=False,apply_pca=False, n_components=None):
        """
        指定されたデータファイルからデータを読み込み、前処理を行った後、訓練用とテスト用データセットに分割します。

        Args:
            learning_datafile (str): 学習データファイルのパス。
            test_size (float): テストデータセットの割合。
            random_state (int, optional): データ分割時の乱数シード。
            augmentation_enabled (bool): データ拡張を行うかどうか。
            apply_pca (bool): PCAを適用するかどうか。
            n_components (int, optional): PCAで使用する主成分の数。

        Returns:
            tuple: 訓練用データセットとテスト用データセット。
        """
        data, targets = self.__load_and_preprocess_data(learning_datafile,augmentation_enabled,apply_pca)
        if apply_pca:
            data = self.apply_pca(data, n_components)
        return train_test_split(data, targets, test_size=test_size, random_state=random_state)

    def remake_feature_columns(self,data):
        """
        データフレームから不要な特徴量を除去し、使用する特徴量のカラムリストを更新します。

        Args:
            data (pd.DataFrame): 元のデータフレーム。

        Returns:
            list: 使用する特徴量のカラム名リスト。
        """
        column_names = data.columns.tolist()
        columns_to_remove = ['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma']
        filtered_columns = [column for column in column_names if column not in columns_to_remove]
        self.__feature_columns = filtered_columns
        return self.__feature_columns


    def __load_and_preprocess_data(self, learning_datafile,augmentation_enabled=False,apply_pca=False):
        """
        データファイルを読み込み、必要に応じてデータ拡張やPCAを適用します。

        Args:
            learning_datafile (str): 読み込むデータファイルのパス。
            augmentation_enabled (bool): データ拡張を行うかどうか。
            apply_pca (bool): PCAを適用するかどうか。

        Returns:
            tuple: 前処理後のデータセットとターゲット。
        """
        data = self.load_data_from_csv(learning_datafile)
        if data is None:
            raise ValueError(f"データが読み込めませんでした: {learning_datafile}")

        data['entry_diff'] = data['entry_price']-data['close']

        if apply_pca:
            self.remake_feature_columns(data)

        sequences, targets = time_series_data(data, TIME_SERIES_PERIOD-1, self.__feature_columns)        #scaler = MinMaxScaler(feature_range=(0, 1))  # トレーニング用スケーラー
        scaled_sequences = np.array([self.__scaler.fit_transform(seq) for seq in sequences])

        if not augmentation_enabled:
            return scaled_sequences, targets
        return self.__balance_classes_by_augmentation(scaled_sequences, targets)

    def __balance_classes_by_augmentation(self, sequences, targets):
        """
        データセットのクラスバランスを調整するために、少数クラスのサンプルを増加させます。

        Args:
            sequences (np.array): 入力データのシーケンス。
            targets (np.array): ターゲットのラベル。

        Returns:
            tuple: クラスバランスを調整後のシーケンスとターゲット。
        """
        # 少数派クラスのサンプルを特定
        minority_class = 1 if np.sum(targets) < len(targets) / 2 else 0
        minority_samples = sequences[targets == minority_class]

        # 多数派クラスのサンプル数を計算
        majority_count = len(sequences) - len(minority_samples)

        # 少数派クラスのサンプルを複製
        num_to_augment = majority_count - len(minority_samples)
        print(f"Augmenting num_to_augment {num_to_augment} samples, minority_len: {len(minority_samples)}")
        augmented_minority_samples = np.repeat(minority_samples, num_to_augment // len(minority_samples), axis=0)

        # ターゲットも同様に複製
        augmented_targets = np.full(len(augmented_minority_samples), minority_class)

        # 元のデータセットと結合
        augmented_sequences = np.concatenate([sequences, augmented_minority_samples], axis=0)
        augmented_targets = np.concatenate([targets, augmented_targets])

        return augmented_sequences, augmented_targets


    def create_transformer_model(self,input_shape, num_heads=8, dff=256, rate=0.1, l2_reg=0.01):
        """
        Transformerモデルを作成します。

        Args:
            input_shape (tuple): 入力データの形状。
            num_heads (int): アテンション機構のヘッド数。
            dff (int): フィードフォワードネットワークの次元数。
            rate (float): ドロップアウト率。
            l2_reg (float): L2正則化の係数。

        Returns:
            tf.keras.Model: 作成されたTransformerモデル。
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = TransformerBlock(input_shape[1], num_heads, dff, rate, l2_reg=l2_reg)(inputs)
        x = Dropout(0.2)(x)
        x = Dense(20, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train, epochs=200, batch_size=32):
        """
        モデルを訓練します。

        Args:
            x_train (np.array): 訓練データ。
            y_train (np.array): 訓練データのラベル。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
        """       # モデルのトレーニング
        self.__model = self.create_transformer_model((x_train.shape[1], x_train.shape[2]))
        self.__model.compile(optimizer=Adam(learning_rate=PARAM_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.__model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    def train_with_cross_validation(self, data, targets, epochs=PARAM_epochs, batch_size=32, n_splits=N_SPLITS):
        """
        クロスバリデーションを使用してモデルを訓練します。

        Args:
            data (np.array): 全データセット。
            targets (np.array): 全ターゲット。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
            n_splits (int): 分割数。

        Returns:
            list: 各分割における評価スコア。
        """        # K-Foldクロスバリデーションを初期化
        kfold = KFold(n_splits=n_splits, shuffle=True)

        # 各フォールドでのスコアを記録するリスト
        fold_no = 1
        scores = []

        for train, test in kfold.split(data, targets):
            # モデルを生成
            self.__model = self.create_transformer_model((data.shape[1], data.shape[2]))

            # モデルをコンパイル
            self.__model.compile(optimizer=Adam(learning_rate=PARAM_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

            # モデルをトレーニング
            print(f'Training for fold {fold_no} ...')
            self.__model.fit(data[train], targets[train], epochs=epochs, batch_size=batch_size)

            # モデルの性能を評価
            scores.append(self.__model.evaluate(data[test], targets[test], verbose=0))

            fold_no += 1

        return scores

    def evaluate(self, x_test, y_test):
        """
        モデルを評価します。

        Args:
            x_test (np.array): テストデータ。
            y_test (np.array): テストデータのラベル。

        Returns:
            tuple: 正確度、分類レポート、混同行列。
        """
        # モデルの評価
        y_pred = (self.__model.predict(x_test) > 0.5).astype(int)
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

    def predict_single(self, data_point):
        """
        単一のデータポイントに対して予測を行います。

        Args:
            data_point (np.array): 予測するデータポイント。

        Returns:
            int: 予測されたクラスラベル。
        """
        # 単一データポイントの予測
        # 予測用スケーラーを使用
        #self.__scaler.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_point = self.__scaler.fit_transform(data_point)

        # モデルによる予測
        prediction = self.__model.predict(scaled_data_point.reshape(1, -1, len(self.__feature_columns)))
        prediction = (prediction > 0.5).astype(int)
        return prediction[0][0]


    def set_predict_scaler(self, features):
        """
        予測時に使用するスケーラーを設定します。

        Args:
            features (np.array): スケーリングに使用する特徴量。
        """
        # 予測用スケーラーを特定の特徴量にフィット
        self.predict_scaler.fit(features)

    def save_model(self, model_file):
        """
        トレーニング済みモデルを保存します。

        Args:
            model_file (str): 保存するモデルのファイル名。
        """
        # モデルの保存
        model_keras_file = model_file.replace('.csv', '.keras')
        model_path = os.path.join(self.__datapath, model_keras_file)
        print(model_path)
        self.__model.save(model_path)

        """
        model_pca_file = model_file.replace('.csv', '.pca')
        model_pca_path = os.path.join(self.__datapath, model_pca_file)
        print(model_pca_path)
        joblib.dump(self.__pca, model_pca_path)
         """

        model_scaler_file = model_file.replace('.csv', '.scaler')
        model_scaler_path = os.path.join(self.__datapath, model_scaler_file)
        print(model_scaler_path)
        joblib.dump(self.__scaler, model_scaler_path)

    def load_model(self, model_file):
        """
        保存されたモデルを読み込みます。

        Args:
            model_file (str): 読み込むモデルのファイル名。
        """        # モデルの読み込み
        model_keras_file = model_file.replace('.csv', '.keras')
        model_path = os.path.join(self.__datapath, model_keras_file)
        self.__model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

        """
        model_pca_file = model_file.replace('.csv', '.pca')
        model_pca_path = os.path.join(self.__datapath, model_pca_file)
        self.__pca = joblib.load(model_pca_path)
        """

        model_scaler_file = model_file.replace('.csv', '.scaler')
        model_scaler_path = os.path.join(self.__datapath, model_scaler_file)
        self.__scaler = joblib.load(model_scaler_path)



def main():
    learning_datafile = 'BTCUSDT_20210101000_20230901000_60_price_upper_mlts.csv'
    # 使用例

    btc_model = Transformer_PredictionModel()


    x_train, x_test, y_train, y_test = btc_model.load_and_prepare_data(learning_datafile,
                                                                        test_size=0.2,
                                                                        random_state=None,
                                                                        augmentation_enabled=False,
                                                                        apply_pca=False,
                                                                        n_components=16)


    # モデルをクロスバリデーションで訓練します
    cv_scores = btc_model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0)
    )

    # クロスバリデーションの結果を表示
    for i, score in enumerate(cv_scores):
        print(f'Fold {i+1}: Accuracy = {score[1]}')

    btc_model.save_model(learning_datafile)

    """
    btc_model = Transformer_PredictionModel()
    btc_model.load_model(learning_datafile)

    test_datafile = 'BTCUSDT_20230901000_20231201000_60_price_upper_mlts.csv'
    for i in range(5):
        x_train, x_test, y_train, y_test = btc_model.load_and_prepare_data(test_datafile,
                                                                        test_size=0.9,
                                                                        random_state=None,
                                                                        augmentation_enabled=False,
                                                                        apply_pca=False,
                                                                        n_components=16)




        accuracy, report, conf_matrix = btc_model.evaluate(x_test, y_test)

        print(f"Accuracy: {accuracy}")
        print(report)
        print(conf_matrix)


    win_count = 0
    lose_count = 0
    one_win_count = 0
    for i in range(len(y_test)):
        y_pred = btc_model.predict_single(x_test[i])
        print(f"Predicted value: {y_pred}, Actual value: {y_test[i]}")

        if y_pred == y_test[i]:
            win_count += 1
        else:
            lose_count += 1

        if y_pred == 1 and y_test[i] == 1:
            one_win_count += 1


    win_rate = win_count / (win_count + lose_count)
    print(f"Win rate: {win_rate}, One win rate: {one_win_count}")


    #print(y_pred)
   # btc_model.save_model('model.keras')
    # 予測用スケーラーの設定
    # ここで予測に使う特徴量の数に合わせたスケーリングを設定します。
    #btc_model.set_predict_scaler(x_test.reshape(-1, x_test.shape[-1]))

    # 個別データポイントに対する予測
   # for i in range(len(y_test)):
    #    print(f"Predicted value: {btc_model.predict_single(x_test[i])}, Actual value: {y_test[i]}")

    """
if __name__ == '__main__':
    main()
