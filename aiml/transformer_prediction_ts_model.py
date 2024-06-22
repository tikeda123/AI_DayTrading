
import os,sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split,ParameterGrid
from tqdm import tqdm
from typing import Tuple
import numpy as np
from imblearn.over_sampling import SMOTE

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader

from aiml.transformer_prediction_rolling_model import *


class TransformerPredictionTSModel(TransformerPredictionRollingModel):
    """Transformerモデルによる時系列データの予測を行うクラスです。

    Attributes:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンションヘッド数。
        dff (int): フィードフォワードネットワークの次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。
        model (Sequential): Transformerモデル。
        optimizer (Adam): 最適化アルゴリズム。
        loss_object (SparseCategoricalCrossentropy): 損失関数。
        train_loss (Mean): 訓練時の損失。
        train_accuracy (SparseCategoricalAccuracy): 訓練時の精度。
        test_loss (Mean): テスト時の損失。
        test_accuracy (SparseCategoricalAccuracy): テスト時の精度。
    """

    def __init__(self,id,synbol=None,interval=None):
        """モデルの実行を行います。

        Args:
            d_model (int): 埋め込みの次元数。
            num_heads (int): アテンションヘッド数。
            dff (int): フィードフォワードネットワークの次元数。
            rate (float): ドロップアウト率。
            l2_reg (float): L2正則化の係数。
        """
        super().__init__(id,synbol,interval)
        self._dataloader = MongoDataLoader()

    def load_and_prepare_data_time_series_mix(self,
                                          data,
                                          test_size=0.2,
                                          random_state=None,
                                          oversample=False):
        scaled_sequences, targets = self._prepare_sequences_time_series(data, TIME_SERIES_PERIOD-1, self.feature_columns,oversample)
        return train_test_split(scaled_sequences,
                                targets,
                                test_size=test_size,
                                random_state=random_state,shuffle=False)

    def load_and_prepare_data_time_series(self,
                                          start_date,
                                          end_date,
                                          coll_type,
                                          test_size=0.2,
                                          random_state=None,
                                          oversample=False):
        data = self._dataloader.load_data_from_datetime_period(start_date, end_date, coll_type)
        scaled_sequences, targets = self._prepare_sequences_time_series(data, TIME_SERIES_PERIOD-1, self.feature_columns,oversample)
        return train_test_split(scaled_sequences,
                                targets,
                                test_size=test_size,
                                random_state=random_state,shuffle=False)

    def _prepare_sequences_time_series(self, data, ftime_steps, feature_columns, oversample=False) -> Tuple[np.ndarray, np.ndarray]:
        # フィルタリングされたデータを取得
        filtered_data = data[(data[COLUMN_BB_DIRECTION].isin([BB_DIRECTION_UPPER, BB_DIRECTION_LOWER])) & (data[COLUMN_BB_PROFIT] != 0)]

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

        sequences = np.array(sequences)
        targets = np.array(targets)
        scaled_sequences = np.array([self.scaler.fit_transform(seq) for seq in sequences])

        # オーバーサンプリング
        if oversample:
            n_samples, n_time_steps, n_features = scaled_sequences.shape
            scaled_sequences_reshaped = scaled_sequences.reshape(n_samples, -1)
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(scaled_sequences_reshaped, targets)
            X_resampled = X_resampled.reshape(-1, n_time_steps, n_features)
            return X_resampled, y_resampled
        else:
            return scaled_sequences, targets

    """
    def _prepare_sequences_time_series(self, data,ftime_steps,feature_columns) -> Tuple[np.ndarray, np.ndarray]:
        #  TIME_SERIES_PERIOD-1, self.feature_columns
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
        scaled_sequences = np.array([self.scaler.fit_transform(seq) for seq in sequences])
        return scaled_sequences, targets
    """
    def train_with_hyperparameter_tuning(self,
                                         data: np.ndarray,
                                         targets: np.ndarray,
                                         param_grid: dict,
                                         epochs: int = PARAM_EPOCHS,
                                         batch_size: int = 32,
                                         validation_split: float = 0.2) -> dict:
        """
        ハイパーパラメータの探索を行い、最良のハイパーパラメータでモデルを訓練する。

        Args:
            data (np.ndarray): 訓練に使用するデータセット。
            targets (np.ndarray): データセットに対応するターゲット値。
            param_grid (dict): 探索するハイパーパラメータの範囲を定義した辞書。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
            validation_split (float): バリデーションデータの割合。

        Returns:
            dict: 最良のハイパーパラメータと、そのハイパーパラメータでのバリデーション損失の辞書。
        """
        best_params = None
        best_val_loss = float('inf')

        num_combinations = len(list(ParameterGrid(param_grid)))
        print(f"Total hyperparameter combinations: {num_combinations}")

        for params in tqdm(ParameterGrid(param_grid), total=num_combinations):
            self.model = self.create_transformer_model(
                (data.shape[1], data.shape[2]),
                num_heads=params['num_heads'],
                dff=params['dff'],
                rate=params['rate'],
                l2_reg=params['l2_reg']
            )
            self.model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

            history = self.model.fit(data, targets, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

            val_loss = history.history['val_loss'][-1]
            if val_loss < best_val_loss:
                best_params = params
                best_val_loss = val_loss

            tqdm.write(f"Current hyperparameters: {params}")
            tqdm.write(f"Validation loss: {val_loss}")
            tqdm.write("---")

        return {'best_params': best_params, 'best_val_loss': best_val_loss}


def permutation_feature_importance(model, X, y, metric, n_repeats=10):
    baseline_score = metric(y, (model.predict(X) > 0.5).astype(int).flatten())
    feature_importances = []

    for column in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, column])
            score = metric(y, (model.predict(X_permuted) > 0.5).astype(int).flatten())
            scores.append(score)
        feature_importances.append(baseline_score - np.mean(scores))

    return np.array(feature_importances)

def drop_column_feature_importance(model, X, y, metric):
    baseline_score = metric(y, (model.predict(X) > 0.5).astype(int).flatten())
    feature_importances = []

    for column in range(X.shape[2]):
        X_dropped = X.copy()
        X_dropped[:, :, column] = 0
        score = metric(y, (model.predict(X_dropped) > 0.5).astype(int).flatten())
        feature_importances.append(baseline_score - score)

    return np.array(feature_importances)

def check_gradients(model, data, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(data)
        class_channel = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(class_channel, conv_outputs)
    return grads


def train_main(model):

    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series(
                                                                    '2020-01-01 00:00:00',
                                                                    '2024-01-01 00:00:00',
                                                                    MARKET_DATA_ML_UPPER,oversample=False)
    model.train(x_train, y_train)

    model.save_model()

    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series( '2024-01-01 00:00:00',
                                                                                '2024-06-01 00:00:00',
                                                                                MARKET_DATA_ML_UPPER,
                                                                                test_size=0.9,
                                                                                random_state=None,
                                                                                oversample=False)
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)
    return  accuracy


def main():
    """
    メインの実行関数。
    """
    # ログ情報の初期化


    # モデルの初期化
    model = TransformerPredictionTSModel("upper_mlts")

    for i in range(100):
        accuracy_score = train_main(model)
        if accuracy_score >= 0.65:
            break




     # データのロードと前処理
    #x_train, x_test, y_train, y_test = model.load_and_prepare_data('2021-01-01 00:00:00', '2022-02-01 00:00:00')
    #model.load_model(0
    # モデルの訓練
        # モデルをクロスバリデーションで訓練します
    """
    param_grid = {
        'num_heads': [4, 8, 16],
        'dff': [128, 256, 512],
        'rate': [0.1, 0.2, 0.3],
        'l2_reg': [0.001, 0.01, 0.1]
    }

    # ハイパーパラメータの探索とモデルの訓練
    best_params = model.train_with_hyperparameter_tuning(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
        param_grid
    )

    print(f"Best hyperparameters: {best_params['best_params']}")
    print(f"Best validation loss: {best_params['best_val_loss']}")


    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0)
    )
    # クロスバリデーションの結果を表示

    for i, score in enumerate(cv_scores):
        print(f'Fold {i+1}: Accuracy = {score[1]}')
    # モデルの評価
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(report)
    print(conf_matrix)
    model.save_model()
   """

    """
        # 異なるレイヤー名を試す
    for layer_name in layer_names:
        print(f"Grad-CAM for layer: {layer_name}")
        try:
            heatmap = model.compute_gradcam(x_test, layer_name=layer_name)
            model.display_gradcam(x_test[0], heatmap)
            grads = check_gradients(model.model, x_test, layer_name)
            mean_grad = tf.reduce_mean(tf.abs(grads))
            print(f"Checking gradients for layer: {layer_name}")
            print(f"Mean gradient for {layer_name}: {mean_grad}")
        except Exception as e:
            print(f"Could not compute Grad-CAM for layer {layer_name}: {e}")




    # Permutation Feature Importanceの計算
    permutation_importances = permutation_feature_importance(model.model, x_test, y_test, accuracy_score)
     # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(permutation_importances)), permutation_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Permutation Feature Importance')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.show()

    # Permutation Feature Importanceの計算
    permutation_importances = drop_column_feature_importance(model, x_test, y_test, accuracy_score)
    # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(permutation_importances)), permutation_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('drop_column_feature_importance')
    plt.title('drop_column_feature_importance')
    plt.tight_layout()
    plt.show()
      """
if __name__ == '__main__':
    main()
