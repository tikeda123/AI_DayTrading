import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)
from common.data_loader_db import DataLoaderDB
from common.constants import COLUMN_CLOSE, COLUMN_RSI, COLUMN_MACDHIST

def bayes_model(df):
    # 特徴量とターゲットを定義
    feature = 'rsi'
    target = 'bb_profit'

    # 価格が5期間の間に上昇したかどうかを判断
    df['bb_profit_positive'] = np.where(df['bb_profit'] < 0, 1, 0)

    # 特徴量とターゲットをnumpyの配列に変換
    X = df[feature].values.reshape(-1, 1)
    y = df['bb_profit_positive'].values

    # データを学習用とテスト用に分割
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ガウシアンナイーブベイズモデルの初期化と学習
    model = GaussianNB()
    model.fit(X_train, y_train)

    # テストデータに対する予測と精度の計算
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # テストデータに対する予測確率の取得
    probabilities = model.predict_proba(X_test)
    print("Probabilities:")
    print(probabilities)

    # RSIの値の範囲を定義
    rsi_range = np.arange(20, 80, 1)

    # 各RSIの値に対して、bb_profitがポジティブになる確率を計算し、結果を保存
    results = []
    for rsi in rsi_range:
        features = np.array([[rsi]])
        probability_positive = model.predict_proba(features)[0][1]
        results.append((rsi, probability_positive))

    # 結果を確率の高い順にソート
    results.sort(key=lambda x: x[1], reverse=True)

    # 上位10個の組み合わせを表示
    print("Top 20 combinations:")
    for i in range(50):
        rsi, probability_positive = results[i]
        print(f"RSI: {rsi:.2f}, Probability Positive: {probability_positive:.2f}")

def main():
    data_loader = DataLoaderDB()
    filename = "data_ml/ETHUSDT_20200101_20230101_60_price_upper_mlnonts.csv"
    df = data_loader.load_data_from_csv(filename)
    print(df)
    bayes_model(df)

if __name__ == "__main__":
    main()