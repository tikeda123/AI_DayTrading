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
from common.constants import COLUMN_CLOSE, COLUMN_RSI,COLUMN_EMA

def bayes_model(df):
    # 特徴量とターゲットを定義
    features = [COLUMN_RSI]
    target = 'price_direction'

    # ターゲット変数の作成
    df[target] = np.where(df[COLUMN_CLOSE].shift(-1)  < df[COLUMN_CLOSE], 1, 0)

    # 最終行を削除（ターゲット変数が欠損しているため）
    df = df[:-1]

    # 特徴量とターゲットをnumpyの配列に変換
    X = df[features].values
    y = df[target].values

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

    # 特徴量の値の範囲を定義
    rsi_range = np.arange(10, 100, 1)

    # 各特徴量の組み合わせに対して、価格がUPする確率を計算し、結果を保存
    results = []
    for rsi in rsi_range:
        features = np.array([[rsi]])
        probability_up = model.predict_proba(features)[0][1]
        results.append((rsi, probability_up))

    # 結果を確率の高い順にソート
    results.sort(key=lambda x: x[1], reverse=True)

    # 上位10個の組み合わせを表示
    print("Top 50 combinations:")
    for i in range(50):
        rsi, probability_up = results[i]
        print(f"RSI: {rsi:.2f}, Probability DOWN: {probability_up:.2f}")

    # 上位の組み合わせに共通する特徴量の値の範囲を特定（例：上位10個の平均値を計算）
    top_n = 50
    rsi_sum = sum(result[0] for result in results[:top_n])
    rsi_mean = rsi_sum / top_n

    print(f"Common range for top {top_n} combinations:")
    print(f"RSI: {rsi_mean:.2f}")

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)
from common.data_loader_db import DataLoaderDB

def main():
    data_loader = DataLoaderDB()
    df = data_loader.load_data_from_db("btcusdt_60_market_data_tech")
    bayes_model(df)

if __name__ == "__main__":
    main()