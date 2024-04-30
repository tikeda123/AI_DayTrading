import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
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
from common.constants import COLUMN_CLOSE, COLUMN_RSI

def bayes_model(df):
    # 特徴量とターゲットを定義
    feature = COLUMN_RSI
    target = 'price_direction'

    # 価格が5期間の間に上昇したかどうかを判断
    df['price_increased'] = np.where(df[COLUMN_CLOSE].rolling(window=3).max().shift(-3) > df[COLUMN_CLOSE], 1, 0)

    # RSIを5の区切りごとに分類
    rsi_bins = pd.cut(df[COLUMN_RSI], bins=range(0, 101, 5), labels=False, right=False)

    # 各区切りでの確率を計算
    rsi_ranges = range(0, 100, 5)
    probabilities = []

    for rsi_start in rsi_ranges:
        rsi_end = rsi_start + 5
        mask = (df[COLUMN_RSI] >= rsi_start) & (df[COLUMN_RSI] < rsi_end)
        X = df.loc[mask, feature].values.reshape(-1, 1)
        y = df.loc[mask, 'price_increased'].values

        # データが存在する場合のみ学習を行う
        if len(X) > 0:
            # データを学習用とテスト用に分割
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # ガウシアンナイーブベイズモデルの初期化と学習
            model = GaussianNB()
            model.fit(X_train, y_train)

            # テストデータに対する予測確率の取得
            if len(np.unique(y_test)) == 2:
                prob = model.predict_proba(X_test)[:, 1].mean()
            else:
                prob = 0.0
        else:
            prob = 0.0
        probabilities.append(prob)

    # 各区切りでの確率を表示
    for i, prob in enumerate(probabilities):
        print(f"RSI range: {rsi_ranges[i]}-{rsi_ranges[i] + 5}, Probability: {prob:.2f}")

    # 最も確率が高い区切りを見つける
    max_prob_index = np.argmax(probabilities)
    max_prob_range = f"{rsi_ranges[max_prob_index]}-{rsi_ranges[max_prob_index] + 5}"
    max_prob = probabilities[max_prob_index]

    print(f"\nRSI range with the highest probability of price increase: {max_prob_range}")
    print(f"Probability: {max_prob:.2f}")

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