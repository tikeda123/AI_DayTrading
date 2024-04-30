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
from common.constants import COLUMN_CLOSE, COLUMN_RSI, COLUMN_MACDHIST

def bayes_model(df):
    # 特徴量とターゲットを定義
    feature = COLUMN_RSI
    target = 'price_direction'

    # 価格が5期間の間に上昇したかどうかを判断
    df['price_increased'] = np.where(df[COLUMN_CLOSE].rolling(window=5).max().shift(-5) > df[COLUMN_CLOSE], 1, 0)

    # RSIを5の区切りごとに分類
    rsi_bins = pd.cut(df[COLUMN_RSI], bins=range(0, 101, 5), labels=False, right=False)

    # 各区切りでの確率を計算（事前確率）
    rsi_ranges = range(0, 100, 5)
    prior_probabilities = []

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
        prior_probabilities.append(prob)

    # macdhistがプラスに転じた時の価格が上がる確率を計算（事後確率）
    df['macdhist_turned_positive'] = np.where((df[COLUMN_MACDHIST] > 0) & (df[COLUMN_MACDHIST].shift(1) <= 0), 1, 0)

    posterior_probabilities = []

    for rsi_start, prior_prob in zip(rsi_ranges, prior_probabilities):
        rsi_end = rsi_start + 5
        mask = (df[COLUMN_RSI] >= rsi_start) & (df[COLUMN_RSI] < rsi_end) & (df['macdhist_turned_positive'] == 1)
        y = df.loc[mask, 'price_increased'].values

        if len(y) > 0:
            prob = y.mean()
        else:
            prob = 0.0
        posterior_probabilities.append(prob)

    # 結果を表示
    print("Prior Probabilities (RSI):")
    for i, prob in enumerate(prior_probabilities):
        print(f"RSI range: {rsi_ranges[i]}-{rsi_ranges[i] + 5}, Probability: {prob:.2f}")

    print("\nPosterior Probabilities (RSI and MACDHIST turned positive):")
    for i, prob in enumerate(posterior_probabilities):
        print(f"RSI range: {rsi_ranges[i]}-{rsi_ranges[i] + 5}, Probability: {prob:.2f}")

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