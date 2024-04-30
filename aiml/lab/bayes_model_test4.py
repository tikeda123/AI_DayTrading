import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os, sys

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)
from common.data_loader_db import DataLoaderDB
from common.constants import *

from common.data_loader_db import DataLoaderDB

def bayes_model(df):
    # 特徴量とターゲットを定義
    features = ['rsi']  # COLUMN_RSI に該当するカラム名に置き換えてください
    target = 'bb_profit_positive'
    # 'bb_profit'が0より大きい場合は1、そうでなければ0とする新しいカラムを作成
    df[target] = np.where(df['bb_profit'] > 0, 1, 0)  # COLUMN_BB_PROFIT に該当するカラム名に置き換えてください

    # 特徴量とターゲットを分割
    X = df[features]
    y = df[target]

    # データを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ガウシアンナイーブベイズモデルを初期化して学習
    model = GaussianNB()
    model.fit(X_train, y_train)

    # テストデータに対する予測精度の計算
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # RSIが50以上のデータポイントに対する確率を求める
    rsi_test = np.array([[rsi] for rsi in range(20, 55)])  # RSIが50から100までの値
    probabilities = model.predict_proba(rsi_test)
    probability_above_50 = probabilities[:, 1]  # 'bb_profit' > 0 の確率

    # 確率を表示
    for rsi, prob in zip(range(20,55), probability_above_50):
        print(f"RSI: {rsi}, Probability of bb_profit > 0: {prob:.2f}")

def main():
    # ここにデータをロードするコードを追加
    # 例: df = pd.read_csv("path/to/your/data.csv")
    data_loader = DataLoaderDB()
    filename = "data_ml/BTCUSDT_20200101_20230101_60_price_lower_mlnonts.csv"
    df = data_loader.load_data_from_csv(filename)
    print(df)
    bayes_model(df)

if __name__ == "__main__":
    main()
