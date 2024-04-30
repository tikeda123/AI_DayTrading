import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def bayes_model(df):
    # 特徴量とターゲットを定義
    features = ['macdhist']  # 'macdhist'カラムを使用
    target = 'bb_profit_positive'

    # 'bb_profit'が0より大きい場合は1、そうでなければ0とする新しいカラムを作成
    df[target] = np.where(df['bb_profit'] > 0, 1, 0)

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

    # macdhistの値が特定の範囲にあるデータポイントに対する確率を求める
    # ここではMACDヒストグラムの値の範囲を適切に設定してください
    macdhist_test = np.array([[macdhist] for macdhist in np.linspace(-10, 10, 100)])  # 例: MACDヒストグラムの値の範囲
    probabilities = model.predict_proba(macdhist_test)
    probability_positive_bb_profit = probabilities[:, 1]  # 'bb_profit' > 0 の確率

    # 確率を表示
    for macdhist, prob in zip(np.linspace(-10, 10, 100), probability_positive_bb_profit):
        print(f"MACD Histogram: {macdhist:.2f}, Probability of bb_profit > 0: {prob:.2f}")

def main():
    # データをロードするコードをここに追加
    # df = pd.read_csv("path/to/your/data.csv")
    # 以下のデモのためのダミーデータを生成
    data = {
        'macdhist': np.random.randn(1000),
        'bb_profit': np.random.randint(-100, 100, 1000)
    }
    df = pd.DataFrame(data)
    print(df.head())
    bayes_model(df)

if __name__ == "__main__":
    main()
