import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.data_loader_db import DataLoaderDB
from common.constants import *

def predict_price_direction(model, row):
    # 特徴量を取得
    rsi_prev = row["rsi_prev"]
    macdhist_prev = row["macdhist_prev"]

    # 特徴量を2次元配列に変換
    features = np.array([[rsi_prev, macdhist_prev]])

    # 予測を実行
    prediction = model.predict(features)
    return prediction[0]

def calculate_profit(entry_price, exit_price, investment, position):
    if position == 1:
        shares = investment / entry_price
        profit = shares * (exit_price - entry_price)
    elif position == 0:
        shares = investment / entry_price
        profit = shares * (entry_price - exit_price)
    else:
        raise ValueError(f"Invalid position: {position}. Position must be 'LONG' or 'SHORT'.")
    return profit

def simulation_bayes_model(df, leverage=1):
    features = [COLUMN_RSI, COLUMN_MACDHIST]
    target = 'price_increased'

    # 特徴量を1期ずらす
    df[f"{COLUMN_RSI}_prev"] = df[COLUMN_RSI].shift(1)
    df[f"{COLUMN_MACDHIST}_prev"] = df[COLUMN_MACDHIST].shift(1)

    # 修正した特徴量を使用
    features = [f"{COLUMN_RSI}_prev", f"{COLUMN_MACDHIST}_prev"]

    df[target] = np.where(df[COLUMN_CLOSE] > df[COLUMN_EMA].shift(1), 1, 0)
    df = df.dropna()

    initial_capital = 1000
    capital = initial_capital
    capital_history = [initial_capital]

    df['entry_price'] = df[COLUMN_EMA].shift(1)
    df['exit_price'] = df[COLUMN_CLOSE]
    df['rsi_prev'] = df['rsi'].shift(1)
    df['macdhist_prev'] = df['macdhist'].shift(1)

    for i in range(1, len(df)):
        train_df = df[:i]
        test_row = df.iloc[i]

        if len(train_df) > 0:  # 訓練データが空でない場合のみ学習と予測を実行
            X_train = train_df[features].values
            y_train = train_df[target].values

            clf = GaussianNB()
            clf.fit(X_train, y_train)

            if not np.isnan(test_row['entry_price']):
                investment = min(capital, 1000)
                if test_row['low'] <= test_row['entry_price'] <= test_row['high']:
                    position = predict_price_direction(clf, test_row)
                    profit = calculate_profit(test_row['entry_price'], test_row['exit_price'], investment, position)
                    capital += profit * leverage
        capital_history.append(capital)

    print(f"\nFinal Capital: {capital:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(capital_history)), capital_history)
    plt.xlabel('Trade')
    plt.ylabel('Capital')
    plt.title('Capital Over Time')
    plt.grid(True)
    plt.show()

def main():
    data_loader = DataLoaderDB()
    df = data_loader.load_data_from_period("2020-01-01", "2024-04-01", "btcusdt_240_market_data_tech")
    simulation_bayes_model(df, leverage=3)

if __name__ == "__main__":
    main()