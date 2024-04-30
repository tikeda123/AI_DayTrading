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


def naive_bayes_model(df):
    features = [COLUMN_RSI, COLUMN_MACDHIST]
    target = 'price_increased'

    # 特徴量を1期ずらす
    df[f"{COLUMN_RSI}_prev"] = df[COLUMN_RSI].shift(1)
    df[f"{COLUMN_MACDHIST}_prev"] = df[COLUMN_MACDHIST].shift(1)

    # 修正した特徴量を使用
    features = [f"{COLUMN_RSI}_prev", f"{COLUMN_MACDHIST}_prev"]

    df[target] = np.where(df[COLUMN_CLOSE] > df[COLUMN_EMA].shift(1), 1, 0)
    df = df.dropna()

    train_size = int(len(df) * 0.9)
    train_df = df[:train_size]
    test_df = df[train_size:]

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nProbabilities:")
    print(f" Price increase: {y_pred_proba[:, 1].mean():.4f}")
    print(f" Price decrease: {y_pred_proba[:, 0].mean():.4f}")

    return clf

def simulation_bayes_model(clf,test_df,leverage=1):

    initial_capital = 1000
    capital = initial_capital
    capital_history = [initial_capital]

    win_count = 0
    total_trades = 0

    test_df['entry_price'] = test_df[COLUMN_EMA].shift(1)
    test_df['exit_price'] = test_df[COLUMN_CLOSE]
    test_df['rsi_prev'] = test_df['rsi'].shift(1)
    test_df['macdhist_prev'] = test_df['macdhist'].shift(1)
    test_df['upper_over'] = np.where(test_df[COLUMN_CLOSE].shift(1) > test_df[COLUMN_UPPER_BAND2].shift(1), 1, 0)
    test_df['lower_over'] = np.where(test_df[COLUMN_CLOSE].shift(1) < test_df[COLUMN_LOWER_BAND2].shift(1), 1, 0)

    for _, row in test_df.iterrows():
        if not np.isnan(row['entry_price']):
            investment = min(capital, 1000)

            if row['upper_over'] == 1 or row['lower_over'] == 1:
                if row['low'] <= row['entry_price'] <= row['high']:
                    position = predict_price_direction(clf, row)
                    profit = calculate_profit(row['entry_price'], row['exit_price'], investment, position)
                    capital += profit*leverage
                    capital_history.append(capital)

                    total_trades += 1
                    if profit > 0:
                        win_count += 1

    print(f"\nFinal Capital: {capital:.4f}")
    print(f"Win Rate: {win_count / total_trades:.2%}")

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
    df = data_loader.load_data_from_period("2020-01-01", "2023-01-01", "ethusdt_60_market_data_tech")
    clf = naive_bayes_model(df)
    test_df = data_loader.load_data_from_period("2023-01-01", "2024-01-01", "ethusdt_60_market_data_tech")
    simulation_bayes_model(clf,test_df,leverage=3)

if __name__ == "__main__":
    main()