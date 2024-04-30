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
    rsi = row["rsi"]
    macdhist = row["macdhist"]

    # 特徴量を2次元配列に変換
    features = np.array([[rsi, macdhist]])

    # 予測を実行
    prediction = model.predict(features)
    return prediction[0]



def calculate_profit(entry_price,exit_price,shares,position,leverge=1):
    fee = shares * entry_price * 0.00055*leverge
    if position == 1:
        profit = shares * (exit_price - entry_price)*leverge
    elif position == 0:
        profit = shares * (entry_price - exit_price)*leverge

    profit -= fee
    return profit,fee


def naive_bayes_model(df):
    features = [COLUMN_RSI, COLUMN_MACDHIST]
    target = 'price_increased'

    # 特徴量を1期ずらす
    df[f"{COLUMN_RSI}_prev"] = df[COLUMN_RSI]
    df[f"{COLUMN_MACDHIST}_prev"] = df[COLUMN_MACDHIST]

    # 修正した特徴量を使用
    features = [f"{COLUMN_RSI}_prev", f"{COLUMN_MACDHIST}_prev"]

    df[target] = np.where(df[COLUMN_CLOSE].shift(-1) > df[COLUMN_EMA], 1, 0)
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
def buy_asset(row,):
    pass

def simulation_bayes_model(clf,test_df,leverage=1):

    initial_capital = 1000
    capital = initial_capital
    capital_history = [initial_capital]

    test_df["entry_price"] = test_df[COLUMN_EMA].shift(1)
    win_count = 0
    total_trades = 0
    position = 0
    shares = 0
    entry_price = 0
    counter = 0
    win_profit = 0
    loss_profit = 0

    for _, row in test_df.iterrows():
        if not np.isnan(row[COLUMN_CLOSE]):
            investment = min(capital, 1000)

            if  shares == 0 and counter == 0:
                entry_price = row["entry_price"]
                if row[COLUMN_HIGH] >  entry_price and  entry_price  > row[COLUMN_LOW]:
                    position = predict_price_direction(clf, row)
                    shares = investment / entry_price
                    print(f"{row['start_at']}:UPPER New Position: {position} at {entry_price:.4f} on {row.name}")
                    counter += 1
                else:
                    counter = 0
                    shares = 0
                    entry_price = 0

            elif shares != 0 and counter == 1:
                profit,fee = calculate_profit(entry_price, row[COLUMN_CLOSE], shares, position, leverage)
                capital += profit
                capital_history.append(capital)

                total_trades += 1

                if profit > 0:
                    win_count += 1
                    win_profit += profit
                else:
                    loss_profit += profit

                shares = 0
                counter = 0
                entry_price = 0
                print(f"{row['start_at']}: profit: {profit:.4f}, fee: {fee},capital: {capital:.4f},exit_price:{row[COLUMN_CLOSE]:.4f} on {row.name}")
                print("")


    print(f"\nFinal Capital: {capital:.4f}")
    print(f"Win Rate: {win_count / total_trades:.2%}")
    print(f"Win Profit: {win_profit:.4f}, Loss Profit: {loss_profit:.4f}")
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
    df = data_loader.load_data_from_period("2020-01-01", "2024-01-01", "btcusdt_60_market_data_tech")
    clf = naive_bayes_model(df)
    test_df = data_loader.load_data_from_period("2024-01-01", "2024-02-01", "btcusdt_60_market_data_tech")
    simulation_bayes_model(clf,test_df,leverage=4)

if __name__ == "__main__":
    main()