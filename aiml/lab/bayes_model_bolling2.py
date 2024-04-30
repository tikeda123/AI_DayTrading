import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.data_loader_db import DataLoaderDB
from aiml.naive_bayes_prediction_model import NaiveBayes_PredictionModel
from common.constants import *



def calculate_profit(entry_price,exit_price,shares,position,leverge=1):
    fee = (shares * entry_price * 0.00055)*2
    if position == 1:
        profit = shares * (exit_price - entry_price)*leverge
    elif position == 0:
        profit = shares * (entry_price - exit_price)*leverge

    profit -= fee
    return profit,fee


def simulation_bayes_model(model,test_df,leverage=1):

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
                    position = model.predict_price_direction(row)
                    shares = investment / entry_price
                    print(f"{row['start_at']}:UPPER New Position: {position} at {entry_price:.4f} on {row.name}")
                    counter += 1
                else:
                    print(36*'*')
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
    model = NaiveBayes_PredictionModel()
    df  = model.load_data_from_db('2020-08-10 00:00:00','2023-01-01 00:00:00')
    model.train(df)

    test_df = data_loader.load_data_from_period("2023-01-01", "2024-01-01", "btcusdt_240_market_data_tech")
    simulation_bayes_model(model,test_df,leverage=2)

if __name__ == "__main__":
    main()