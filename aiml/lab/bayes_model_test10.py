import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.data_loader_db import DataLoaderDB
from common.constants import COLUMN_CLOSE, COLUMN_UPPER_BAND2, COLUMN_LOWER_BAND2

def calculate_profit(entry_price, exit_price, investment, position):
    if position == 'LONG':
        shares = investment / entry_price
        profit = shares * (exit_price - entry_price)
    elif position == 'SHORT':
        shares = investment / entry_price
        profit = shares * (entry_price - exit_price)
    else:
        raise ValueError(f"Invalid position: {position}. Position must be 'LONG' or 'SHORT'.")
    return profit

def trading_simulation(df):
    initial_investment = 1000
    investment_amount = 1000
    total_assets = initial_investment
    position = None
    entry_price = None
    profits = []
    investment_counts = []

    for i in range(len(df)):
        close_price = df[COLUMN_CLOSE].iloc[i]
        upper_band = df[COLUMN_UPPER_BAND2].iloc[i]
        lower_band = df[COLUMN_LOWER_BAND2].iloc[i]

        if position is None:
            if close_price < lower_band:
                position = 'LONG'
                entry_price = close_price
                investment = min(investment_amount, total_assets)
                total_assets -= investment
            elif close_price > upper_band:
                position = 'SHORT'
                entry_price = close_price
                investment = min(investment_amount, total_assets)
                total_assets -= investment
        else:
            if position == 'LONG' and close_price > upper_band:
                profit = calculate_profit(entry_price, close_price, investment, position)
                total_assets += investment + profit
                profits.append(profit)
                investment_counts.append(len(profits))
                position = None
                entry_price = None
            elif position == 'SHORT' and close_price < lower_band:
                profit = calculate_profit(entry_price, close_price, investment, position)
                total_assets += investment + profit
                profits.append(profit)
                investment_counts.append(len(profits))
                position = None
                entry_price = None

    plt.figure(figsize=(10, 6))
    plt.plot(investment_counts, [initial_investment + sum(profits[:i+1]) for i in range(len(profits))])
    plt.xlabel('Investment Count')
    plt.ylabel('Total Assets (USD)')
    plt.title('Trading Simulation Results')
    plt.grid(True)
    plt.show()

def main():
    data_loader = DataLoaderDB()
    df = data_loader.load_data_from_period("2023-01-01","2024-01-01","btcusdt_60_market_data_tech")
    trading_simulation(df)

if __name__ == '__main__':
    main()