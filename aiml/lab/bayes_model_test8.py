import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
import numpy as np
from scipy.stats import bernoulli

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.data_loader_db import DataLoaderDB
from common.constants import COLUMN_CLOSE, COLUMN_RSI

def bayes_model(df, rsi_min, rsi_max):
    features = [COLUMN_RSI]
    target = 'price_increased'

    df[target] = np.where(df[COLUMN_CLOSE].rolling(window=5).max().shift(-5) > df[COLUMN_CLOSE], 1, 0)

    rsi_signal = df[(df[COLUMN_RSI] > rsi_min) & (df[COLUMN_RSI] <= rsi_max)]
    success_count = rsi_signal[target].sum()
    total_count = len(rsi_signal)
    theta = success_count / total_count

    simulation_count = 1000
    simulated_data = bernoulli.rvs(theta, size=simulation_count)

    return simulated_data

def main():
    data_loader = DataLoaderDB()
    df = data_loader.load_data_from_db("btcusdt_60_market_data_tech")

    rsi_ranges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i, (rsi_min, rsi_max) in enumerate(rsi_ranges):
        simulated_data = bayes_model(df, rsi_min, rsi_max)

        axs[i].hist(simulated_data, bins=2, rwidth=0.8)
        axs[i].set_xticks([0.25, 0.75])
        axs[i].set_xticklabels(['Price Decrease', 'Price Increase'])
        axs[i].set_xlabel('Stock Price Change')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'RSI {rsi_min} to {rsi_max}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()