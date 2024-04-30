import pandas as pd
import random

# 定数の設定
INITIAL_CAPITAL = 1000
PROB_ACCURACY = 0.72
LEVARGE = 3
INVESTMENT_PERCENTAGE = 1.0  # 投資する資本の割合
MAX_LOSS_PERCENTAGE = 0.03  # 最大損失割合
START_DATE = '2023-01-01'
END_DATE = '2023-06-01'
FILE_NAME_AND_PATH = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20220101000_20240120000_60_price_ml.csv'

def generate_trading_decision(rate, row):
    if pd.isna(row['bb_direction']) or pd.isna(row['bb_profit']):
        raise ValueError("Missing required data in row")

    random_flag = random.random() < rate
    positive_condition = (row['bb_direction'] == 'upper' and row['bb_profit'] >= 0)
    negative_condition = (row['bb_direction'] == 'lower' and row['bb_profit'] < 0)

    is_long_trade = positive_condition or negative_condition
    return is_long_trade if random_flag else not is_long_trade

def calculate_trade_profit(is_long_trade, investment_amount, close_price, exit_price):
    leveraged_amount = investment_amount * LEVARGE
    amount = leveraged_amount / close_price
    return (amount * exit_price - leveraged_amount) if is_long_trade else \
           (leveraged_amount - amount * exit_price)

def update_trade_statistics(statistics, profit):
    statistics['total_trades'] += 1
    statistics['profit_count'] += profit > 0
    statistics['loss_count'] += profit < 0

def simulate_trading(row, total_capital, statistics):
    if row['bb_profit'] == 0 or row['bb_direction'] not in statistics:
        return 0

    investment_amount = total_capital * INVESTMENT_PERCENTAGE
    is_long_trade = generate_trading_decision(PROB_ACCURACY, row)
    profit = calculate_trade_profit(is_long_trade, investment_amount, row['close'], row['exit_price'])
    
    update_trade_statistics(statistics[row['bb_direction']], profit)
    update_trade_statistics(statistics['overall'], profit)

    return profit

def trading_simulation(trading_data, initial_capital):
    statistics = {direction: {'profit_count': 0, 'loss_count': 0, 'total_trades': 0}
                  for direction in ['lower', 'upper', 'overall']}
    total_capital = initial_capital

    trading_data['profit'] = trading_data.apply(lambda row: simulate_trading(row, total_capital, statistics), axis=1)
    total_capital += trading_data['profit'].sum()
    total_capital = max(total_capital, initial_capital * (1 - MAX_LOSS_PERCENTAGE))

    return statistics, total_capital

def main():
    file_path = FILE_NAME_AND_PATH

    trading_data = pd.read_csv(file_path)
    trading_data['date'] = pd.to_datetime(trading_data['date'])

    # データフィルタリング
    filtered_data = trading_data[(trading_data['date'] >= START_DATE) & (trading_data['date'] <= END_DATE)]

    stats, final_capital = trading_simulation(filtered_data, INITIAL_CAPITAL)

    print("Stats:", stats)
    print("Final Capital:", final_capital)

if __name__ == "__main__":
    main()



