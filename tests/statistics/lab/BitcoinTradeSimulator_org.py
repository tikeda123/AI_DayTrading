# Importing required libraries
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def generate_random_bit(rate):
    """
    Generates a random bit (1 or 0) based on the specified rate.

    Parameters:
    rate (float): The probability of generating a 1. Should be a value between 0 and 1.

    Returns:
    int: 1 or 0, where 1 is generated with the probability specified by 'rate'.
    """
    return 1 if random.random() < rate else 0

# Reading the provided CSV file
file_path = 'oi_test01_240.csv'
trading_data_uploaded = pd.read_csv(file_path)

# Convert 'date' column to datetime objects for plotting
trading_data_uploaded['date'] = pd.to_datetime(trading_data_uploaded['date'])

# Re-define the initial capital, investment percentage, and max loss percentage
initial_capital = 2000  # Initial capital
investment_percentage = 0.5  # 50% of the capital will be used for trading
max_loss_percentage = 0.03  # Maximum loss percentage set to 3%

# Re-initialize capital for the simulation
total_capital_uploaded = initial_capital

# Initialize lists to track capital and dates over time
capital_over_time = []
dates_over_time = []

# Simulate trading with stop-loss concept
for index, row in trading_data_uploaded.iterrows():
    investment_amount = total_capital_uploaded * investment_percentage
    remaining_amount = total_capital_uploaded - investment_amount
    initial_investment_amount = investment_amount

    if row['bb_flag'] == 'lower':
        long_probability = generate_random_bit(0.6)
        if long_probability == 1:
            amount = investment_amount / row['close']
            investment_amount = amount * row['exit_price']
        elif long_probability == 0:
            amount = investment_amount / row['close']
            investment_amount += (investment_amount / row['close'] * (row['close'] - row['exit_price']))

    elif row['bb_flag'] == 'upper':
        long_probability = generate_random_bit(0.4)
        if long_probability == 0:
            amount = investment_amount / row['close']
            investment_amount += (investment_amount / row['close'] * (row['close'] - row['exit_price']))
        elif long_probability == 1:
            amount = investment_amount / row['close']
            investment_amount = amount * row['exit_price']

    # Apply stop-loss
    if investment_amount < initial_investment_amount * (1 - max_loss_percentage):
        investment_amount = initial_investment_amount * (1 - max_loss_percentage)

    # Update total capital and record the capital and date
    total_capital_uploaded = investment_amount + remaining_amount
    capital_over_time.append(total_capital_uploaded)
    dates_over_time.append(row['date'])

# Plotting the capital over time
plt.figure(figsize=(12, 6))
plt.plot(dates_over_time, capital_over_time, marker='o')
plt.title('Capital Over Time')
plt.xlabel('Time (Date)')
plt.ylabel('Capital')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gcf().autofmt_xdate() # Beautify the x-labels
plt.show()

