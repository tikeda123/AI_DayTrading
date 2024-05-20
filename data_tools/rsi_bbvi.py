# Define the date range for filtering
import numpy as np
import pandas as pd

# Define the ranges for RSI and BBVI to search through
data = pd.read_csv('data.csv')
rsi_min = data['rsi'].min()
rsi_max = data['rsi'].max()
bbvi_min = data['bbvi'].min()
bbvi_max = data['bbvi'].max()

# Define the step sizes for RSI and BBVI
rsi_step = 1
bbvi_step = 0.1

# Initialize variables to store the best range and the maximum bb_profit sum
best_rsi_range = None
best_bbvi_range = None
max_bb_profit_sum = -np.inf

start_date = "2023-01-01"
end_date = "2024-01-01"

data = pd.read_csv('data.csv')
# Convert 'date' column to datetime format for filtering
data['date'] = pd.to_datetime(data['date'])

# Filter data by the specified date range
filtered_data_by_date = data[(data['date'] >= start_date) & (data['date'] < end_date)]

# Define the step sizes for RSI and BBVI for this specific threshold search
rsi_step = 1
bbvi_step = 0.1

# Initialize variables to store the best range and the maximum bb_profit sum
best_rsi_threshold = None
best_bbvi_threshold = None
max_bb_profit_sum_threshold = -np.inf

# Perform grid search for thresholds within the date range
for rsi_threshold in np.arange(rsi_min, rsi_max, rsi_step):
    for bbvi_threshold in np.arange(bbvi_min, bbvi_max, bbvi_step):
        # Filter the data for the current thresholds within the date range
        filtered_data_threshold = filtered_data_by_date[(filtered_data_by_date['rsi'] >= rsi_threshold) &
                                                        (filtered_data_by_date['bbvi'] >= bbvi_threshold)]

        # Calculate the sum of bb_profit for the filtered data
        bb_profit_sum_threshold = filtered_data_threshold['bb_profit'].sum()

        # Check if this is the best sum we've found
        if bb_profit_sum_threshold > max_bb_profit_sum_threshold:
            max_bb_profit_sum_threshold = bb_profit_sum_threshold
            best_rsi_threshold = rsi_threshold
            best_bbvi_threshold = bbvi_threshold

best_rsi_threshold, best_bbvi_threshold, max_bb_profit_sum_threshold
