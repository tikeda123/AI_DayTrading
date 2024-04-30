import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Since the VPVR focuses on the volume distribution across different price levels,
# we need to aggregate the volume for each price level.
# We'll consider the 'close' price for simplicity.


file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20231224000_2024011110_240.csv'

btc_data = pd.read_csv(file_path)

# Define the number of bins for the price levels
num_bins = 50

# Find the min and max close price to define the range of our bins
min_price = btc_data['close'].min()
max_price = btc_data['close'].max()
bins = np.linspace(min_price, max_price, num_bins)

# Aggregate the volume for each bin
volume_per_bin = btc_data.groupby(pd.cut(btc_data['close'], bins=bins))['volume'].sum()

# Prepare the data for plotting
bins_centers = (bins[:-1] + bins[1:]) / 2
volumes = volume_per_bin.values

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(bins_centers, volumes, width=bins[1] - bins[0], color='blue', alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Volume')
plt.title('Volume Profile Visible Range (VPVR) for Bitcoin')
plt.show()
