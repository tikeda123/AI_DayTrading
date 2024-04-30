import pandas as pd
import matplotlib.pyplot as plt

file_path_new = 'oi_test01_60.csv'
data_new = pd.read_csv(file_path_new)

# 最後の60時間のデータを選択
data_new = data_new.iloc[-60:]

# Check if 'oi' and 'close' columns exist in the new dataset
if 'oi' in data_new.columns and 'close' in data_new.columns:
    plt.figure(figsize=(12, 6))

    # Calculating 50-hour moving average for oi and close
    moving_avg_oi = data_new['oi'].rolling(window=5).mean()
    moving_avg_close = data_new['close'].rolling(window=1).mean()
    # Plotting 50-hour moving averages
    plt.plot(moving_avg_oi, label='50-hour MA of oi', color='red')
    plt.plot(moving_avg_close, label='50-hour MA of close', color='blue')

    plt.title('50-Hour Moving Average of OI and Close Price (Last 60 Hours)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    error_message_new_dataset = "One or both of the columns 'oi' and 'close' do not exist in the new dataset."
    print(error_message_new_dataset)

