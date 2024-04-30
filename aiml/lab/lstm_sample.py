

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np



# ファイルからデータを読み込む
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aiml/BTCUSDT_20220101000_20231231000_240.csv' # ファイルパスを適宜変更してください
data = pd.read_csv(file_path)
# Selecting relevant columns for features
feature_columns = ['close', 'oi', 'volume', 'upper2', 'middle', 'lower2']

# Filtering rows based on the condition specified
filtered_data = data[(data['bb_direction'].isin(['upper', 'lower'])) & (data['bb_profit'] != 0)]

# Initializing list to hold sequences and corresponding targets
sequences = []
targets = []

# Looping through filtered data to create sequences
for i in range(len(filtered_data)):
    start_index = filtered_data.index[i]
    end_index = start_index + 8

    # Ensure the sequence is within the bounds of the dataset
    if end_index > len(data):
        break

    # Extract the sequence and corresponding target
    sequence = data.loc[start_index:end_index-1, feature_columns].values
    target = data.loc[end_index-1, 'bb_profit'] > 0  # True if bb_profit > 0, False otherwise

    sequences.append(sequence)
    targets.append(target)

# Converting lists to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sequences = np.array([scaler.fit_transform(seq) for seq in sequences])

# Displaying the shape of the sequences and a sample for validation
scaled_sequences.shape, scaled_sequences[0], targets[0]

print(scaled_sequences.shape)
print(scaled_sequences[0])
