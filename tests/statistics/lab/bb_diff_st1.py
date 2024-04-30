import pandas as pd

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'

# Load the CSV file
data = pd.read_csv(file_path_new)

# Step 1: Find the first row where bb_flag=='upper' & bb_diff>=0.
filtered_data = data[data['bb_flag'].isin(['upper', 'lower'])]

# Counting the occurrences of 'upper' and 'lower'
upper_count = filtered_data[filtered_data['bb_flag'] == 'upper'].shape[0]
lower_count = filtered_data[filtered_data['bb_flag'] == 'lower'].shape[0]

print(f'upper_count: {upper_count}, lower_count: {lower_count}')


uper_after_lower_count = 0
uper_after_upper_count = 0
lower_after_lower_count = 0
lower_after_upper_count = 0

for i in range(len(filtered_data)-1):
    row = filtered_data.iloc[i]
    row_next = filtered_data.iloc[i+1]
    if row['bb_flag'] == 'upper' and row_next['bb_flag'] == 'lower':
        uper_after_lower_count += 1
    elif row['bb_flag'] == 'upper' and row_next['bb_flag'] == 'upper':
        uper_after_upper_count += 1
    elif row['bb_flag'] == 'lower' and row_next['bb_flag'] == 'lower':
        lower_after_lower_count += 1
    elif row['bb_flag'] == 'lower' and row_next['bb_flag'] == 'upper':
        lower_after_upper_count += 1

print(f'uper_after_lower_count: {uper_after_lower_count}')
print(f'uper_after_upper_count: {uper_after_upper_count}')
print(f'lower_after_lower_count: {lower_after_lower_count}')
print(f'lower_after_upper_count: {lower_after_upper_count}')

 



