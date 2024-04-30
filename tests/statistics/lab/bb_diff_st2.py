import pandas as pd

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'

# Load the CSV file
data = pd.read_csv(file_path_new)


# Filter rows where bb_flag is either 'upper' or 'lower'
filtered_data = data[data['bb_flag'].isin(['upper', 'lower'])]

# Counting the occurrences of 'upper' and 'lower'
upper_count = filtered_data['bb_flag'].value_counts().get('upper', 0)
lower_count = filtered_data['bb_flag'].value_counts().get('lower', 0)

# Create a new column for the next bb_flag
filtered_data['next_bb_flag'] = filtered_data['bb_flag'].shift(-1)

# Using groupby to count the occurrences of each combination
combination_counts = filtered_data.groupby(['bb_flag', 'next_bb_flag']).size()

# Extracting counts for each combination
uper_after_lower_count = combination_counts.get(('upper', 'lower'), 0)
uper_after_upper_count = combination_counts.get(('upper', 'upper'), 0)
lower_after_lower_count = combination_counts.get(('lower', 'lower'), 0)
lower_after_upper_count = combination_counts.get(('lower', 'upper'), 0)

# Printing the counts
print(f'uper_after_lower_count: {uper_after_lower_count}')
print(f'uper_after_upper_count: {uper_after_upper_count}')
print(f'lower_after_lower_count: {lower_after_lower_count}')
print(f'lower_after_upper_count: {lower_after_upper_count}')




 



