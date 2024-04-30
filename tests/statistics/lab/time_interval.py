import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'

# Load the CSV file
data = pd.read_csv(file_path_new)


## Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter rows where bb_flag is 'upper' or 'lower'
filtered_data = data[data['bb_flag'].isin(['upper', 'lower'])]

# Calculate the time difference between each filtered row in hours
filtered_data['time_diff_hours'] = filtered_data['date'].diff().dt.total_seconds() / 3600

# Drop the first NaN value
filtered_data = filtered_data.dropna(subset=['time_diff_hours'])

# Extract hour from the date
filtered_data['hour'] = filtered_data['date'].dt.hour

# Group data by hour and plot distribution for each hour
plt.figure(figsize=(15, 8))
sns.histplot(data=filtered_data, x='time_diff_hours', bins=24, kde=True)
plt.title("Hourly Distribution of Time Intervals Between 'upper' or 'lower' bb_flag Events")
plt.xlabel('Time Interval (hours)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
