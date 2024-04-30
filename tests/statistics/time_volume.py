# Re-importing necessary libraries and reloading the data, as there was an exception
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the new CSV file provided
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'

df = pd.read_csv(file_path_new)

# Convert 'date' column to datetime and extract the hour
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

# Group by the hour and calculate the average volume for each hour
hourly_volume_avg = df.groupby(df['hour']).volume.mean()

# Plotting the average volume data as a bar graph
plt.figure(figsize=(15, 6))
plt.bar(hourly_volume_avg.index, hourly_volume_avg.values)
plt.title('Average Hourly Volume of Bitcoin')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Volume')
plt.xticks(range(0, 24))  # X-axis ticks for 24 hours
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
