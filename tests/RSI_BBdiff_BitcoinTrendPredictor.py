import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Step 1: Load the Data
btc_data = pd.read_csv('test02_60.csv')  # Replace with your file path

# Step 2: Create Subsets Based on 'bb_diff'
group1 = btc_data[btc_data['bb_diff'] > 0]  # 'bb_diff' > 0
group2 = btc_data[btc_data['bb_diff'] < 0]  # 'bb_diff' < 0

# Step 3: Descriptive Statistics for 'rsi'
print("Group 1 (bb_diff > 0) RSI Statistics:")
print(group1['bb_diff'].describe())
print("\nGroup 2 (bb_diff < 0) RSI Statistics:")
print(group2['bb_diff'].describe())

# Step 4: Visualize the Distribution of 'rsi' in Both Groups
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(group1['bb_diff'], kde=True)
plt.title('RSI Distribution for bb_diff > 0')

plt.subplot(1, 2, 2)
sns.histplot(group2['bb_diff'], kde=True)
plt.title('RSI Distribution for bb_diff < 0')
plt.show()

# Step 5: Correlation Analysis
correlation, _ = pearsonr(btc_data['bb_diff'], btc_data['bb_diff'])
print(f"Correlation coefficient between RSI and bb_diff: {correlation}")
