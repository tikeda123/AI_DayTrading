import pandas as pd

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aiml/test01_lower_60.csv'

# Load the CSV file
data = pd.read_csv(file_path_new)

# Step 1: Identify rows where `bb_flag` is either 'upper' or 'lower'.
bb_rows = data[data['bb_flag'].isin(['upper', 'lower'])]

# Step 2: Confirm whether `bb_diff` in these bb rows is either <=0 or >=0.
# This step is inherent in the data structure and doesn't require a separate calculation.

# Step 3 & 4: Calculate the probabilities
# Initialize counters
next_bb_diff_less_equal_0_after_less_equal_0 = 0
next_bb_diff_greater_equal_0_after_less_equal_0 = 0
next_bb_diff_less_equal_0_after_greater_equal_0 = 0
next_bb_diff_greater_equal_0_after_greater_equal_0 = 0

# Total counts for each case
total_less_equal_0 = 0
total_greater_equal_0 = 0

# Iterate through the bb_rows
for i in range(len(bb_rows) - 1):
    current_bb_diff = bb_rows.iloc[i]['bb_diff']
    next_bb_diff = bb_rows.iloc[i + 1]['bb_diff']

    if current_bb_diff <= 0:
        total_less_equal_0 += 1
        if next_bb_diff <= 0:
            next_bb_diff_less_equal_0_after_less_equal_0 += 1
        else:
            next_bb_diff_greater_equal_0_after_less_equal_0 += 1

    else:  # current_bb_diff >= 0
        total_greater_equal_0 += 1
        if next_bb_diff <= 0:
            next_bb_diff_less_equal_0_after_greater_equal_0 += 1
        else:
            next_bb_diff_greater_equal_0_after_greater_equal_0 += 1

# Calculate probabilities
prob_less_equal_0_after_less_equal_0 = next_bb_diff_less_equal_0_after_less_equal_0 / total_less_equal_0 if total_less_equal_0 > 0 else 0
prob_greater_equal_0_after_less_equal_0 = next_bb_diff_greater_equal_0_after_less_equal_0 / total_less_equal_0 if total_less_equal_0 > 0 else 0
prob_less_equal_0_after_greater_equal_0 = next_bb_diff_less_equal_0_after_greater_equal_0 / total_greater_equal_0 if total_greater_equal_0 > 0 else 0
prob_greater_equal_0_after_greater_equal_0 = next_bb_diff_greater_equal_0_after_greater_equal_0 / total_greater_equal_0 if total_greater_equal_0 > 0 else 0

# Output probabilities
print(f'Probability of bb_diff <= 0 after a bb_diff <= 0": {prob_less_equal_0_after_less_equal_0}')
print(f'Probability of bb_diff >= 0 after a bb_diff <= 0": {prob_greater_equal_0_after_less_equal_0}')
print(f'Probability of bb_diff <= 0 after a bb_diff >= 0": {prob_less_equal_0_after_greater_equal_0}')
print(f'Probability of bb_diff >= 0 after a bb_diff >= 0": {prob_greater_equal_0_after_greater_equal_0}')




