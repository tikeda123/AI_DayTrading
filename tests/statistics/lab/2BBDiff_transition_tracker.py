import pandas as pd

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aiml/test01_lower_60.csv'

data = pd.read_csv(file_path_new)

# Create two shifted columns for 'bb_diff'
data['bb_diff_shifted_1'] = data['bb_diff'].shift(-1)
data['bb_diff_shifted_2'] = data['bb_diff'].shift(-2)

# Define the conditions for each 3-step transition type
conditions = {
    '>=0 to >=0 to >=0': (data['bb_diff'] >= 0) & (data['bb_diff_shifted_1'] >= 0) & (data['bb_diff_shifted_2'] >= 0),
    '>=0 to >=0 to <=0': (data['bb_diff'] >= 0) & (data['bb_diff_shifted_1'] >= 0) & (data['bb_diff_shifted_2'] <= 0),
    '>=0 to <=0 to >=0': (data['bb_diff'] >= 0) & (data['bb_diff_shifted_1'] <= 0) & (data['bb_diff_shifted_2'] >= 0),
    '>=0 to <=0 to <=0': (data['bb_diff'] >= 0) & (data['bb_diff_shifted_1'] <= 0) & (data['bb_diff_shifted_2'] <= 0),
    '<=0 to >=0 to >=0': (data['bb_diff'] <= 0) & (data['bb_diff_shifted_1'] >= 0) & (data['bb_diff_shifted_2'] >= 0),
    '<=0 to >=0 to <=0': (data['bb_diff'] <= 0) & (data['bb_diff_shifted_1'] >= 0) & (data['bb_diff_shifted_2'] <= 0),
    '<=0 to <=0 to >=0': (data['bb_diff'] <= 0) & (data['bb_diff_shifted_1'] <= 0) & (data['bb_diff_shifted_2'] >= 0),
    '<=0 to <=0 to <=0': (data['bb_diff'] <= 0) & (data['bb_diff_shifted_1'] <= 0) & (data['bb_diff_shifted_2'] <= 0)
}

# Initialize a dictionary to store results
results = {}

# Calculate counts and probabilities for each condition
for key, condition in conditions.items():
    # Count the number of rows that meet the condition
    count = len(data[condition])

    # Calculate the total number of occurrences for the denominator
    total = len(data)

    # Calculate probability
    probability = count / total if total > 0 else 0
    results[key] = {'Count': count, 'Probability': probability}

# Display the results
for transition, result in results.items():
    print(f"{transition}: Count = {result['Count']}, Probability = {result['Probability']}")

