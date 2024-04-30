import pandas as pd

def load_csv(file_path):
    """ Load a CSV file into a pandas DataFrame. """
    return pd.read_csv(file_path)

def calculate_probability_refactored(counts):
    """ Calculate probabilities such that the sum of probabilities in each category equals 1. """
    # Define categories and their scenarios
    categories = {
        'upper_up_after_upper': ['upper_up_after_upper_up', 'upper_up_after_upper_dw'],
        'upper_up_after_lower': ['upper_up_after_lower_up', 'upper_up_after_lower_dw'],
        'upper_dw_after_upper': ['upper_dw_after_upper_up', 'upper_dw_after_upper_dw'],
        'upper_dw_after_lower': ['upper_dw_after_lower_up', 'upper_dw_after_lower_dw'],
        'lower_up_after_upper': ['lower_up_after_upper_up', 'lower_up_after_upper_dw'],
        'lower_up_after_lower': ['lower_up_after_lower_up', 'lower_up_after_lower_dw'],
        'lower_dw_after_upper': ['lower_dw_after_upper_up', 'lower_dw_after_upper_dw'],
        'lower_dw_after_lower': ['lower_dw_after_lower_up', 'lower_dw_after_lower_dw'],
        # Add more categories as needed
    }

    probabilities = {}

    for category_scenarios in categories.values():
        # Calculate the sum for each category
        category_sum = sum(counts.get(scenario, 0) for scenario in category_scenarios)

        # Calculate probabilities for each scenario within the category
        for scenario in category_scenarios:
            probabilities[scenario] = counts.get(scenario, 0) / category_sum if category_sum > 0 else 0

    return probabilities


def process_data(data):
    # Filter data for 'upper' and 'lower' bb_flag
    filtered_data = data[data['bb_flag'].isin(['upper', 'lower'])]

    # Initializing counts with all possible keys
    keys = [f"{flag}_{diff}_after_{next_flag}_{next_diff}" 
            for flag in ['upper', 'lower'] 
            for diff in ['up', 'dw'] 
            for next_flag in ['upper', 'lower'] 
            for next_diff in ['up', 'dw']]
    counts = {key: 0 for key in keys}

    # Loop through filtered data
    for i in range(len(filtered_data) - 1):
        row, row_next = filtered_data.iloc[i], filtered_data.iloc[i+1]
        key = f"{row['bb_flag']}_{ 'up' if row['bb_diff'] >= 0 else 'dw' }_after_{row_next['bb_flag']}_{ 'up' if row_next['bb_diff'] >= 0 else 'dw' }"
        counts[key] += 1


    # Calculating probabilities
    probabilities = calculate_probability_refactored(counts)

    return counts, probabilities

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'
data = load_csv(file_path_new)

# Process data
counts, probabilities = process_data(data)

# Printing results
for key, count in counts.items():
    print(f"{key}: {count}, Probability: {probabilities[key]:.2f}")

# Note: The keys in the counts dictionary are dynamically generated based on the conditions in the data.
# This may require adjustment depending on the specific conditions and format of the input data.





 



