import pandas as pd

# Load the CSV file
file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/test01_60.csv'

# Load the CSV file
data = pd.read_csv(file_path_new)

# Step 1: Find the first row where bb_flag=='upper' & bb_diff>=0.
filtered_data = data[data['bb_flag'].isin(['upper', 'lower'])]


upper_up_after_lower_up_count = 0
upper_up_after_lower_dw_count = 0
upper_dw_after_lower_up_count = 0
upper_dw_after_lower_dw_count = 0

lower_up_after_upper_up_count = 0
lower_up_after_upper_dw_count = 0
lower_dw_after_upper_up_count = 0
lower_dw_after_upper_dw_count = 0

upper_up_after_upper_up_count = 0
upper_up_after_upper_dw_count = 0
upper_dw_after_upper_up_count = 0
upper_dw_after_upper_dw_count = 0

lower_up_after_lower_up_count = 0
lower_up_after_lower_dw_count = 0
lower_dw_after_lower_up_count = 0
lower_dw_after_lower_dw_count = 0




for i in range(len(filtered_data)-1):
    row = filtered_data.iloc[i]
    row_next = filtered_data.iloc[i+1]
    if  ( row['bb_flag'] == 'upper' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] >= 0 ):
        upper_up_after_lower_up_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] <= 0 ):
        upper_up_after_lower_dw_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] >= 0 ):
        upper_dw_after_lower_up_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] <= 0 ):
        upper_dw_after_lower_dw_count += 1

    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] >= 0 ):
        lower_up_after_upper_up_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] <= 0 ):
        lower_up_after_upper_dw_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] >= 0 ):
        lower_dw_after_upper_up_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] <= 0 ):
        lower_dw_after_upper_dw_count += 1

    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] >= 0 ):
        upper_up_after_upper_up_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] <= 0 ):
        upper_up_after_upper_dw_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] >= 0 ):
        upper_dw_after_upper_up_count += 1
    elif ( row['bb_flag'] == 'upper' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'upper' and row_next['bb_diff'] <= 0 ):
        upper_dw_after_upper_dw_count += 1

    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] >= 0 ):
        lower_up_after_lower_up_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] >= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] <= 0 ):
        lower_up_after_lower_dw_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] >= 0 ):
        lower_dw_after_lower_up_count += 1
    elif ( row['bb_flag'] == 'lower' and row['bb_diff'] <= 0 ) and ( row_next['bb_flag'] == 'lower' and row_next['bb_diff'] <= 0 ):
        lower_dw_after_lower_dw_count += 1




sum_upper_up = upper_up_after_lower_up_count + upper_up_after_lower_dw_count
upper_up_after_lower_up_probability = upper_up_after_lower_up_count / sum_upper_up if sum_upper_up > 0 else 0
upper_up_after_lower_dw_probability = upper_up_after_lower_dw_count / sum_upper_up if sum_upper_up > 0 else 0

sum_upper_dw = upper_dw_after_lower_up_count + upper_dw_after_lower_dw_count
upper_dw_after_lower_up_probability = upper_dw_after_lower_up_count / sum_upper_dw if sum_upper_dw > 0 else 0
upper_dw_after_lower_dw_probability = upper_dw_after_lower_dw_count / sum_upper_dw if sum_upper_dw > 0 else 0

sum_lower_up = lower_up_after_upper_up_count + lower_up_after_upper_dw_count
lower_up_after_upper_up_probability = lower_up_after_upper_up_count / sum_lower_up if sum_lower_up > 0 else 0
lower_up_after_upper_dw_probability = lower_up_after_upper_dw_count / sum_lower_up if sum_lower_up > 0 else 0

sum_lower_dw = lower_dw_after_upper_up_count + lower_dw_after_upper_dw_count
lower_dw_after_upper_up_probability = lower_dw_after_upper_up_count / sum_lower_dw if sum_lower_dw > 0 else 0
lower_dw_after_upper_dw_probability = lower_dw_after_upper_dw_count / sum_lower_dw if sum_lower_dw > 0 else 0

sum_upper_up = upper_up_after_upper_up_count + upper_up_after_upper_dw_count
upper_up_after_upper_up_probability = upper_up_after_upper_up_count / sum_upper_up if sum_upper_up > 0 else 0
upper_up_after_upper_dw_probability = upper_up_after_upper_dw_count / sum_upper_up if sum_upper_up > 0 else 0

sum_upper_dw = upper_dw_after_upper_up_count + upper_dw_after_upper_dw_count
upper_dw_after_upper_up_probability = upper_dw_after_upper_up_count / sum_upper_dw if sum_upper_dw > 0 else 0
upper_dw_after_upper_dw_probability = upper_dw_after_upper_dw_count / sum_upper_dw if sum_upper_dw > 0 else 0

sum_lower_up = lower_up_after_lower_up_count + lower_up_after_lower_dw_count
lower_up_after_lower_up_probability = lower_up_after_lower_up_count / sum_lower_up if sum_lower_up > 0 else 0
lower_up_after_lower_dw_probability = lower_up_after_lower_dw_count / sum_lower_up if sum_lower_up > 0 else 0

sum_lower_dw = lower_dw_after_lower_up_count + lower_dw_after_lower_dw_count
lower_dw_after_lower_up_probability = lower_dw_after_lower_up_count / sum_lower_dw if sum_lower_dw > 0 else 0
lower_dw_after_lower_dw_probability = lower_dw_after_lower_dw_count / sum_lower_dw if sum_lower_dw > 0 else 0






print('*'*80)
print(f'upper_up_after_lower_up_count: {upper_up_after_lower_up_count}')
print(f'upper_up_after_lower_dw_count: {upper_up_after_lower_dw_count}')
print(f'upper_up_after_lower_up_probability: {upper_up_after_lower_up_probability}')
print(f'upper_up_after_lower_dw_probability: {upper_up_after_lower_dw_probability}')
print('*'*80)

print(f'upper_dw_after_lower_up_count: {upper_dw_after_lower_up_count}')
print(f'upper_dw_after_lower_dw_count: {upper_dw_after_lower_dw_count}')
print(f'upper_dw_after_lower_up_probability: {upper_dw_after_lower_up_probability}')
print(f'upper_dw_after_lower_dw_probability: {upper_dw_after_lower_dw_probability}')
print('*'*80)

print(f'lower_up_after_upper_up_count: {lower_up_after_upper_up_count}')
print(f'lower_up_after_upper_dw_count: {lower_up_after_upper_dw_count}')
print(f'lower_up_after_upper_up_probability: {lower_up_after_upper_up_probability}')
print(f'lower_up_after_upper_dw_probability: {lower_up_after_upper_dw_probability}')
print('*'*80)

print(f'lower_dw_after_upper_up_count: {lower_dw_after_upper_up_count}')
print(f'lower_dw_after_upper_dw_count: {lower_dw_after_upper_dw_count}')
print(f'lower_dw_after_upper_up_probability: {lower_dw_after_upper_up_probability}')
print(f'lower_dw_after_upper_dw_probability: {lower_dw_after_upper_dw_probability}')
print('*'*80)

print(f'upper_up_after_upper_up_count: {upper_up_after_upper_up_count}')
print(f'upper_up_after_upper_dw_count: {upper_up_after_upper_dw_count}')
print(f'upper_up_after_upper_up_probability: {upper_up_after_upper_up_probability}')
print(f'upper_up_after_upper_dw_probability: {upper_up_after_upper_dw_probability}')
print('*'*80)

print(f'upper_dw_after_upper_up_count: {upper_dw_after_upper_up_count}')
print(f'upper_dw_after_upper_dw_count: {upper_dw_after_upper_dw_count}')
print(f'upper_dw_after_upper_up_probability: {upper_dw_after_upper_up_probability}')
print(f'upper_dw_after_upper_dw_probability: {upper_dw_after_upper_dw_probability}')
print('*'*80)

print(f'lower_up_after_lower_up_count: {lower_up_after_lower_up_count}')
print(f'lower_up_after_lower_dw_count: {lower_up_after_lower_dw_count}')
print(f'lower_up_after_lower_up_probability: {lower_up_after_lower_up_probability}')
print(f'lower_up_after_lower_dw_probability: {lower_up_after_lower_dw_probability}')
print('*'*80)

print(f'lower_dw_after_lower_up_count: {lower_dw_after_lower_up_count}')
print(f'lower_dw_after_lower_dw_count: {lower_dw_after_lower_dw_count}')
print(f'lower_dw_after_lower_up_probability: {lower_dw_after_lower_up_probability}')
print(f'lower_dw_after_lower_dw_probability: {lower_dw_after_lower_dw_probability}')
print('*'*80)




 



