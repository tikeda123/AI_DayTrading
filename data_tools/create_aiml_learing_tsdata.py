import sys,os
import operator
from datetime import datetime
import pandas as pd
from dependency_injector.wiring import inject, Provide

# Import your necessary modules here
# ...
 # b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.config_manager import ConfigManager
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
from trading_analysis_kit.ml_strategy import MLDataCreationStrategy


def check_arguments_and_format_dates(argv):
    # Check for the presence of '-db' flag and remove it from the list to avoid affecting date parsing

    if len(argv) != 3:
        exit_with_message("Usage: python script.py <start_date> <end_date>")

    start_date, end_date = format_dates(argv[1], argv[2])
    return start_date, end_date

def format_dates(start_date, end_date):
    if len(start_date) == 10:
        start_date += " 00:00:00"
    if len(end_date) == 10:
        end_date += " 00:00:00"

    validate_date_format(start_date)
    validate_date_format(end_date)

    return start_date, end_date

def validate_date_format(date):
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        exit_with_message("Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")

def exit_with_message(message):
    print(message)
    sys.exit(1)



def create_ml_data_files_name(start_date,end_date,config_manager)->tuple:
    formatted_s_date = start_date[:-9].replace(':', '').replace('-', '').replace(' ', '')
    formatted_e_date = end_date[:-9].replace(':', '').replace('-', '').replace(' ', '')

    data_ml_path = os.path.join(parent_dir, config_manager.get('AIML', 'DATAPATH'))
    symbol = config_manager.get('ONLINE', 'SYMBOL')
    interval = config_manager.get('ONLINE', 'INTERVAL')

    datafile = f"{symbol}_{formatted_s_date}_{formatted_e_date}_{interval}_price"
    # MLデータファイルのパス
    ml_datafile_upper = os.path.join(data_ml_path, f"{datafile}_upper_mlts.csv")
    ml_datafile_lower = os.path.join(data_ml_path, f"{datafile}_lower_mlts.csv")
    mlnonts_datafile_upper = os.path.join(data_ml_path, f"{datafile}_upper_mlnonts.csv")
    mlnonts_datafile_lower = os.path.join(data_ml_path, f"{datafile}_lower_mlnonts.csv")


    return  ml_datafile_upper, ml_datafile_lower,mlnonts_datafile_upper,mlnonts_datafile_lower

def create_table_name(config_manager)->str:
    symbol = config_manager.get('ONLINE', 'SYMBOL')
    interval = config_manager.get('ONLINE', 'INTERVAL')
    return f'{symbol}_{interval}_market_data_tech'


def create_ml_data(start_date,end_date,config_manager)->pd.DataFrame:

    table_name = create_table_name(config_manager)
    strategy_context = MLDataCreationStrategy()
    context = SimulationStrategyContext(strategy_context)
    context.load_data_from_datetime_period(start_date, end_date, table_name)
    df = context.dataloader.get_raw()
    context.run_trading(context)
    result = context.get_data()
    return result

def create_time_series_data(df)->tuple:
    filtered_df = df[(df['bb_direction'].isin(['upper', 'lower'])) & (df['bb_profit'] != 0)]

    df['entry_volume'] = 0
    # 各データセットを分けるためのリスト
    time_series_data_upper = []
    time_series_data_lower = []

    # 各行に対して、その行とその前7行を抽出し、'upper' と 'lower' に分ける
    for index in filtered_df.index:
        start_index = max(0, index - 7)
        end_index = index + 1

        df.loc[start_index:end_index,'entry_price'] = df.iloc[end_index]['entry_price']
        df.loc[start_index:end_index,'entry_volume'] = df.iloc[end_index]['volume']

        extracted_data = df.iloc[start_index:end_index]
        if df.iloc[index]['bb_direction'] == 'upper':
            time_series_data_upper.append(extracted_data)
        elif df.iloc[index]['bb_direction'] == 'lower':
            time_series_data_lower.append(extracted_data)

    # 各データセットを結合してDataFrameにする
    combined_data_upper = pd.concat(time_series_data_upper, ignore_index=True)
    combined_data_lower = pd.concat(time_series_data_lower, ignore_index=True)
    return combined_data_upper, combined_data_lower

def create_nonets_data(upper_df,lower_df)->tuple:
    upper_df_filtered = upper_df[upper_df['bb_profit'] != 0]
    lower_df_filtered = lower_df[lower_df['bb_profit'] != 0]
    return upper_df_filtered,lower_df_filtered

def main():
    start_date, end_date = check_arguments_and_format_dates(sys.argv)

    config_manager = ConfigManager()

    upper_data_filename, lower_data_filename,upper_data_nonets_filename, lower_data_nonets_filename = create_ml_data_files_name(start_date, end_date, config_manager)
    ml_data = create_ml_data(start_date, end_date, config_manager)
    print(ml_data)
    upper_data, lower_data = create_time_series_data(ml_data)
    print(upper_data)
    print(lower_data)
    upper_data.to_csv(upper_data_filename, index=False)
    lower_data.to_csv(lower_data_filename, index=False)

    upper_data_nonets, lower_data_nonets = create_nonets_data(upper_data, lower_data)
    upper_data_nonets.to_csv(upper_data_nonets_filename, index=False)
    lower_data_nonets.to_csv(lower_data_nonets_filename, index=False)







if __name__ == "__main__":
    main()