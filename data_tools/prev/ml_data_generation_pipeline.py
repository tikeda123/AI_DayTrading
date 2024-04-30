import sys,os
import operator
from datetime import datetime
from dependency_injector.wiring import inject, Provide

# Import your necessary modules here
# ...
 # b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.utils import configure_container
from common.init_common_module import init_common_module
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext
from trading_analysis_kit.ml_strategy import MLDataCreationStrategy


def check_arguments_and_format_dates(argv)->tuple:
    """
    コマンドライン引数を検証し、日付形式を整形する。

    Args:
        argv (list): コマンドライン引数のリスト。

    Returns:
        tuple: 整形された開始日と終了日。
    """
    # Check for the presence of '-db' flag and remove it from the list to avoid affecting date parsing

    if len(argv) != 3:
        exit_with_message("Usage: python script.py <start_date> <end_date>")

    start_date, end_date = format_dates(argv[1], argv[2])
    return start_date, end_date

def format_dates(start_date, end_date)->tuple:
    """
    日付文字列の形式をYYYY-MM-DD HH:MM:SSに整形する。

    Args:
        start_date (str): 開始日の文字列。
        end_date (str): 終了日の文字列。

    Returns:
        tuple: 整形された開始日と終了日。
    """
    if len(start_date) == 10:
        start_date += " 00:00:00"
    if len(end_date) == 10:
        end_date += " 00:00:00"

    validate_date_format(start_date)
    validate_date_format(end_date)

    return start_date, end_date

def validate_date_format(date):
    """
    日付文字列が正しい形式であるか検証する。

    Args:
        date (str): 検証する日付の文字列。
    """
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        exit_with_message("Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")

def exit_with_message(message):
    """
    指定されたメッセージを表示してプログラムを終了する。

    Args:
        message (str): 表示するメッセージ。
    """
    print(message)
    sys.exit(1)

def change_format_date(date_str)->str:
    """
    日付文字列の形式をYYYY-MM-DDに変更する。

    Args:
        date_str (str): 変更する日付の文字列。

    Returns:
        str: 形式が変更された日付文字列。
    """
    try:
        # Try parsing the datetime with time
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        # If successful, format it to date only
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # If the first format fails, try parsing as date only
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            # If this succeeds, return the original string
            return date_str
        except ValueError:
            # If both fail, it's not in the expected format
            return "Invalid format"


def create_ml_data_files_name(start_date,end_date,config_manager)->tuple:
    """
    機械学習データファイルの名前を生成する。

    Args:
        start_date (str): 開始日。
        end_date (str): 終了日。
        config_manager: 設定管理オブジェクト。

    Returns:
        tuple: データファイルのパス、上限ファイルのパス、下限ファイルのパス、テーブル名。
    """
    formatted_s_date = start_date[:-9].replace(':', '').replace('-', '').replace(' ', '')
    formatted_e_date = end_date[:-9].replace(':', '').replace('-', '').replace(' ', '')

    data_ml_path = os.path.join(parent_dir, config_manager.get('AIML', 'DATAPATH'))
    symbol = config_manager.get('ONLINE', 'SYMBOL')
    interval = config_manager.get('ONLINE', 'INTERVAL')

    datafile = f"{symbol}_{formatted_s_date}_{formatted_e_date}_{interval}_price"
    table_name = f'{symbol}_{interval}_market_data_tech'
    # MLデータファイルのパス
    ml_datafile = os.path.join(data_ml_path, f"{datafile}_ml.csv")
    ml_datafile_upper = os.path.join(data_ml_path, f"{datafile}_upper_ml.csv")
    ml_datafile_lower = os.path.join(data_ml_path, f"{datafile}_lower_ml.csv")

    return ml_datafile, ml_datafile_upper, ml_datafile_lower, table_name

def create_ml_data(start_date,end_date,config_manager):
    """
    指定された期間のデータから機械学習データを生成する。

    Args:
        start_date (str): 開始日。
        end_date (str): 終了日。
        config_manager: 設定管理オブジェクト。
    """

    ml_datafile, ml_datafile_upper, ml_datafile_lower,table_name = create_ml_data_files_name(start_date,end_date,config_manager)

    strategy_context = MLDataCreationStrategy()
    context = SimulationStrategyContext(strategy_context)
    context.load_data_from_datetime_period(start_date, end_date, table_name)
    df = context.dataloader.get_raw()
    context.run_trading(context)
    result = context.get_data()
    result.to_csv(ml_datafile)

    data_loader = context.dataloader

    for direction, file_path in [("upper", ml_datafile_upper), ("lower", ml_datafile_lower)]:
        result.dropna(subset=['bb_profit'])
        df_filtered = data_loader.filter_and('bb_direction', operator.eq, direction, 'bb_profit', operator.ne, 0.0)
        df_filtered.to_csv(file_path)
        print(f"Saved to {file_path}")
        print(df_filtered)


def main():
    start_date, end_date = check_arguments_and_format_dates(sys.argv)
    container = configure_container()
    init_common_module()
    config_manager = container.config_manager()

    create_ml_data(start_date, end_date, config_manager)

if __name__ == "__main__":
    main()