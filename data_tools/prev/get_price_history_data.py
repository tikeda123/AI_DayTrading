import sys
import os
from datetime import datetime

def setup_sys_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

def check_arguments_and_format_dates(argv):
    # Check for the presence of '-db' flag and remove it from the list to avoid affecting date parsing
    db_flag = '-db' in argv
    if db_flag:
        argv.remove('-db')

    if len(argv) != 3:
        exit_with_message("Usage: python script.py <start_date> <end_date> [-db]")

    start_date, end_date = format_dates(argv[1], argv[2])
    return start_date, end_date, db_flag

def format_dates(start_date, end_date):
    if len(start_date) == 10:
        start_date += " 00:00:00+0900"
    if len(end_date) == 10:
        end_date += " 00:00:00+0900"

    validate_date_format(start_date)
    validate_date_format(end_date)

    return start_date, end_date

def validate_date_format(date):
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        exit_with_message("Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS+ZZZZ")

def exit_with_message(message):
    print(message)
    sys.exit(1)

def main():
    setup_sys_path()
    from common.utils import configure_container
    from common.init_common_module import init_common_module
    from bybit_api.bybit_api import BybitOnlineAPI

    # The function now also returns the db_flag indicating whether the '-db' flag was present
    start_date, end_date, db_flag = check_arguments_and_format_dates(sys.argv)

    container = configure_container(name=__name__)
    init_common_module()

    bybit_online_api = BybitOnlineAPI()
    result = bybit_online_api.fetch_historical_data_all(start_date, end_date)
    # Assuming you want to use the db_flag here
    print(result)

    if db_flag:
        from common.data_loader_db import DataLoaderDB
        data_loader_db = container.data_loader_db()
        data_loader_db.import_to_db(dataframe=result)

        from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        analyzer.load_data_from_db()
        result = analyzer.analize()
        analyzer.import_to_db()
        print(result)

        # create technical analyzer tabled

        print("Data was saved to the database")

if __name__ == "__main__":
    main()


