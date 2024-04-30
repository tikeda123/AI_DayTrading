import sys,os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.utils import configure_container
from common.init_common_module import init_common_module
from bybit_api.bybit_api import BybitOnlineAPI

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_date> <end_date>")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    # Append "00:00:00+0900" if only the date is provided
    if len(start_date) == 10:  # YYYY-MM-DD
        start_date += " 00:00:00+0900"
    if len(end_date) == 10:  # YYYY-MM-DD
        end_date += " 00:00:00+0900"

    # Ensure that the dates are in the correct format
    try:
        datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S%z")
        datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS+ZZZZ")
        sys.exit(1)



    configure_container()
    init_common_module()

    # BybitOnlineAPI インスタンスを作成
    bybit_online_api = BybitOnlineAPI()
    result = bybit_online_api.fetch_historical_data_all(start_date, end_date)
    print(result)


if __name__ == "__main__":
    main()
