

import sys
#from pathlib import Path
import pandas as pd

# Adjust path imports

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from trading_analysis_kit.trailing_stop import TrailingStopCalculator


def main():
# Example usage of the class
    trailing_stop = TrailingStopCalculator()
    trailing_stop.set_entry_conditions(entry_price=10000, trailing_percent=5, is_long_position=False)

    trade_triggered,activation_price_after_fall = trailing_stop.update_price(current_price=9000)
    print(trade_triggered,activation_price_after_fall) 

    trade_triggered,activation_price_after_fall = trailing_stop.update_price(current_price=8000)
    print(trade_triggered,activation_price_after_fall) 

    trade_triggered,activation_price_after_fall = trailing_stop.update_price(current_price=7000)
    print(trade_triggered,activation_price_after_fall) 

    trade_triggered,activation_price_after_fall = trailing_stop.update_price(current_price=8000)
    print(trade_triggered,activation_price_after_fall) 


if __name__ == "__main__":
    main()