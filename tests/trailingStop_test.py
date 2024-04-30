import os, sys


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from trading_analysis_kit.trailing_stop import TrailingStopCalculator
from common.constants import *

def test_long_position():
    calculator = TrailingStopCalculator()
    entry_price = 100.0
    start_trailing_price = 120.0
    calculator.set_entry_conditions(entry_price, start_trailing_price, ENTRY_TYPE_LONG)

    assert calculator.entry_price == 100.0
    assert calculator.start_trailing_price == 120.0
    assert calculator.trade_tpye == ENTRY_TYPE_LONG


    current_price = 125.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")

    current_price = 135.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")

    current_price = 105.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")



def test_short_position():
    calculator = TrailingStopCalculator()
    entry_price = 110.0
    start_trailing_price = 90.0
    calculator.set_entry_conditions(entry_price, start_trailing_price, ENTRY_TYPE_SHORT)


    current_price = 80.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")

    current_price = 75.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")

    current_price = 100.0
    triggered, price = calculator.update_price(current_price)
    calculator.current_best_price
    print(f"crrent_price: {current_price}")
    print(f"triggered: {triggered}, price: {price}, current_best_price: {calculator.current_best_price}, activation_price: {calculator.activation_price}")



# テストの実行
test_long_position()
test_short_position()
print("All tests passed!")