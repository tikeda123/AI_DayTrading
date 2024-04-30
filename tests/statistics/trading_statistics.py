import pandas as pd
import numpy as np
import sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager

class TradingStatistics:
    def __init__(self, data: pd.DataFrame, config_manager: ConfigManager, trading_logger: TradingLogger):
        self.__data = self._filter_data(data)  # データの前処理
        self.__config_manager = config_manager
        self.__logger = trading_logger
        self._setup_counts()  # カウント用のディクショナリをセットアップ


    def _filter_data(self, data):
        """データを前処理する専用のメソッド"""
        return data[data['bb_flag'].isin(['upper', 'lower'])]

    def _setup_counts(self):
        """カウント用のディクショナリをセットアップする専用のメソッド"""
        flags = ['upper', 'lower']
        diffs = ['up', 'dw']
        self.__updw_counts =  {f"{flag}_{diff}_after_{next_flag}_{next_diff}": 0
                                for flag in flags
                                for diff in diffs
                                for next_flag in flags
                                for next_diff in diffs}
        

    def calculate_flag_differences_and_probabilities(self):
        """データをループ処理してフラグと差異のカウント、確率を計算するメソッド"""
        for index in range(len(self.__data) - 1):
            self.count_flags_and_differences(index)
        probabilities = self._calculate_probability_refactored()
        return self.__updw_counts, probabilities

    def count_flags_and_differences(self, index: int):
        """フラグと差異のカウントを行うメソッド"""
        current_row = self.__data.iloc[index]
        previous_row = self.__data.iloc[index - 1]
        flag_diff_combo = self._generate_key_for_row(current_row, previous_row)
        if flag_diff_combo in self.__updw_counts:
            self.__updw_counts[flag_diff_combo] += 1

    def _calculate_probability_refactored(self):
        """各カテゴリの確率を計算するメソッド"""
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
            category_sum = sum(self.__updw_counts.get(scenario, 0) for scenario in category_scenarios)
            for scenario in category_scenarios:
                probabilities[scenario] = self.__updw_counts.get(scenario, 0) / category_sum if category_sum > 0 else 0
        return probabilities

    def _generate_key_for_row(self, current_row, previous_row):
        """現在の行と前の行に基づいてキーを生成するメソッド"""
        def get_flag_diff(row):
            return f"{row['bb_flag']}_{ 'up' if row['bb_diff'] >= 0 else 'dw'}"
        return f"{get_flag_diff(current_row)}_after_{get_flag_diff(previous_row)}"
    


def main():

    config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'

    # ConfigManager インスタンスを作成
    config_manager = ConfigManager(config_path)
    trading_logger = TradingLogger(config_manager)

    # Load the CSV file
    file_path_new = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/statistics/oi_test01_240.csv'
    data = pd.read_csv(file_path_new)

    trading_statistics = TradingStatistics(data, config_manager, trading_logger)
    counts, probabilities = trading_statistics.calculate_flag_differences_and_probabilities()

    # Printing results
    for key, count in counts.items():
        print(f"{key}: {count}, Probability: {probabilities[key]:.2f}")

if __name__ == '__main__':
    main()

