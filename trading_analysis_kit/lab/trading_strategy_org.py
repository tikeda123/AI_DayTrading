import pandas as pd

from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from trading_analysis_kit.trailing_stop import TrailingStopAnalyzer


class TradingStrategy:
    def __init__(self, data: pd.DataFrame, config_manager: ConfigManager, trading_logger: TradingLogger):
        self.__config_manager = config_manager
        self.__logger = trading_logger
        self.__data = data
        self.__trailing_stop_analyzer = TrailingStopAnalyzer(config_manager, trading_logger)


    def find_band_walk(self, data: pd.DataFrame, index: int, direction: str) -> int:
        if direction not in ['upper', 'lower']:
            raise ValueError("direction must be 'upper' or 'lower'")

        consecutive_count = 0

        for i in range(index, min(index + 3, len(data))):
            if direction == 'upper' and data['close'][i] > data['upper'][i]:
                consecutive_count += 1
            elif direction == 'lower' and data['close'][i] < data['lower'][i]:
                consecutive_count += 1
            else:
                return data['close'][index]

            if consecutive_count == 3:
                return data['close'][i]

        return  data['close'][index]
    
    def find_price_on_condition(self, data: pd.DataFrame, index: int, direction: str) -> float:
        if direction not in ['upper', 'lower']:
            raise ValueError("Direction must be 'upper' or 'lower'")

        for i in range(index, min(index + 12, len(data))):
            if direction == 'upper' and data['close'][i] < data['middle'][i]:
                return data['close'][i]
            elif direction == 'lower' and data['close'][i] > data['middle'][i]:
                return data['close'][i]

        # 12期目の価格を返す
        return data['close'][min(index + 11, len(data) - 1)]


    def get_up_ratio(self, prev_direction: str,prev_bb: float,current_direction: str) -> float:
        #print(f'prev_direction:{prev_direction}, prev_bb:{prev_bb}, current_direction:{current_direction}')
        if prev_direction == 'upper' and prev_bb > 0 and current_direction == 'upper' :
            return 0.48
        elif prev_direction == 'upper' and prev_bb > 0 and current_direction == 'lower':
            return 0.24
        elif prev_direction == 'upper' and prev_bb <= 0 and current_direction == 'upper':
            return 0.44
        elif prev_direction == 'upper' and prev_bb <= 0 and current_direction == 'lower':
            return 0.43
        elif prev_direction == 'lower' and prev_bb > 0 and current_direction == 'upper':
            return 0.45
        elif prev_direction == 'lower' and prev_bb > 0 and current_direction == 'lower':
            return 0.44
        elif prev_direction == 'lower' and prev_bb <= 0 and current_direction == 'upper':
            return 0.30
        elif prev_direction == 'lower' and prev_bb <= 0 and current_direction == 'lower':
            return 0.27
        
        print(f'error:prev_direction:{prev_direction}, prev_bb:{prev_bb}, current_direction:{current_direction}')
        return 1.0

    def find_prev_bb_flag(self, data: pd.DataFrame, index: int):
        for i in range(index, 0, -1):
            if data.iloc[i]['bb_flag'] is not None:
                flag = data.iloc[i]['bb_flag']
                bb_diff = data.iloc[i]['bb_diff']
                return flag, bb_diff

        print(f'warning:index:{index}')
        return 'upper', 1.0
    
    """ 
    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None

        for i in range(len(data)):
            row = data.iloc[i]
            if row['close'] > row['upper']:
                best_exit_price, is_long =self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                #print(f'best_exit_price:{best_exit_price}, is_long:{is_long}')
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif row['close'] < row['lower']:
                #print(f'index:{i} row[close]:{row["close"]}, row[lower]:{row["lower"]}')
                best_exit_price, is_long =self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                #print(f'best_exit_price:{best_exit_price}, is_long:{is_long}')
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data
    
   
    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None

        for i in range(len(data)):
            row = data.iloc[i]
            if row['close'] > row['upper']:
                best_exit_price = self.find_band_walk(data, i, 'upper')
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif row['close'] < row['lower']:
                best_exit_price = self.find_band_walk(data, i, 'lower')
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data




    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None

        for i in range(len(data)):
            row = data.iloc[i]
            if row['close'] > row['upper']:
                best_exit_price, is_long =self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                print(f'best_exit_price:{best_exit_price}, is_long:{is_long}')
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif row['close'] < row['lower']:
                print(f'index:{i} row[close]:{row["close"]}, row[lower]:{row["lower"]}')
                best_exit_price, is_long =self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                print(f'best_exit_price:{best_exit_price}, is_long:{is_long}')
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data


    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None

        for i in range(len(data)):
            row = data.iloc[i]
            if row['close'] > row['upper']:
                end_index = min(i + 5, len(data))
                mean_close = data['close'].iloc[i:end_index].mean()
                row['bb_diff'] = round(mean_close, 2) - row['close']
                row['bb_flag'] = 'upper'
            elif row['close'] < row['lower']:
                end_index = min(i + 5, len(data))
                mean_close = data['close'].iloc[i:end_index].mean()
                row['bb_diff'] = row['close'] - round(mean_close, 2)
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data
 
    
    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None
        data['exit_price'] = 0

        for i in range(len(data)):
            row = data.iloc[i]
            upper_breakout = True
            lower_breakout = True

            # 過去5期間のチェック
            for j in range(max(0, i - 5), i):
                if data.iloc[j]['bb_flag'] == 'upper':
                    upper_breakout = False
                if data.iloc[j]['bb_flag'] == 'lower':
                    lower_breakout = False

            if upper_breakout and row['close'] > row['upper']:
                best_exit_price, is_long = self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                row['exit_price'] = best_exit_price
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif lower_breakout and row['close'] < row['lower']:
                best_exit_price, is_long = self.__trailing_stop_analyzer.apply_trailing_stop_to_row(data, i)
                row['exit_price'] = best_exit_price
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data

        def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None
        data['exit_price'] = 0


        for i in range(len(data)):
            row = data.iloc[i]
            upper_breakout = True
            lower_breakout = True

            # 過去5期間のチェック
            for j in range(max(0, i - 5), i):
                if data.iloc[j]['bb_flag'] == 'upper':
                    upper_breakout = False
                if data.iloc[j]['bb_flag'] == 'lower':
                    lower_breakout = False

            if upper_breakout and row['close'] > row['upper']:
                best_exit_price = self.find_price_on_condition(data, i, 'upper')
                row['exit_price'] = best_exit_price
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif lower_breakout and row['close'] < row['lower']:
                best_exit_price = self.find_price_on_condition(data, i, 'lower')
                row['exit_price'] = best_exit_price
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data
        """ 
    
    def get_rsi_up_ratio(self, rsi: float, direction: str) -> float:

        if direction == 'upper' and rsi >= 50 and rsi < 60:
            return 0.3714
        elif direction == 'upper' and rsi >= 60 and rsi < 70:
            return 0.3898
        elif direction == 'upper' and rsi >= 70 and rsi < 80:
            return 0.4048
        elif direction == 'upper' and rsi >= 80 and rsi < 90:
            return 0.75
        elif direction == 'upper' and rsi >= 90 and rsi < 100:
            return 0.01
        
        elif direction == 'lower' and rsi >= 40 and rsi < 50:
            return 0.375
        elif direction == 'lower' and rsi >= 30 and rsi < 40:
            return 0.3469
        elif direction == 'lower' and rsi >= 20 and rsi < 30:
            return 0.3667
        elif direction == 'lower' and rsi >= 10 and rsi < 20:
            return 0.5714
        elif direction == 'lower' and rsi >= 0 and rsi < 10:
            return 0.01
        
        print(f'error:rsi:{rsi}, direction:{direction}')
        return 0




    def analyze_bollinger_band_breakouts(self) -> pd.DataFrame:
        data = self.__data
        data['bb_diff'] = 0
        data['bb_flag'] = None
        data['exit_price'] = 0
        data['up_ratio'] = 0
        data['dw_ratio'] = 0
        data['rsi_up_ratio'] = 0


        for i in range(len(data)):
            row = data.iloc[i]
            upper_breakout = True
            lower_breakout = True

            # 過去5期間のチェック
            for j in range(max(0, i - 5), i):
                if data.iloc[j]['bb_flag'] == 'upper':
                    upper_breakout = False
                if data.iloc[j]['bb_flag'] == 'lower':
                    lower_breakout = False

            if upper_breakout and row['close'] > row['upper']:
                flag,diff = self.find_prev_bb_flag(data, i)
                row['up_ratio'] = self.get_up_ratio(flag,diff,'upper')
                row['dw_ratio'] = 1.0 - row['up_ratio']

                row['rsi_up_ratio'] = self.get_rsi_up_ratio(row['rsi'],'upper')

                best_exit_price = self.find_price_on_condition(data, i, 'upper')
                row['exit_price'] = best_exit_price
                row['bb_diff'] = best_exit_price - row['close']
                row['bb_flag'] = 'upper'
            elif lower_breakout and row['close'] < row['lower']:
                flag,diff = self.find_prev_bb_flag(data, i)
                row['up_ratio'] = self.get_up_ratio(flag,diff,'lower')
                row['dw_ratio'] = 1.0 - row['up_ratio']

                row['rsi_up_ratio'] = self.get_rsi_up_ratio(row['rsi'],'lower')

                best_exit_price = self.find_price_on_condition(data, i, 'lower')
                row['exit_price'] = best_exit_price
                row['bb_diff'] = row['close'] - best_exit_price
                row['bb_flag'] = 'lower'
            else:
                row['bb_flag'] = None
                row['bb_diff'] = 0

            data.iloc[i] = row

        return data

    def analyze_macd_trends_after_bollinger_breakouts(self, flag: str) -> pd.DataFrame:
        data = self.__data.copy()
        #data[['prev_macd', 'prev_macdsignal']] = data[['macd', 'macdsignal']].shift(1)
        #self.__add_macd_analysis(data)

        filtered = data[data['bb_flag'] == flag]
        return filtered

    def __add_macd_analysis(self, data_frame):
        data_frame['macdhist_positive'] = data_frame['macdhist'] > 0
        data_frame['macd_rising'] = data_frame['macd'] > data_frame['prev_macd']
        data_frame['macdsignal_rising'] = data_frame['macdsignal'] > data_frame['prev_macdsignal']
        data_frame['macd_positive'] = data_frame['macd'] > 0
        data_frame['macdsignal_positive'] = data_frame['macdsignal'] > 0

    def get_data(self):
        return self.__data
