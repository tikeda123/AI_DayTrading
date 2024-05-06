import sys,os
from pathlib import Path
import pandas as pd
import numpy as np

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger

class DataLoader:
    """
    データをロードし、加工するクラスです。

    Attributes:
        logger (TradingLogger): ログ情報を管理するオブジェクト。
        _df (pd.DataFrame): ロードされた生データ。
        tstpath (Path): テストデータのパス。
        tstfile (str): テストファイルの名前。

    Args:
        conf (dict): 設定情報が含まれる辞書。'TSTPATH'と'TSTFILE'をキーとして持ちます。
    """

    def __init__(self):
        self.logger = TradingLogger()
        self._df = None # Aディレクトリーのパスを取得

    def remove_unuse_colums(self):
        """
        # "Unnamed: 0" という列がある場合は削除
        """
        if '_id' in self._df.columns:
            self._df = self._df.drop(columns=['_id'])

    def reset_index(self):
        """
        インデックスをリセットします。
        """
        self._df = self._df.reset_index(drop=False)

    def get_df_raw(self) -> pd.DataFrame:
        """
        ロードされた生データのDataFrameを取得します。

        Returns:
            pd.DataFrame: ロードされた生データ。
        """
        return self._df

    def set_df_raw(self, df: pd.DataFrame):
        """
        生データのDataFrameを設定します。

        Args:
            df (pd.DataFrame): 新しい生データ。
        """
        self._df = df

    def get_df(self, index:int,column) -> pd.DataFrame:
        """
        指定したインデックスとカラム名に対応するデータフレームの値を取得します。

        Args:
            index (int): 値を取得したい行のインデックス。
            column (str): 値を取得したいカラム名。

        Returns:
            pd.DataFrame: 指定されたインデックスとカラムに対応する値。
        """
        return self._df.at[index,column]

    def set_df(self, index:int,column,value) -> pd.DataFrame:
        """
        指定したインデックスとカラム名の位置に値を設定します。

        Args:
            index (int): 値を設定する行のインデックス。
            column (str): 値を設定するカラム名。
            value: 設定する値。
        """
        self._df.at[index,column] = value

    def set_df_fromto(self, start_index:int,end_index:int,column,value):
        """
        指定した範囲のインデックスとカラム名の位置に値を一括設定します。

        Args:
            start_index (int): 値を設定する開始行のインデックス。
            end_index (int): 値を設定する終了行のインデックス。
            column (str): 値を設定するカラム名。
            value: 設定する値。
        """
        self._df.loc[start_index:end_index,column] = value

    def get_df_fromto(self, start_index:int,end_index:int) -> pd.DataFrame:
        """
        指定した範囲のインデックスに対応するデータフレームの部分集合を取得します。

        Args:
            start_index (int): 部分集合の開始行のインデックス。
            end_index (int): 部分集合の終了行のインデックス。

        Returns:
            pd.DataFrame: 指定された範囲のデータフレーム。
        """
        return self._df.loc[start_index:end_index]

    def df_new_column(self, column, value, dtype):
        """
        新しいカラムをデータフレームに追加し、指定された値とデータ型で初期化します。

        Args:
            column (str): 新しいカラムの名前。
            value: カラムの初期値。
            dtype: カラムのデータ型。
        """
        self._df[column] = value
        self._df[column] = self._df[column].astype(dtype)

    def is_first_column_less_than_second(self, index:int, col1,col2) -> bool:
        """
        指定されたインデックスで、一つ目のカラムの値が二つ目のカラムの値より小さいかどうかを判定します。

        Args:
            index (int): 判定する行のインデックス。
            col1 (str): 一つ目のカラム名。
            col2 (str): 二つ目のカラム名。

        Returns:
            bool: 一つ目のカラムの値が二つ目のカラムの値より小さい場合はTrue、そうでない場合はFalse。
        """
        #print(f'col1:{col1} col2:{col2}')
        #print(f'f<s first:{self._df.at[index,col1]} second:{self._df.at[index,col2]}')
        return self._df.at[index,col1] < self._df.at[index,col2]

    def is_first_column_greater_than_second(self, index:int, col1,col2) -> bool:
        """
        指定されたインデックスで、一つ目のカラムの値が二つ目のカラムの値より大きいかどうかを判定します。

        Args:
            index (int): 判定する行のインデックス。
            col1 (str): 一つ目のカラム名。
            col2 (str): 二つ目のカラム名。

        Returns:
            bool: 一つ目のカラムの値が二つ目のカラムの値より大きい場合はTrue、そうでない場合はFalse。
        """
        #print(f'col1:{col1} col2:{col2}')
        #print(f's<f first:{self._df.at[index,col1]} second:{self._df.at[index,col2]}')
        return self._df.at[index,col1] > self._df.at[index,col2]

    def max_value(self, start_index:int,end_index:int,column) -> float:
        """
        指定された範囲のカラムで最大値を求めます。

        Args:
            start_index (int): 範囲の開始行のインデックス。
            end_index (int): 範囲の終了行のインデックス。
            column (str): 最大値を求めるカラム名。

        Returns:
            float: 指定された範囲のカラムの最大値。
        """
        return self._df.loc[start_index:end_index,column].max()

    def min_value(self, start_index:int,end_index:int,column) -> float:
        """
        指定された範囲のカラムで最小値を求めます。

        Args:
            start_index (int): 範囲の開始行のインデックス。
            end_index (int): 範囲の終了行のインデックス。
            column (str): 最小値を求めるカラム名。

        Returns:
            float: 指定された範囲のカラムの最小値。
        """
        return self._df.loc[start_index:end_index,column].min()

    def max_value_index(self, start_index:int,end_index:int,column) -> int:
        """
        指定されたカラムにおいて、指定された範囲で最大値を持つ行のインデックスを返します。

        Args:
            start_index (int): 検索範囲の開始インデックス。
            end_index (int): 検索範囲の終了インデックス。
            column (str): 値を検索するカラム名。

        Returns:
            int: 最大値を持つ行のインデックス。
        """
        return self._df.loc[start_index:end_index,column].idxmax()

    def min_value_index(self, start_index:int,end_index:int,column) -> int:
        """
        指定されたカラムにおいて、指定された範囲で最小値を持つ行のインデックスを返します。

        Args:
            start_index (int): 検索範囲の開始インデックス。
            end_index (int): 検索範囲の終了インデックス。
            column (str): 値を検索するカラム名。

        Returns:
            int: 最小値を持つ行のインデックス。
        """
        return self._df.loc[start_index:end_index,column].idxmin()

    def mean_value(self, start_index:int,end_index:int,column) -> float:
        """
        指定されたカラムの、指定された範囲における平均値を計算します。

        Args:
            start_index (int): 平均値計算の開始インデックス。
            end_index (int): 平均値計算の終了インデックス。
            column (str): 平均値を計算するカラム名。

        Returns:
            float: 指定された範囲での平均値。
        """
        return self._df.loc[start_index:end_index,column].mean()

    def std_value(self, start_index:int,end_index:int,column) -> float:
        """
        指定されたカラムの、指定された範囲における標準偏差を計算します。

        Args:
            start_index (int): 標準偏差計算の開始インデックス。
            end_index (int): 標準偏差計算の終了インデックス。
            column (str): 標準偏差を計算するカラム名。

        Returns:
            float: 指定された範囲での標準偏差。
        """
        return self._df.loc[start_index:end_index,column].std()

    def describe(self, start_index:int,end_index:int,column)-> pd.Series:
        """
        指定されたカラムの、指定された範囲における統計的記述を提供します。

        Args:
            start_index (int): 統計記述の開始インデックス。
            end_index (int): 統計記述の終了インデックス。
            column (str): 統計的記述を取得するカラム名。

        Returns:
            pd.Series: 指定された範囲での統計的記述（最小値、25%パーセンタイル、平均値、75%パーセンタイル、最大値など）。
        """
        return self._df.loc[start_index:end_index,column].describe()

    def filter(self,column,operator,value) -> pd.DataFrame:
        """
        指定されたカラムに対してフィルタ操作を適用します。

        Args:
            column (str): フィルタを適用するカラム名。
            operator (callable): 比較演算子またはフィルタ条件を表す関数。
            value: フィルタ条件に使用する値。

        Returns:
            pd.DataFrame: フィルタ条件を満たす行のみを含むデータフレーム。
        """
        #mask = operator(self._df[column], value)
        return self._df[operator(self._df[column],value)]
        #return self._df[mask]

    def filter_and(self,column1,operator1,value1,column2,operator2,value2) -> pd.DataFrame:
        """
        二つのカラムに対してAND条件でフィルタ操作を適用します。

        Args:
            column1 (str): 最初のフィルタを適用するカラム名。
            operator1 (callable): 最初のカラムの比較演算子またはフィルタ条件を表す関数。
            value1: 最初のフィルタ条件に使用する値。
            column2 (str): 二つ目のフィルタを適用するカラム名。
            operator2 (callable): 二つ目のカラムの比較演算子またはフィルタ条件を表す関数。
            value2: 二つ目のフィルタ条件に使用する値。

        Returns:
            pd.DataFrame: 両方のフィルタ条件を満たす行のみを含むデータフレーム。
        """
        return self._df[operator1(self._df[column1],value1) & operator2(self._df[column2],value2)]

    def filter_or(self,column1,operator1,value1,column2,operator2,value2) -> pd.DataFrame:
        """
        二つのカラムに対してOR条件でフィルタ操作を適用します。

        Args:
            column1 (str): 最初のフィルタを適用するカラム名。
            operator1 (callable): 最初のカラムの比較演算子またはフィルタ条件を表す関数。
            value1: 最初のフィルタ条件に使用する値。
            column2 (str): 二つ目のフィルタを適用するカラム名。
            operator2 (callable): 二つ目のカラムの比較演算子またはフィルタ条件を表す関数。
            value2: 二つ目のフィルタ条件に使用する値。

        Returns:
            pd.DataFrame: いずれかのフィルタ条件を満たす行のみを含むデータフレーム。
        """
        return self._df[operator1(self._df[column1],value1) | operator2(self._df[column2],value2)]











