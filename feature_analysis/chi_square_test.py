import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats as stats
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得
sys.path.append(parent_dir)

from mongodb.data_loader_mongo  import MongoDataLoader
from common.constants import *
# Aディレクトリーのパスをsys.pathに追加


def chi_square_test(df, feature_columns, target_column):
	"""
	カイ二乗検定を行う関数です。
	クロス集計表を作成し、カイ二乗検定を実行します。

	Args:
		df (pd.DataFrame): データフレーム
		feature_columns (str): 特徴量のカラム名
		target_column (str): 目的変数のカラム名

	Returns:
		float: カイ二乗統計量
		float: p値
		int: 自由度
		np.ndarray: 期待度数の配列
	"""
	# クロス集計表の作成
	contingency_table = pd.crosstab(df[feature_columns],  df[target_column],dropna=False)

	# カイ二乗検定の実行
	chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

	return contingency_table,chi2, p, dof, expected

def main():
	# ダミーデータの作成
	feature_columns = 'macdhist'
	target_column = 'bb_profit'
	mongo = MongoDataLoader()
	df = mongo.load_data_from_datetime_period("2020-01-01", "2024-06-03",MARKET_DATA_ML_LOWER)
	df = df[df[target_column]!=0]

	df['profit'] = df[target_column] > 0
	df['feature_bins'] = pd.cut(df[feature_columns], bins=6)


	# カイ二乗検定の実行
	contingency_table,chi2, p, dof, expected = chi_square_test(df, 'feature_bins', 'profit')

	print("クロス集計表:")
	print(contingency_table)
	print("\nカイ二乗統計量:", chi2)
	print("p値:", p)
	print("自由度 (dof):", dof)


if __name__ == '__main__':
	main()