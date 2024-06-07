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

def main():
	# ダミーデータの作成
	feature_columns = 'macdhist'
	target_column = 'bb_profit'
	mongo = MongoDataLoader()
	df = mongo.load_data_from_datetime_period("2024-01-01", "2024-06-03",MARKET_DATA_ML_LOWER)
	df = df[df[target_column]!=0]

	profit_p = df[df[target_column] > 0]
	profit_m = df[df[target_column] < 0]

	profit_p_sum = profit_p[target_column].sum()
	profit_m_sum = profit_m[target_column].sum()
	profit_sum  = profit_p_sum + profit_m_sum

	print(f'profit_p_sum: {profit_p_sum}, profit_m_sum: {profit_m_sum}, profit_sum: {profit_sum}')
	#print(f'profit_p_ratio: {profit_p_sum/profit_sum}, profit_m_ratio: {profit_m_sum/profit_sum}')
	print(f'profit_p count:{profit_p[target_column].count()},profit_m count:{profit_m[target_column].count()}')

if __name__ == '__main__':
	main()