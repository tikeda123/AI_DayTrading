import pandas as pd
from scipy.stats import chi2_contingency
mport pandas as pd
import os,sys

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from common.utils import get_config

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy


def main():
	# ダミーデータの作成
	data = {'カテゴリ1': ['A', 'A', 'B', 'B', 'C', 'C'],
			'カテゴリ2': ['X', 'Y', 'X', 'Y', 'X', 'Y']}
	df = pd.DataFrame(data)

	# クロス集計表の作成
	contingency_table = pd.crosstab(df['カテゴリ1'], df['カテゴリ2'])

	# カイ二乗検定の実行
	chi2, p, dof, expected = chi2_contingency(contingency_table)

	print(f"カイ二乗統計量: {chi2}")
	print(f"p値: {p}")
	print(f"自由度: {dof}")
	print("期待頻度表:")
	print(expected)

if __name__ == '__main__':
	main()

