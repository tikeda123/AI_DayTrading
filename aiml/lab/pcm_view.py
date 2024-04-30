import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

# データの読み込み（あなたのデータセットに合わせてパスを変更してください）
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_lower_ml.csv'
data = pd.read_csv(file_path)

# PCAのために目的変数を除外
features = data.drop(columns=['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA()
pca_data = pca.fit_transform(scaled_features)
pca_columns = ['PC' + str(i+1) for i in range(pca_data.shape[1])]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)

# 寄与率と累積寄与率の可視化
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 7))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
         label='cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()