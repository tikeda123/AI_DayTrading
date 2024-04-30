import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

# CSVファイルの読み込み
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_upper_ml.csv'


data = pd.read_csv(file_path)

# PCAのために目的変数を除外
features = data.drop(columns=['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma'])

# データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA()
pca_data = pca.fit_transform(scaled_features)
pca_columns = ['PC' + str(i+1) for i in range(pca_data.shape[1])]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)

pca_features = pca_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9','PC10','PC11','PC12']]
#pca_features = pca_df[['PC1', 'PC2', 'PC3', 'PC4']]
# 目的変数の作成
target = (data['bb_profit'] >= 0).astype(int)

# 訓練セットとテストセットへの分割
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.3, random_state=None)

# SMOTEのインスタンス化
#sm = SMOTE(random_state=None)

# 訓練データのオーバーサンプリング
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

X_train_res, y_train_res = X_train, y_train
# LightGBMのパラメータ
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'lambda_l1': 0.5,  # L1正則化の強さを指定
    'lambda_l2': 0.5   # L2正則化の強さを指定
}

# LightGBMデータセットの作成
train_data = lgb.Dataset(X_train_res, label=y_train_res)

# モデルの訓練
num_round = 300
bst = lgb.train(params, train_data, num_round)

# テストセットでの予測
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
# LightGBM outputs probabilities by default. Convert these to binary predictions:
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred_binary)
class_report = classification_report(y_test, y_pred_binary)

# 混同行列と分類レポートの表示
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# 新たなファイルAからデータを読み込む
file_path_A = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230901000_20231201000_60_price_upper_ml.csv'
#file_path_A = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_upper_ml.csv'

data_A = pd.read_csv(file_path_A)

# ファイルAのデータに対して同じ前処理を適用
# 目的変数に関連しない特徴量を除去
features_A = data_A.drop(columns=['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma'])

# データの標準化（訓練データでfitしたscalerを使用）
scaled_features_A = scaler.transform(features_A)

# PCA適用（訓練データでfitしたPCAを使用）
pca_data_A = pca.transform(scaled_features_A)
pca_df_A = pd.DataFrame(pca_data_A, columns=pca_columns)

pca_features = pca_df_A[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9','PC10','PC11','PC12']]
#pca_features = pca_df_A[['PC1', 'PC2', 'PC3', 'PC4']]

# 目的変数（ここではファイルAにも`bb_profit`列が存在すると仮定）
target_A = (data_A['bb_profit'] >= 0).astype(int)

# 訓練されたモデルを使用してファイルAのデータの予測を行う
y_pred_A = bst.predict(pca_features, num_iteration=bst.best_iteration)
y_pred_binary_A = [1 if x > 0.5 else 0 for x in y_pred_A]

# 予測結果の評価
conf_matrix_A = confusion_matrix(target_A, y_pred_binary_A)
class_report_A = classification_report(target_A, y_pred_binary_A)

# 混同行列と分類レポートの表示
print("Confusion Matrix for File A:")
print(conf_matrix_A)
print("\nClassification Report for File A:")
print(class_report_A)
