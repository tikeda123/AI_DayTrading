import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # SMOTEのインポート

# CSVファイルの読み込み
#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230101000_20240110000_30_price_upper_ml.csv'
#file_path = 'result_lower_01.csv'  # ファイルパスを指定
#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20220101000_20240120000_60_price_upper_ml.csv'

#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230901000_20231201000_60_price_lower_ml.csv'
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_lower_ml.csv'



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

pca_features = pca_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9']]

# 目的変数の作成
target = (data['bb_profit'] >= 0).astype(int)

# 訓練セットとテストセットへの分割
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.2, random_state=None)

# SMOTEのインスタンス化
sm = SMOTE(random_state=None)

# 訓練データのオーバーサンプリング
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# SVMのパラメータグリッド
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# グリッドサーチの初期化
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# グリッドサーチの実行（オーバーサンプリングした訓練データを使用）
grid_search.fit(X_train_res, y_train_res)

# 最適なパラメータの表示
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# 最適なモデルを取得
best_model = grid_search.best_estimator_

# モデルの訓練（オーバーサンプリングした訓練データを使用）
best_model.fit(X_train_res, y_train_res)

# テストセットでの予測
y_pred = best_model.predict(X_test)

# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 混同行列と分類レポートの表示
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


