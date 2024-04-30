import pandas as pd
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# CSVファイルの読み込み
#file_path = 'result_lower_01.csv'  # ファイルパスを指定
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230101000_20240110000_30_price_lower_ml.csv'
btc_data = pd.read_csv(file_path)


ds=btc_data['volume'].describe()
print(ds)

# 目的変数の作成
#btc_data = btc_data[ operator.ge(btc_data['volume'],7000) ]
#btc_data = btc_data[ operator.ge(btc_data['turnover'],2.6) ]
ds=btc_data['turnover'].describe()
print(ds)


btc_data['target'] = btc_data['bb_profit'] >= 0
ds = btc_data['bb_profit'].describe()
print(ds)
#Trueの件数, Falseの件数
print(btc_data['target'].value_counts())

#btc_data['target'] = btc_data['profit_mean'] > 0
#btc_data['target'] のTrueの比率
print(btc_data['target'].value_counts(normalize=True))


features = [
  'close','volume','volume_ma_diff','di_diff','middle_diff','macdhist','oi'
]


# 特徴量 (X) とターゲット (y) の分離
X = btc_data[features]
y = btc_data['target']

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




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

# グリッドサーチの実行
grid_search.fit(X_train_scaled, y_train)

# 最適なパラメータの表示
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)
best_model = grid_search.best_estimator_



# テストデータでの評価
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

