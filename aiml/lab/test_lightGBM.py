import pandas as pd
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# CSVファイルの読み込み
file_path = 'result_lower_01.csv'  # ファイルパスを指定
btc_data = pd.read_csv(file_path)

# 目的変数の作成
btc_data = btc_data[ operator.ge(btc_data['volume'].abs(),6000) ]
ds=btc_data['volume'].describe()
print(ds)

btc_data['target'] = btc_data['bb_profit'] >= 0
ds = btc_data['bb_profit'].describe()
print(ds)
#Trueの件数, Falseの件数
print(btc_data['target'].value_counts())


print(btc_data['target'].value_counts(normalize=True))

""""
# 特徴量の選択
features = [
    'close','volume_ma_diff','di_diff','up_ratio'
]
"""
features = [
  'close','volume','volume_ma_diff','macdhist','oi'
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

#X_train_scaled = X_train
#X_test_scaled = X_test

# LightGBMのパラメータグリッド
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'num_leaves': [31, 50, 100],
    'max_depth': [10, 20, 40],
    'min_child_samples': [20, 30, 40],
}

# グリッドサーチの初期化
grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

# グリッドサーチの実行
grid_search.fit(X_train_scaled, y_train)

# 最適なパラメータの表示
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)
# クロスバリデーションの結果を確認
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[['param_n_estimators', 'param_learning_rate', 'mean_test_score', 'std_test_score']])


# 最適なモデルを取得
best_model = grid_search.best_estimator_

# テストデータでの評価
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 特徴量の重要度の表示
feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), [features[i] for i in sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

