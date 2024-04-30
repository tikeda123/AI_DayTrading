import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# データの読み込み（あなたのデータセットに合わせてパスを変更してください）
file_path = file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_upper_ml.csv'
data = pd.read_csv(file_path)

# PCAのために目的変数を除外
features = data.drop(columns=['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA()
pca_data = pca.fit_transform(scaled_features)
pca_columns = ['PC' + str(i+1) for i in range(pca_data.shape[1])]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)



pca_features = pca_df[['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6']]
target = data['target'] = data['bb_profit'] >= 0  # 'target'列は、価格が上がる（1）か下がる（0）かを表す列と仮定


random_state = None
# データセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.3, random_state=random_state)







from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 個別のモデルを定義
random_state = None

# ベースモデル
base_models = [
    ('rf', RandomForestClassifier(n_estimators=400, random_state=random_state)),
    ('gb', GradientBoostingClassifier(n_estimators=400, random_state=random_state)),
    ('svc', SVC(probability=True, random_state=random_state))
]

# メタモデル
meta_model = LogisticRegression()

# スタッキング分類器の定義
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# スタッキングモデルの訓練
stacking_model.fit(X_train, y_train)

# テストデータでの予測
y_pred_stacking = stacking_model.predict(X_test)



# 性能評価（例：正解率）

accuracy = accuracy_score(y_test, y_pred_stacking)
# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred_stacking)
class_report = classification_report(y_test, y_pred_stacking)

# 混同行列と分類レポートの表示
print(f'Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)





# データの読み込み（あなたのデータセットに合わせてパスを変更してください）
# 新たなファイルAからデータを読み込む
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20230901000_20231201000_60_price_upper_ml.csv'
data = pd.read_csv(file_path)

# 特徴量とターゲットの選択
features = data.drop(columns=['profit_mean', 'state', 'bb_profit', 'start_at', 'date', 'exit_price', 'bb_direction', 'bb_profit', 'profit_max', 'profit_min', 'profit_ma'])

scaled_features = scaler.fit_transform(features)

pca_data = pca.fit_transform(scaled_features)
pca_columns = ['PC' + str(i+1) for i in range(pca_data.shape[1])]
pca_df = pd.DataFrame(pca_data, columns=pca_columns)


pca_features =  pca_df[['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6']]
target = data['target'] = data['bb_profit'] >= 0  # 'target'列は、価格が上がる（1）か下がる（0）かを表す列と仮定


random_state = None
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.9, random_state=random_state)



# テストデータでの予測
# テストデータでの予測
y_pred_stacking = stacking_model.predict(X_test)

# 性能評価（例：正解率）

accuracy = accuracy_score(y_test, y_pred_stacking)
# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred_stacking)
class_report = classification_report(y_test, y_pred_stacking)

# 混同行列と分類レポートの表示
print(f'Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)