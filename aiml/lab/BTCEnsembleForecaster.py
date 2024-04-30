import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


# データの読み込み（あなたのデータセットに合わせてパスを変更してください）
file_path = file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_upper_ml.csv'
data = pd.read_csv(file_path)

# 特徴量とターゲットの選択
#X = data[['ema', 'macdhist', 'rsi']]
#X = data[['close','entry_price','bol_diff',"rsi_buy","rsi_sell","macd_diff","macdhist","dmi_diff","volume"]]
X = data[['close','rsi','macd_diff','macdhist','dmi_diff']]
y = data['target'] = data['bb_profit'] >= 0  # 'target'列は、価格が上がる（1）か下がる（0）かを表す列と仮定

random_state = None
# データセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

random_state = None
#sm = SMOTE(random_state=None)

# トレーニングデータをオーバーサンプリング
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 個別のモデルを定義
random_state = None
model_rf = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=100, random_state=random_state))
model_gb = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(random_state=random_state))
model_svc = make_pipeline(MinMaxScaler(), SVC(probability=True, random_state=random_state))

# アンサンブルモデル（多数決）を定義
ensemble_model = VotingClassifier(estimators=[
    ('rf', model_rf),
    ('gb', model_gb),
    ('svc', model_svc)],
    voting='soft')

# モデルの訓練
ensemble_model.fit(X_train, y_train)

# テストデータでの予測
y_pred = ensemble_model.predict(X_test)

# 性能評価（例：正解率）

accuracy = accuracy_score(y_test, y_pred)
# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

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
X = data[['close','rsi','macd_diff','macdhist','dmi_diff']]
#X = data[['close','rsi',"rsi_buy","rsi_sell"]]
y = data['target'] = data['bb_profit'] >= 0  # 'target'列は、価格が上がる（1）か下がる（0）かを表す列と仮定

random_state = None
# データセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_state)


# テストデータでの予測
y_pred = ensemble_model.predict(X_test)

# 性能評価（例：正解率）

accuracy = accuracy_score(y_test, y_pred)
# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 混同行列と分類レポートの表示
print(f'Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)