import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# CSVファイルの読み込み
#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20220101000_20240120000_60_price_lower_ml.csv'
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_lower_ml.csv'
#file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_upper_ml.csv'
trn_data = pd.read_csv(file_path)




trn_data['target'] = trn_data['bb_profit'] >= 0


ds = trn_data['bb_profit'].describe()
print(ds)
#Trueの件数, Falseの件数
print(trn_data['target'].value_counts())

#btc_data['target'] のTrueの比率
print(trn_data['target'].value_counts(normalize=True))



#features = ['close','dmi_diff','macd_diff','macdhist','rsi_buy','rsi_sell','entry_price','bol_diff','oi','rsi']
features = ['close','entry_price','bol_diff',"rsi","oi","rsi_buy","rsi_sell","macd_diff","macdhist","dmi_diff"]



# 特徴量 (X) とターゲット (y) の分離
X = trn_data[features]
y = trn_data['target']

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
# SMOTEインスタンスを生成


# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_model = SVC()

#best_model.fit(X_train_scaled, y_train)

sm = SMOTE(random_state=42)

# トレーニングデータをオーバーサンプリング
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# オーバーサンプリング後のデータでモデルを訓練
best_model.fit(X_train_res, y_train_res)


# テストデータでの評価
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


