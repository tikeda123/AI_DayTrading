import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'result_lower_01.csv'  # ファイルパスを指定

data = pd.read_csv(file_path)
selected_columns = ['close','volume','volume_ma_diff','di_diff','middle_diff','macdhist','oi','funding_rate','bb_profit']
selected_data = data[selected_columns]

# 選択したカラムの相関行列を計算
selected_corr_matrix = selected_data.corr()

# KMeansクラスタリングの適用（例として2つのクラスターを使用）
kmeans = KMeans(n_clusters=2, random_state=0).fit(selected_corr_matrix)

# クラスターラベルの取得
selected_cluster_labels = kmeans.labels_

# 各変数とそのクラスターラベルを対応付け
selected_variable_clusters = pd.DataFrame({'Variable': selected_corr_matrix.columns, 'Cluster': selected_cluster_labels})

# クラスターごとに変数を表示
selected_clustered_variables = selected_variable_clusters.groupby('Cluster')['Variable'].apply(list)

# クラスタリング結果の可視化
sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Selected Variables')
plt.show()

selected_clustered_variables
