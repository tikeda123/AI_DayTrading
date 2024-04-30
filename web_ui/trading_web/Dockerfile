# ベースイメージの指定
FROM node:16

# アプリケーションディレクトリを作成
WORKDIR /usr/src/app

# アプリケーションの依存関係をインストール
# package.json と package-lock.json をコピー
COPY package*.json ./

RUN npm install
# プロダクション用の場合は、npm ci --only=production を使用

# アプリケーションのソースをバンドル
COPY . .

# アプリケーションがリッスンするポートを指定
EXPOSE 8080

# アプリケーションの起動
CMD [ "npm", "start" ]

