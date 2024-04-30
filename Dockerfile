# Python 3.10.13をベースイメージとして使用
FROM python:3.10.13

# 必要なPythonパッケージをインストール
RUN pip install fastapi uvicorn psycopg2-binary pandas numpy

# アプリケーションのファイルをコンテナ内の作業ディレクトリにコピー
WORKDIR /app
COPY ./api_serv /app/
COPY ./common /app/common
COPY ./aitrading_settings_ver2.json /app/
COPY ./data /app/data

# CORS設定を含むFastAPIアプリケーションを実行するコマンド
CMD ["uvicorn", "trading_api:app", "--host", "0.0.0.0", "--port", "8000"]

