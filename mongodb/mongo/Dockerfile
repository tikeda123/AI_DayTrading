FROM mongo:4

# mongo自身のrootユーザとパスワード設定
ENV MONGO_INITDB_ROOT_USERNAME sample_user
ENV MONGO_INITDB_ROOT_PASSWORD sample_pass

ENV TZ="Asia/Tokyo"
RUN echo $TZ > /etc/timezone

# setup.jsをentrypointにコピーし、イメージ作成時に初期セットアップを実行
COPY setup.js /docker-entrypoint-initdb.d/
