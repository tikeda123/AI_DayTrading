from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys,os

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger


# 既存のDataLoaderTransactionDBクラスのインポート（上記のプログラムを想定）
from common.data_loader_tran import DataLoaderTransactionDB
from common.data_loader_db import DataLoaderDB
from common.utils import get_config

app = FastAPI()

# CORSを許可するオリジンのリスト
origins = [
    "http://localhost:3000",  # Reactアプリのオリジンを許可
    "http://127.0.0.1:3000",  # 必要に応じて追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可するには ["*"] を使用
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのHTTPヘッダを許可
)


# リクエストボディのためのモデルを定義
class ReadDBRequest(BaseModel):
    table_name: str | None = None
    num_rows: int = 1000

@app.post("/read_db/")
async def read_db(request: ReadDBRequest):
    # DataLoaderTransactionDBのインスタンスを生成
    db_loader = DataLoaderTransactionDB()

    # read_dbメソッドを呼び出してデータを読み込む
    try:
        df = db_loader.read_db(request.table_name, request.num_rows)
        # pandas DataFrameをJSON形式に変換して返す
        return df.to_dict(orient="records")
    except Exception as e:
        # エラー処理
        raise HTTPException(status_code=500, detail=str(e))

class ReadDBMarketDataRequest(BaseModel):
    num_rows: int = 1000

@app.post("/read_db_market_data/")
async def read_db_market_data(request: ReadDBMarketDataRequest):

    # DataLoaderTransactionDBのインスタンスを生成
    db_loader = DataLoaderDB()
    conf =  get_config("ONLINE")
    symbol = conf["SYMBOL"]
    interval = conf["INTERVAL"]
    table_name = f"{symbol}_{interval}_market_data_tech"

    # read_db_market_dataメソッドを呼び出してデータを読み込む
    try:
        df = db_loader.load_recent_data_from_db(table_name, request.num_rows)
        df = df.filter(["start_at", "close","upper2","middle","lower2"])
        # pandas DataFrameをJSON形式に変換して返す
        return df.to_dict(orient="records")
    except Exception as e:
        # エラー処理
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_fxtransaction_data/")
async def read_fxtransaction_data(request: ReadDBMarketDataRequest):

    # DataLoaderTransactionDBのインスタンスを生成
    db_loader = DataLoaderTransactionDB()
    conf =  get_config("ONLINE")
    symbol = conf["SYMBOL"]
    table_name = f"{symbol}_fxtransaction"

    # read_db_market_dataメソッドを呼び出してデータを読み込む
    try:
        df = db_loader.read_db(table_name, request.num_rows)
        # pandas DataFrameをJSON形式に変換して返す
        return df.to_dict(orient="records")
    except Exception as e:
        # エラー処理
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_account_data/")
async def read_account_data(request: ReadDBMarketDataRequest):

    # DataLoaderTransactionDBのインスタンスを生成
    db_loader = DataLoaderTransactionDB()
    conf =  get_config("ONLINE")
    symbol = conf["SYMBOL"]
    table_name = f"{symbol}_account"

    # read_db_market_dataメソッドを呼び出してデータを読み込む
    try:
        df = db_loader.read_db(table_name, request.num_rows)
        # pandas DataFrameをJSON形式に変換して返す
        return df.to_dict(orient="records")
    except Exception as e:
        # エラー処理
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


