#!/bin/bash

# Pythonスクリプトのパス

# Python仮想環境のアクティベーション（必要な場合）
# source /path/to/your/venv/bin/activate

# Pythonスクリプトをバックグラウンドで実行し、nohupでプロセスをデタッチし、標準出力と標準エラー出力をログファイルにリダイレクトする
nohup npm start > npm_output.log 2>&1 &

# ジョブIDを表示して確認
echo "Pythonスクリプトがバックグラウンドで実行されました。"

