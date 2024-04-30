import sys
from datetime import datetime

def setup_sys_path():
    """
    システムのパスを設定します。

    現在のスクリプトが存在するディレクトリの親ディレクトリをシステムパスに追加します。
    これにより、親ディレクトリにあるモジュールやパッケージをインポートできるようになります。
    """
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


def format_dates(start_date, end_date) -> tuple:
    """
    開始日と終了日をフォーマットし、その形式を検証する。
    必要に応じて時間とタイムゾーンを追加する。

    Args:
        start_date (str): 開始日。
        end_date (str): 終了日。

    Returns:
        tuple: (フォーマットされた開始日, フォーマットされた終了日)
    """
    formatted_start_date = append_time_if_missing(start_date)
    formatted_end_date = append_time_if_missing(end_date)

    for date in [formatted_start_date, formatted_end_date]:
        validate_date_format(date)

    return formatted_start_date, formatted_end_date


def append_time_if_missing(date) -> str:
    """
    日付文字列に時間とタイムゾーンが含まれていない場合、デフォルトの時間とタイムゾーンを追加する。

    Args:
        date (str): 日付文字列。

    Returns:
        str: 時間とタイムゾーンが追加された日付文字列。
    """
    return date + " 00:00:00+0900" if len(date) == 10 else date


def validate_date_format(date):
    """
    日付文字列の形式が正しいか検証する。無効な場合はエラーメッセージを出力し、プログラムを終了する。

    Args:
        date (str): 検証する日付文字列。
    """
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        exit_with_message(
            "無効な日付形式です。YYYY-MM-DD または YYYY-MM-DD HH:MM:SS+ZZZZ を使用してください")


def exit_with_message(message):
    """
    エラーメッセージを出力し、ステータスコード1でプログラムを終了する。

    Args:
        message (str): 出力するメッセージ。
    """
    print(message)
    sys.exit(1)



def get_config_fullpath():
    """
    環境変数から設定ファイルのパスを取得します。

    Returns:
        str: 設定ファイルのパス。
    """
    import os
    # 現在のスクリプトのディレクトリの絶対パスを取得
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # 設定ファイルが存在するディレクトリのパスを構築（親ディレクトリ）
    config_file_directory = os.path.join(current_script_path, os.pardir)
    # 設定ファイルの相対パスを構築
    config_path = os.path.join(config_file_directory, 'aitrading_settings_ver2.json')
    # 相対パスを絶対パスに変換
    config_path = os.path.abspath(config_path)
    return config_path

def get_config(tag=None):
    """
    設定ファイルを開いてJSONとして読み込みます。

    Returns:
        dict: 設定情報。
    """
    import json
    config_path = get_config_fullpath()
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    if tag is not None:
        return config_data[tag]
    return config_data
