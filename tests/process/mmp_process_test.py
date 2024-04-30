import mmap
import time
import os
import sys
from multiprocessing import Process

START_PROCESS_FLAG = b"01\n"
STOP_PROCESS_FLAG = b"11\n"
RESET_PROCESS_FLAG = b"00\n"
MMP_FILENAME = 'mmp_commandfile.txt'

def set_command(cmd):
    """
    指定されたコマンドを共有メモリファイルに書き込む関数
    """
    # ファイルが存在しない、または空の場合は初期化する
    if not os.path.exists(MMP_FILENAME) or os.path.getsize(MMP_FILENAME) == 0:
        with open(MMP_FILENAME, "wb") as f:
            f.write(RESET_PROCESS_FLAG)  # ここでファイルを初期化

    with open(MMP_FILENAME, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.seek(0)
        mm.write(cmd)
        mm.flush()
        mm.close()

def worker():
    """
    仕事をするプロセスの関数
    終了コマンドが設定されるまでループします
    """
    print("Worker process started")
    while True:
        with open(MMP_FILENAME, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.seek(0)
            mode = mm.readline()
            mm.close()

            if mode == STOP_PROCESS_FLAG:
                print("Worker process stopping")
                break
            elif mode == RESET_PROCESS_FLAG:
                print("Reset command received, continuing")

            # 仕事をする（ここでは単に待つだけ）
            time.sleep(1)

if __name__ == '__main__':
    # 共有メモリファイルをリセット
    set_command(RESET_PROCESS_FLAG)

    # ワーカープロセスを生成して開始
    p = Process(target=worker)
    p.start()

    # メインプロセスはユーザーの入力を待つ
    cmd = input("Enter command (start/stop/reset): ")
    while cmd != 'stop':
        if cmd == 'start':
            set_command(START_PROCESS_FLAG)
        elif cmd == 'reset':
            set_command(RESET_PROCESS_FLAG)
        cmd = input("Enter command (start/stop/reset): ")

    # ワーカープロセスを停止
    set_command(STOP_PROCESS_FLAG)
    p.join()  # ワーカープロセスの終了を待つ
    print("Main process ended")
