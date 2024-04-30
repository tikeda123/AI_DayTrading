from multiprocessing import Process, Queue
import time

# メッセージを送信するプロセス
def sender(queue, messages):
    for message in messages:
        print(f"Sending message: {message}")
        queue.put(message)
        time.sleep(1)  # メッセージ送信間のディレイ
    queue.put(None)  # 終了シグナル

# メッセージを受信するプロセス
def receiver(queue):
    while True:
        message = queue.get()
        if message is None:
            break  # 終了シグナルを受け取ったらループを抜ける
        print(f"Received message: {message}")

if __name__ == "__main__":
    # メッセージキューの作成
    queue = Queue()

    # 送信プロセスと受信プロセスの開始
    sender_process = Process(target=sender, args=(queue, ["hello", "world", "test"]))
    receiver_process = Process(target=receiver, args=(queue,))

    # プロセスの開始
    sender_process.start()
    receiver_process.start()

    # プロセスの終了待ち
    sender_process.join()
    receiver_process.join()

    print("Processing complete")
