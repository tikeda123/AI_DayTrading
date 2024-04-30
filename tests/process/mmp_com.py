#
# Python Script with Base Class
# for Event-Based Backtesting
#
# Python for Algorithmic Trading
# (c) Toshihiko Ikeda
# The Python Quants GmbH
#

import mmap
import time
import sys

START_PROCESS_FLAG = b"01"
STOP_PROCESS_FLAG = b"11"
RESET_PROCESS_FLAG = b"00"
MMP_FILENAME = 'mmp_commandfile.txt'

class mmp_communication:
    def __init__(self, filename: str):
        self.__proc_sys_filename = filename

    def get_command(self):
        try:
            with open(self.__proc_sys_filename, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm.seek(0)  # メモリmmを頭から読み出し(seek(0))、読み出した値をmmに格納する。
                mode = mm.readline()  # メモリmmの内容を変数modeに書き出す。
                return mode
        except FileNotFoundError:
            print(f'File not found: {self.__proc_sys_filename}')
            exit()

    def set_commnad(self, cmd):
        try:
            with open(self.__proc_sys_filename, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm[:5] = cmd
        except FileNotFoundError:
            print(f'File not found: {self.__proc_sys_filename}')
            exit()


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print('There are no arguments.')
        print('usage: python mmp_com.py <command>')
        exit(0)

    cmd = args[1]
    mmp = mmp_communication(MMP_FILENAME)

    if cmd == 'start':
        mmp.set_commnad(START_PROCESS_FLAG)
        print('start process.')

    elif cmd == 'stop':
        mmp.set_commnad(STOP_PROCESS_FLAG)
        print('stop process.')
