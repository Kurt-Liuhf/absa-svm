import os, sys
from datetime import datetime

class logger:
    @staticmethod
    def Log(msg: str):
        print(msg)

    @staticmethod
    def Info(msg: str):
        pass
        # print(msg, datetime.now().ctime())

    @staticmethod
    def Error(msg: str, ex: Exception = None):

        excetype, exceobj, excetb = sys.exc_info()
        fname = os.path.split(excetb.tb_frame.f_code.co_filename)[1]
        print(excetype, fname, excetb.tb_lineno)
        print(msg, datetime.now().ctime())

    @staticmethod
    def Warnning(msg: str):
        print(f'Warnning : {msg} {datetime.now().ctime()}')
