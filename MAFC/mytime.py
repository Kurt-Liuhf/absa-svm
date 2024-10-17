import datetime


def getNowTimeStr():
    thetime = str(datetime.datetime.now()).replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    return thetime

def getNowTime():
    return str(datetime.datetime.now())