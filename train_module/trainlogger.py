import logging
import datetime
import os

def train_logger():
    train_logger = logging.getLogger("train_logger")    #logger名loggerを取得
    train_logger.setLevel(logging.DEBUG)  #loggerとしてはDEBUGで

    #handlerを作成
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    d = now.strftime('%Y%m%d%H%M%S')
    logdir="./Log"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    handler = logging.FileHandler(filename=f"./Log/train_{d}.log")  #handlerはファイル出力
    # handler = logging.FileHandler(filename=f"./Log/train.log")  #handlerはファイル出力

    handler.setLevel(logging.INFO)     #handlerはLevel.INFO以上
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    #loggerにハンドラを設定
    train_logger.addHandler(handler)

    return train_logger,d