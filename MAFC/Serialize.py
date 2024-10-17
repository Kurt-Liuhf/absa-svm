import pickle
from logger.logger import logger
def serialize(filepath, obj):
    '''
    将模型序列化
    :param filepath:
    :param obj:
    :return:
    '''
    try:
        with open(filepath, "wb") as file:
            pickle.dump(obj, file)
    except Exception as ex:
        logger.Error(f"serialize error: {ex}" ,ex)

def deserialize(filepath):
    '''
    将模型反序列化
    :param filepath:
    :return:
    '''
    try:
        with open(filepath, "rb") as file:
            obj = pickle.load(file)
            return obj
    except Exception as ex:
        logger.Error(f"deserialize error: {ex}", ex)