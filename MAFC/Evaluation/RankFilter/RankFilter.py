from MAFC_Operator.ColumnInfo import ColumnInfo
from MAFC_Operator.Operators import Operators
from properties.properties import theproperty

class RankFilter:
    def __init__(self):
        pass

    def rankAndFilter(self, operatorslist, previousiterchosenatts = None):
        operatorslist.sort(key=lambda ob: ob.getFScore(), reverse=True)
        newopertorslist = list(filter(lambda ops: ops.getFScore() > theproperty.fsocre, operatorslist))
        return newopertorslist

    def rankAndWrapper(self, operatorslist, previousiterchosenatts= None):
        operatorslist.sort(key=lambda ob: ob.getWScore(), reverse=True)
        newopertorslist = list(filter(lambda ops: ops.getWScore() > theproperty.wsocre, operatorslist))
        return newopertorslist
