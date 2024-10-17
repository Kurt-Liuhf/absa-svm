from MAFC import FeatureConstruction as FC
from MAFC.properties.properties import theproperty
def featureConstruction(dataset, maxnum = 5000, isother = True):
    data = FC.FC(dataset, isother = isother)
    data = data.copy()
    return data