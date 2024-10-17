from abc import abstractmethod
from MAFC_Operator.operator_base import outputType
from Evaluation.Evaluation import Evaluation
from MAFC_Operator.Unary.discretizer import Discretizer
from MAFC_Operator.ColumnInfo import ColumnInfo
from logger.logger import logger
class FEvaluation(Evaluation):
    def __init__(self):
        super(FEvaluation, self).__init__()
        self.analycolumns = []

    @abstractmethod
    def recalculateDatasetBasedFeatures(self, datadict):
        pass

    def getType(self):
        return self.evalutionType.Filte

    @abstractmethod
    def needToRecalcScore(self) -> bool:
        pass

    def discretizeColumns(self, datadict, bins):
        try:
            flag = False
            for acs in self.analycolumns:
                if acs[2].getType() != outputType.Discrete:
                    discretizer = Discretizer(bins)
                    soucol = [{"name": acs[2].getName(), "type": acs[2].getType()}]
                    discretizer.processTrainingSet(datadict["data"], soucol, None)
                    newcolumn = discretizer.generateColumn(datadict["data"], soucol, None)
                    acs[0] = newcolumn["name"]
                    acs[1] = newcolumn["data"]
                    acs[2] = ColumnInfo([acs[2]], None, discretizer, acs[0], False, discretizer.getOutputType(), bins)
                    flag = True
        except Exception as ex:
            logger.Error(f"FEvaluation discretizeColumns Error: {ex}", ex)
        finally:
            return flag

    def initFEvaluation(self, columntoanalyze):
        self.analycolumns = columntoanalyze




