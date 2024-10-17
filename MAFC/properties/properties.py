
class properties:
    def __init__(self):
        '''
        :param
        :param
        :param
        :param
        :param
        :param
        :param
        :param
        :param

        '''
        self.unaryoperator = ['DayofWeek', 'Discretizer', 'HourofDay', 'IsWeekend', 'StdOperator']
        self.otheroperator = ['AddOperator', 'SubOperator', 'MultiOperator', 'DivisOperator']#, 'GroupMax', 'GroupMean', 'GroupMin', 'GroupStd', 'GroupCount']
        self.DiscretizerBinsNumber = 10 #将实数离散化后的类数
        self.maxcombination = 1
        self.classifier = 'RandomForest'
        self.datasetlocation = "data/datasets/"
        self.filter = "MLFEvaluation"
        self.wrapper = "AucWrapperEvaluation"
        self.thread = 8
        self.fsocre = 0.1
        self.wsocre = 0.01
        self.maxevaluationattsperiter = 1500
        self.backmodelpath = "data/model/"
        self.datasetname = "Alldata2000"
        self.classifiersforMLAttributes = ["RandomForest"]
        self.targetindex = -1
        self.image_path = None
        self.resultfilepath = "data/result/"
        self.temppath = "data/temp/"
        self.targetmutil = False
        self.targetclasses = 2
        self.extendpath = "test/mymafcoper.py"
        self.finalchosenopspath = "data/finalchosen/"
        self.dataframe = "pandas"
        self.maxFEvaluationnums = 10000
        self.randomseed = 1
        self.rootpath = "~/code/absa-svm//MAFC/"
        self.maxoperators = 5000
        self.mutilprocess = True
if 'theproperty' in globals():
    print('theproperty has exist!')
else:    
    theproperty = properties()