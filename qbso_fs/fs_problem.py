from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier , VotingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier , LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.svm import SVC
from thundersvm import SVC as TSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.preprocessing import StandardScaler
from qbso_fs.solution import Solution

class FsProblem :
    def __init__(self, typeOfAlgo, train_X=None, train_y=None, test_X=None, test_y=None, qlearn = None, numsfea = 0, params = None, modelname = None, gpuid = 0, max_iter=500):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.nb_attribs = numsfea # The number of features is the size of the dataset - the 1 column of labels
        # self.outPuts=self.data.iloc[:,self.nb_attribs] # We initilize the labels from the last column of the dataset
        self.ql = qlearn
        self.typeOfAlgo = typeOfAlgo
        self.gpuid = gpuid
        self.maxiter = max_iter
        
        self.classifier = self.initclassifier(modelname, params)
        


    def initclassifier(self, modelname, params):
        if modelname is None:
            return None
        # print("modelname: ", modelname)
        if "LibSVM" in modelname:
            clf = SVC(**params, random_state=42, max_iter=self.maxiter)
        elif "rf" in modelname:
            clf = RandomForestClassifier(**params, random_state=42, n_jobs=8)
        elif "gbdt" in modelname:
            clf = GradientBoostingClassifier(**params, random_state=42)
        elif "xgb" in modelname:
            clf = SGDClassifier(**params, random_state=42, n_jobs=8, max_iter=self.maxiter)
        elif "mlp" in modelname:
            clf = MLPClassifier(**params, random_state=42, max_iter=self.maxiter)
        elif "knn" in modelname:
            clf = KNeighborsClassifier(**params, n_jobs=8)
        elif "nb" in modelname:
            clf = GaussianNB(**params)
        elif "lr" in modelname:
            clf = LogisticRegression(**params, random_state=42, n_jobs=8, max_iter=self.maxiter)
        elif "dt" in modelname:
            clf = DecisionTreeClassifier(**params, random_state=42)
        elif "et" in modelname:
            clf = ExtraTreesClassifier(**params, random_state=42, n_jobs=8)
        elif "ada" in modelname:
            clf = AdaBoostClassifier(**params, random_state=42)
        elif "bag" in modelname:
            clf = BaggingClassifier(**params, random_state=42, n_jobs=8)
        elif "voting" in modelname:
            clf = VotingClassifier(**params, n_jobs=8)
        elif "stacking" in modelname:
            clf = StackingClassifier(**params, n_jobs=8)
        else :
            clf = TSVC(**params, random_state=42, max_iter=self.maxiter, n_jobs=8, gpu_id=self.gpuid)
        return clf
        

    def evaluate2(self, solution):
        
        sol_list = Solution.sol_to_list(solution)
        if (len(sol_list) == 0):
            return 0
         
        X = self.train_X[:,sol_list]
        Y = self.train_y
        
        self.classifier.fit(X, Y)
        Xtest = self.test_X[:,sol_list]
        predict= self.classifier.predict(Xtest) 
        return metrics.accuracy_score(predict, self.test_y)


    def evaluate(self, solution):
        sol_list = Solution.sol_to_list(solution)
        if (len(sol_list) == 0):
            return 0
         
        X = self.train_X[:,sol_list]
        Y = self.train_y
        
        self.classifier.fit(X, Y)
        Xtest = self.test_X[:,sol_list]
        predict= self.classifier.predict(Xtest) 
        return metrics.accuracy_score(predict, self.test_y)
    
    def getPred(self, solution):
        sol_list = Solution.sol_to_list(solution)
        if (len(sol_list) == 0):
            return 0
         
        X = self.train_X[:,sol_list]
        Y = self.train_y
        
        self.classifier.fit(X, Y)
        Xtest = self.test_X[:,sol_list]
        predict= self.classifier.predict(Xtest)
        return predict