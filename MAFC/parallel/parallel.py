from joblib import Parallel, delayed
from properties.properties import theproperty
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from logger.logger import logger
from multiprocessing import Process, Pool


def palallelForEach(function, listofargs):
    res = Parallel(n_jobs=theproperty.thread, backend="threading")(delayed(function)(arg) for arg in listofargs)
    return res


def ParallelForEachShare(function, listofargs):
    res = Parallel(n_jobs=theproperty.thread, require='sharedmem', backend="threading")(delayed(function)(arg) for arg in listofargs)
    return res


class MyThreadPool:
    def __init__(self, threadnums, operators, maxops=0, opername=None, infosep=1000):
        self.operators = operators
        self.threadnums = threadnums
        self.pool = ThreadPoolExecutor(max_workers=threadnums)
        if maxops == 0:
            self.maxops = len(operators)
        else:
            self.maxops = min(maxops, len(operators))
        self.iterops = 0
        self.iterthread = 0
        self.threadlist = []
        self.opername = opername
        self.infosep = infosep

    def run(self, myfunc, **kwargs):
        try:
            for _ in range(min(self.threadnums, self.maxops)):
                self.threadlist.append(self.pool.submit(myfunc, self.operators[self.iterops], **kwargs))
                #print(f"{self.iterops} run begin")
                self.iterops += 1
            while self.iterops < self.maxops:
                for _ in range(self.threadnums):
                    if self.iterops >= self.maxops:
                        break
                    for _ in as_completed([self.threadlist[self.iterthread]]):
                        #print(f"{self.iterthread} run end")
                        self.threadlist.append(self.pool.submit(myfunc, self.operators[self.iterops], **kwargs))
                        #print(f"{self.iterops} run begin")
                        self.iterops += 1
                    self.iterthread += 1
                    if self.iterthread % self.infosep == 0:
                        logger.Info(
                            self.opername + " this is " + str(self.iterthread) + " / " + str(self.maxops) + " and time is " + str(datetime.datetime.now()))
            while self.iterthread < self.maxops:
                for _ in as_completed([self.threadlist[self.iterthread]]):
                    #print(f"{self.iterthread} run end")
                    self.iterthread += 1
        except Exception as ex:
            logger.Error(f"run error: {ex}", ex)

class MyProcess(Process):

    def __init__(self, func, ops, kwargs):
        super(MyProcess, self).__init__()
        self.kwargs = kwargs
        self.func = func
        self.ops = ops

    def run(self):

        self.func(self.ops, **self.kwargs)

def myfunction(func, oop, iterops, kwargs):
    print(f"{iterops} run begin")
    func(oop, **kwargs)
    print(f"{iterops} run end")


class MutilProcess:

    def __init__(self, threadnums, operators, maxops=0, opername=None, infosep=1000):
        self.operators = operators
        self.threadnums = min(8, threadnums)
        self.pool = Pool(processes=threadnums)
        if maxops == 0:
            self.maxops = len(operators)
        else:
            self.maxops = min(maxops, len(operators))
        self.iterops = 0
        self.iterthread = 0
        self.threadlist = []
        self.opername = opername
        self.infosep = infosep

    def run(self, myfunc, **kwargs):
        try:
            operatorslist = []
            for _ in range(0, self.maxops):
                res = self.pool.apply_async(myfunc, (self.operators[self.iterops], kwargs))
                operatorslist.append(res)
                self.iterops += 1
                if self.iterops % self.infosep == 0:
                    logger.Info(
                        self.opername + " this is " + str(self.iterops) + " / " + str(
                            self.maxops) + " and time is " + str(datetime.datetime.now()))
            self.pool.close()
            self.pool.join()
            return operatorslist
            '''
            for _ in range(min(self.threadnums, self.maxops)):
                self.pool.apply_assync(myfunction, (self.operators[self.iterops], kwargs))
                print(f"{self.iterops} run begin")

                self.iterops += 1
            while self.iterops < self.maxops:
                for _ in range(self.threadnums):
                    if self.iterops >= self.maxops:
                        break
                    self.threadlist[self.iterthread].join()
                    print(f"{self.iterthread} run end")
                    self.pool.apply_assync(myfunction, (self.operators[self.iterops], kwargs))
                    #proc = MyProcess(func=myfunc, ops=self.operators[self.iterops], kwargs=kwargs)
                    #self.threadlist.append(proc)
                    #proc.start()
                    print(f"{self.iterops} run begin")
                    self.iterops += 1
                    self.iterthread += 1
                    if self.iterthread % self.infosep == 0:
                        logger.Info(
                            self.opername + " this is " + str(self.iterthread) + " / " + str(
                                self.maxops) + " and time is " + str(datetime.datetime.now()))
            while self.iterthread < self.maxops:
                self.threadlist[self.iterthread].join()
                print(f"{self.iterthread} run end")
                self.iterthread += 1
            '''
        except Exception as ex:
            logger.Error(f"run error: {ex}", ex)
