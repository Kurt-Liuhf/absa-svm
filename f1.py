from sklearn import metrics

def loadfile(path):
    y_pred = []
    y_true = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            y_true.extend([int(x) for x in lines[i].split(',')])
            y_pred.extend([int(x) for x in lines[i+1].split(',')])
    return y_true, y_pred

if __name__ == "__main__":
    path = 'rest.result'
    y_true, y_pred = loadfile(path)
    print(print(metrics.classification_report(y_true, y_pred)))
    print("100")
