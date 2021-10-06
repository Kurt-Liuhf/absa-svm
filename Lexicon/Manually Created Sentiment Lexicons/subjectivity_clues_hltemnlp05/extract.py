myDic = {}

def score(priorpolarity, type):
    if priorpolarity == "negative":
        if type == "weaksubj":
            return "-0.5"
        elif type == "strongsubj":
            return "-1.0"
    elif priorpolarity == "positive":
        if type == "weaksubj":
            return "0.5"
        elif type == "strongsubj":
            return "1.0"
    # else: # neutral both
    #     return "0.0"

with open("subjclueslen1-HLTEMNLP05.tff", encoding='utf-8') as f:
    for i in f:
        i = i.split() # type len word pos stemmed priorpolarity
        if i[-1].split("=")[1] != "positive" and i[-1].split("=")[1] != "negative":
            continue
        word = i[2].split("=")[1]
        if word in myDic.keys():
            continue
        sent_score = score(i[-1].split("=")[1], i[0].split("=")[1])
        myDic[word] = sent_score
        # myDic.append(word+"\t"+sent_score+"\n")

myList = ["\t".join([k,v])+"\n" for k,v in myDic.items()]

with open("DCU_MPQA.txt", 'w+') as fp:
    fp.writelines(myList)