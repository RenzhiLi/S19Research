import numpy as np
from sklearn.externals import joblib
import math, torch
import nltk
import Attention as attn
import os
import torch
import Vectorizer

if os.path.exists("coreModel.pth"):
    mod = torch.load("coreModel.pth")
else:
    mod = attn.wAttnModule()

NUM_Feature = 50

def loadPlainDoc(path):
    with open(path,'r') as f:
        text = f.read()
    sents = nltk.sent_tokenize(text)
    return sents

def answerGen(query,docs):
    # using text query
    return answerGenL(Vectorizer.queryVec(query),docs)


def answerGenL(query,docs):
    # using vectorized params
    doc = None
    num, maxx = 0, -1000000
    for i in range(len(docs)):
        z = mod.forward(query,docs[i]['dVec'])
        if z > maxx:
            maxx, doc, num = z, docs[i], i
    return doc['ans']['text'],maxx
    
def senPair(query, doc, senNum = 3):
    senS = attn.senScoreM()
    V = mod.crsA.forward(query,doc['dVec'])
    x = mod.dI1.forward(V)
    s = []
    for sen in x:
        s.append(float(senS.forward(sen)))
    sDict = dict(zip(s, nltk.sent_tokenize(doc['plainText'])))
    sortkey = sorted(sDict.keys(),reverse = True)
    result = ""
    for i in range(senNum):
        result = result + str(sortkey[i]) + " | " + sDict[sortkey[i]] + "\n"
    return result

if __name__ == "__main__":
    pass