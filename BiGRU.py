import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.externals import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def queryGRU(query):
    query = query.reshape((query.shape[0],1,query.shape[1]))
    gRU = nn.GRU(50, 25, bidirectional = True)
    output, _ = gRU(torch.tensor(query,dtype = torch.float))
    output = output.reshape((query.shape[0],query.shape[2]))
    return output

def documentGRU(doc):
    gRU = nn.GRU(50, 25, bidirectional = True)
    output = []
    for sen in doc:
        senIn = sen.reshape((sen.shape[0],1,sen.shape[1]))
        senOut, _ = gRU(torch.tensor(senIn,dtype = torch.float))
        senOut = senOut.reshape((senOut.shape[0],senOut.shape[2]))
        output.append(senOut)
    return output

def saveCurrentWork(query,doc):
    joblib.dump(query,"query_u_test.pkl")
    joblib.dump(doc,"doc_u_test.pkl")

def test1():
    pass

if __name__ == '__main__':
    pass