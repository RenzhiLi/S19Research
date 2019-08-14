import Embedding
import BiGRU
from sklearn.externals import joblib
from torch.utils.data import DataLoader, Dataset
import torch

def queryVec(query):
    q = Embedding.queryEncoder(query)
    q = BiGRU.queryGRU(q)
    return q

def docVec(doc):
    d = Embedding.docEncoder(doc)
    d = BiGRU.documentGRU(d)
    return d

def batchQVec(querys):
    output = querys
    for i in range(len(output)):
        output[i]['qVec'] = queryVec(output[i]['q'])
    return output

def batchDVec(docs):
    output = docs
    for i in range(len(output)):
        output[i]['dVec'] = docVec(output[i]['plainText'])
    return output

class qTrainSet(Dataset):
    def __init__(self,querys):
        self.querys = querys
    def __getitem__(self,index):
        return self.querys[index]['qVec'].requires_grad_(False)
    def __len__(self):
        return len(self.querys)

class docTrainSet(Dataset):
    def __init__(self,docs):
        self.docs = docs
    def __getitem__(self,index):
        return self.docs[index]['dVec']
    def __len__(self):
        return len(self.docs)

def saveVecs(querys,docs):
    joblib.dump(querys,"TestQuerys.pkl")
    joblib.dump(docs,"TestDocs.pkl")
