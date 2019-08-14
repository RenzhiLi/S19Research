import numpy as np
from sklearn.externals import joblib
import math, torch
from torch import nn, optim
NUM_Feature = 50

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class crossAttnM(nn.Module):
    def __init__(self):
        super(crossAttnM,self).__init__()
        self.cl = nn.Linear(NUM_Feature*3, 1 ,bias=False)
        self.cl.weight.data.fill_(1)
    def forward(self,query,doc):
        output = []
        for senNum in range(len(doc)):
            sen = doc[senNum]
            sXY = torch.zeros(sen.shape[0], query.shape[0])
            for x in range(sen.shape[0]):
                for y in range(query.shape[0]):
                    conc = torch.cat((sen[x],query[y],sen[x]*query[y]),0) 
                    sXY[x][y] = self.cl(conc)
            sD2Q = nn.functional.softmax(sXY, dim = 0)
            sQ2D = nn.functional.softmax(sXY, dim = 1)
            aD2Q = torch.mm(sD2Q,query)
            aQ2D = torch.mm(torch.mm(sD2Q,sQ2D.t()),sen)
            V = torch.cat((sen,aD2Q,sen*aD2Q,sen*aQ2D),1)
            output.append(V)
        return output

class qInnerAttnM(nn.Module):
    def __init__(self):
        super(qInnerAttnM,self).__init__()
        self.ql = nn.Linear(NUM_Feature,1,bias=False)
        self.ql.weight.data.fill_(1)
        self.qW = nn.Parameter(torch.tensor(1.0))
    def forward(self,query):
        cExp = []
        for i in range(query.shape[0]):
            tan = torch.tanh(self.qW * query[i])
            c = self.ql(tan)
            cExp.append(math.exp(c))
        deno = sum(cExp)
        alpha = []
        for i in range(query.shape[0]):
            alpha.append(cExp[i]/deno)
        z = torch.zeros(NUM_Feature)
        for i in range(query.shape[0]):
            z = z + alpha[i]*query[i]
        return z

class docHInAttnM1(nn.Module):
    def __init__(self):
        super(docHInAttnM1,self).__init__()
        self.dl1 = nn.Linear(NUM_Feature*4,1,bias=False)
        self.dl1.weight.data.fill_(1)
        self.dW1 = nn.Parameter(torch.tensor(1.0))
    def forward(self,V):
        x = []
        for sen in V:
            cExp = []
            for i in range(sen.shape[0]):
                tan = torch.tanh(self.dW1 * sen[i])
                c = self.dl1(tan.float())
                cExp.append(math.exp(c))
            deno = sum(cExp)
            alpha = []
            for i in range(sen.shape[0]):
                alpha.append(cExp[i] / deno)
            xi = torch.zeros(4*NUM_Feature)
            for i in range(sen.shape[0]):
                xi = xi + alpha[i]*sen[i].float()
            x.append(xi)
        return x

class docHInAttnM2(nn.Module):
    def __init__(self):
        super(docHInAttnM2,self).__init__()
        self.dl2 = nn.Linear(NUM_Feature*4,1,bias=False)
        self.dl2.weight.data.fill_(1)
        self.dW2 = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        bExp = []
        for sent in x:
            tan = torch.tanh(self.dW2 * sent)
            b = self.dl2(tan)
            bExp.append(math.exp(b))
        deno2 = sum(bExp)
        beta = []
        for i in bExp:
            beta.append(i / deno2)
        y = torch.zeros(4*NUM_Feature)
        for i in range(len(x)):
            y = y + beta[i]*x[i]
        return y

class senScoreM(nn.Module):
    def __init__(self, num_feature = NUM_Feature):
        super(senScoreM,self).__init__()
        self.l1 = nn.Linear(num_feature*4,1)
        self.l1.weight.data.fill_(1)
        self.l1.bias.data.fill_(0)
    def forward(self,x):
        return self.l1(x)

class finalScoreM(nn.Module):
    def __init__(self, num_feature = NUM_Feature, drop = 0.05):
        super(finalScoreM,self).__init__()
        self.l1 = nn.Linear(num_feature*4,num_feature)
        self.l2 = nn.Linear(num_feature,1)
        self.l1.weight.data.fill_(1)
        self.l2.weight.data.fill_(1)
        self.l1.bias.data.fill_(0)
        self.l2.bias.data.fill_(0)
        self.drop = nn.Dropout(drop)
    def forward(self, z, y):
        y = self.drop(y)
        y = self.l1(y.float())
        y = y * z.float()
        y = self.drop(y)
        return float(self.l2(y))

class wAttnModule(nn.Module):
    def __init__(self):
        super(wAttnModule,self).__init__()
        self.crsA = crossAttnM()
        self.qI = qInnerAttnM()
        self.dI1 = docHInAttnM1()
        self.dI2 = docHInAttnM2()
        self.fS = finalScoreM(drop = 0)
    def forward(self,query,doc):
        V = self.crsA(query,doc)
        z = self.qI(query)
        x = self.dI1(V)
        y = self.dI2(x)
        return self.fS(z,y)

def test():
    pass

def checkweights(m):
    for n,p in m.named_parameters():
        print(n,p.shape)

if __name__ == "__main__":
    a = wAttnModule()
    for i in a.parameters():
        print(i)
        