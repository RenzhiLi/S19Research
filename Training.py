import numpy as np
import torch
import Attention as attn
from sklearn.externals import joblib
import os
import Main
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

MARGIN = torch.tensor(5.0)
NUM_Iters = 5

learning_rate = 2

if os.path.exists("coreModel.pth"):
    mod = torch.load("coreModel.pth")
else:
    mod = attn.wAttnModule()
opt = torch.optim.Adadelta(mod.parameters(),lr = learning_rate)

# mod = mod.to(device)

'''
data structure for training data:
query - docNumber for positive doc
documents
'''

def training(querys, docs):
    global mod, opt
    for t in range(NUM_Iters):
        ttloss = 0
        for i in querys:
            opt.zero_grad()
            scores = []
            for j in docs:
                s = mod.forward(i['qVec'],j['dVec'])
                scores.append(s)
            loss = loss_fn(scores,i)
            ttloss += loss
            print(loss,list(mod.parameters())[0])
            loss.backward()
            opt.step()
        #Main.MainWindow.ui.label2.setText("epoch" + str(t) + ",query" + str(i) +', loss=' + str(ttloss))
        print("epoch" + str(t) + ",query" + str(i) +', loss=' + str(ttloss))
    torch.save(mod,"coreModel.pth")
    return mod
    

def loss_fn(scores,query,margin = MARGIN):
    l = torch.tensor(0.0,requires_grad = True)
    scoreP = scores[query['docNum']]
    for i in range(len(scores)):
        if i != query['docNum']:
            ll = F.relu(margin - scoreP + scores[i])
            l = l + ll
    return l

