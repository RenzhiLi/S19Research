# weights
cw = np.ones((3*NUM_Feature))
dW1 = 1
dw1 = np.ones((4*NUM_Feature))
dW2 = 1
dw2 = np.ones((4*NUM_Feature))

def crossAttn():
    output = []
    for senNum in range(len(doc)):
        sen = doc[senNum]
        sXY = np.zeros((sen.shape[0], query.shape[0]))
        for x in range(sen.shape[0]):
            for y in range(query.shape[0]):
                conc = np.hstack((sen[x],query[y],sen[x]*query[y]))
                sXY[x][y] = np.dot(cw.T,conc)
        sD2Q = softmax(sXY, 0)
        sQ2D = softmax(sXY, 1)
        aD2Q = np.dot(sD2Q,query)
        aQ2D = np.dot(np.dot(sD2Q,sQ2D.T),sen)
        V = np.hstack((sen,aD2Q,sen*aD2Q,sen*aQ2D))
        output.append(V)
    return output

def queryInnerAttn():
    cExp = []
    for i in range(query.shape[0]):
        c = np.dot(qw.T, np.tanh(qW * query[i]))  
        cExp.append(math.exp(c))
    deno = sum(cExp)
    alpha = []
    for i in range(query.shape[0]):
        alpha.append(cExp[i]/deno)
    z = np.zeros((NUM_Feature))
    for i in range(query.shape[0]):
        z = z + alpha[i]*query[i]
    return z

def docHInnerAttn():
    V = crossAttn()
    # level1: attention in sentence
    x = []
    for sen in V:
        cExp = []
        for i in range(sen.shape[0]):
            c = np.dot(dw1.T, np.tanh(dW1 * sen[i]))
            cExp.append(math.exp(c))
        deno = sum(cExp)
        alpha = []
        for i in range(sen.shape[0]):
            alpha.append(cExp[i] / deno)
        xi = np.zeros((4*NUM_Feature))
        for i in range(sen.shape[0]):
            xi = xi + alpha[i]*sen[i]
        x.append(xi)
    # level2: attention for doc
    bExp = []
    for sent in x:
        b = np.dot(dw2.T, np.tanh(dW2 * sent))
        bExp.append(math.exp(b))
    denol2 = sum(bExp)
    beta = []
    for i in bExp:
        beta.append(i / denol2)
    y = np.zeros((4*NUM_Feature))
    for i in range(len(x)):
        y = y + beta[i]*x[i]
    return y