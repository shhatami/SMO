import numpy as np
import matplotlib.pyplot as plt
import random



def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.zeros((m,1))
    if (kTup[0]=='lin') :   #Linear kernel
        K = X * A.T
    elif (kTup[0]=='rbf') : #Gaussian kernel (radial bias function)
        for i in range(m):
            deltaRow = X[i,:] - A
            K[i] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))  #Applied on each element of K
    else:
        raise NameError("No kernel of such name defined.")
    return K



#Container for the model used for sequential minimal optimization
class smoUtil:
    def __init__(self, X, labels, c, tolerence, kTup):
        self.X = X  #training data matrix
        self.labels = labels  #label matrix. It is a column matrix.
        self.c = c  #regularization parameter
        self.tol = tolerence
        self.m = len(self.X)    #Size of training set
        self.alphas = np.zeros((self.m,1))    #lagrange multiplicator matrix. It is a column matrix.
        self.b = 0  #scalar bias term
        self.eCache = np.zeros((self.m,2))    #error cache. 1st col states whether eCache is valid and 2nd col contains actual E value.
        self.K = np.zeros((self.m,self.m))    #number of landmarks will be equal to the number of training samples, and each landmark will contribute to a feature of transformed training sample.
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)[0]

        
        
def calcEk(oS, k):
    #np.multiply() function multiply elements of matrices and * operator multiplies matrices
    fXk = float(np.multiply(oS.alphas, oS.labels).T * np.matrix(oS.K[:,k]).transpose()) + oS.b
    Ek = fXk - float(oS.labels[k])
    return Ek



def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]



def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(random.uniform(0,m))
    return j



def selectJ(i, oS, Ei):
    j = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i,:] = [1,Ei] #Set Ei to be valid in eCache. Valid means it has been calcuated.
    validECacheList = np.nonzero(oS.eCache[:,0])[0] #returns the row index of valid eCache in array format.
    if len(validECacheList) > 1 : #Select j which maximizes Ei-Ej
        for k in validECacheList:
            if k==i :
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                j = k
                maxDeltaE = deltaE
                Ej = Ek
    else:   #When looped for first time, select random j
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej



def innerL(i ,oS):
    Ei = calcEk(oS, i)
    #Preceed if error is within specified tolerance
    if ((oS.labels[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.c)) or ((oS.labels[i]*Ei > oS.tol) and (oS.alphas[i]>0)) :
        j, Ej = selectJ(i, oS, Ei)
        alphaIOld = oS.alphas[i].copy()
        alphaJOld = oS.alphas[j].copy()
        if (oS.labels[i] != oS.labels[j]) :
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.c, oS.c + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.c)
            H = min(oS.c, oS.alphas[j] + oS.alphas[i])
        if L==H :
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta < 0 :
            oS.alphas[j] = oS.alphas[j] - oS.labels[j]*(Ei-Ej)/eta
            if oS.alphas[j] > H :
                oS.alphas = H
            elif oS.alphas[j] < L :
                oS.alphas[j] = L
        else :
            #
            #
            #
            return 0
            #
            #
            #
        updateEk(oS, j)
        if (abs(oS.alphas[j,0] - alphaJOld[0]) < 0.00001) :
            print("j not moving enough")
            return 0
        s = oS.labels[i]*oS.labels[j]
        oS.alphas[i] = oS.alphas[i] + s*(alphaJOld - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labels[i]*(oS.alphas[i] - alphaIOld)*oS.K[i,i] - oS.labels[j]*(oS.alphas[j] - alphaJOld)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labels[i]*(oS.alphas[i] - alphaIOld)*oS.K[i,j] - oS.labels[j]*(oS.alphas[j] - alphaJOld)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.c > oS.alphas[i]) :
            oS.b = b1
        elif (0< oS.alphas[j]) and (oS.c > oS.alphas[j]) :
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0



def smoP(myData, myLabels, c, tolerence, maxIter, kTup=('lin',0)):
    oS = smoUtil(np.matrix(myData), np.matrix(myLabels).transpose(), c, tolerence, kTup)
    numiter = 0
    iterEntireSet = True
    alphaPairsChanged = 0
    while (numiter < maxIter) and ((alphaPairsChanged > 0) or (iterEntireSet)):
        alphaPairsChanged = 0
        if iterEntireSet :
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("Entire Set, numiter: %d, i: %d, alpha pairs changed: %d" %(numiter,i,alphaPairsChanged))
            numiter += 1
        else :
            nonBoundIs = np.where((oS.alphas != 0) & (oS.alphas != oS.c))[0]
            for i in nonBoundIs :
                alphaPairsChanged += innerL(i, oS)
                print("Non Bound, numiter: %d, i: %d, alpha pairs changed: %d" %(numiter,i,alphaPairsChanged))
            numiter += 1
        if iterEntireSet :
            iterEntireSet = False
        elif (alphaPairsChanged==0) :
            iterEntireSet = True
        print("numiter number: %d" %numiter)
    return oS.b, oS.alphas



def calcWs(alphas, myData, myLabels):
    X = np.matrix(myData)
    labelMatrix = np.matrix(myLabels).transpose()
    m,n = np.shape(X);
    w = np.zeros((n,1)) #weight vector, perpendicular to hyperplane.
    for i in range(m):
        w += np.multiply(alphas[i]*labelMatrix[i], X[i,:].T)
    return w



def loadDataSet(filename):
    myData = [];
    myLabel = [];
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        myData.append( [float(lineArr[0]), float(lineArr[1])] )
        myLabel.append( float(lineArr[2]) )
    return myData, myLabel



def trainAndtestSvm(sigma=1.3):
    myData, myLabels = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(myData, myLabels, 200, 0.0001, 10000, ('rbf', sigma))
    dataMatrix = np.matrix(myData)
    labelMatrix = np.matrix(myLabels).transpose()
    m,n = np.shape(dataMatrix)
    svIndexs = np.nonzero(alphas.A > 0)[0]
    svMatrix = dataMatrix[svIndexs]
    svLabel = labelMatrix[svIndexs]
    print("There are %d Support Vectors" %(np.shape(svMatrix)[0]))
    errorCount = 0
    for i in range(m):
        kernelEval = kernalTrans(svMatrix, dataMatrix[i,:],('rbf', sigma))
        predict = kernelEval.T * np.multiply(svLabel, alphas[svIndexs]) + b
        if (np.sign(predict) != np.sign(labelMatrix[i])):
            errorCount += 1
    print("The training error rate is: %f" %(float(errorCount)/m))
    myData, myLabels = loadDataSet('testSetRBF2.txt')
    dataMatrix = np.matrix(myData)
    labelMatrix = np.matrix(myLabels).transpose()
    m,n = np.shape(dataMatrix)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svMatrix, dataMatrix[i,:],('rbf',sigma))
        predict = kernelEval.T * np.smultiply(svLabel, alphas[svIndexs]) + b
        if (np.sign(predict) != np.sign(labelMatrix[i])):
            errorCount += 1
    print("The testing error rate is: %f" %(float(errorCount)/m))
