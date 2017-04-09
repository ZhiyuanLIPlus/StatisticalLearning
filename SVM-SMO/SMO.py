from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#线性核函数，对于J的求和被矩阵替代
def kernel(dataMatrix, i):
    return dataMatrix * dataMatrix[i, :].T

def smoComputation(dataMatrix, labelMatrix, C, err, maxIter):
    b = 0;
    m, n = shape(dataMatrix)#m: mums of examples, n: numb of features -> m=50, n=2
    alphas = mat(zeros((m, 1)))#alphas:m*1 -> 50*1
    iter = 0
    E = {}
    j_blocked = {}
    i_blocked = {}
    while (iter < maxIter):
        #alphaPairsChanged = 0
        #第1个变量i选择
        violatedMax, I = 0, -1
        for i in range(m):
            G_xi = float(multiply(alphas, labelMatrix).T * kernel(dataMatrix, i)) + b
            E[i] = float(G_xi - labelMatrix[i])
            #违反KKT判断:
            # alpha_i = 0       *****>> y_i * G_xi >= 1-e           *****>>  y_i * G_xi < 1-e
            # 0 < alpha_i < C   *****>> 1-e <= y_i * G_xi <= 1+e    *****>>  y_i * G_xi < 1-e or y_i * G_xi > 1+e
            # alpha_i = C       *****>> y_i * G_xi <= 1+e           *****>>  y_i * G_xi > 1+e
            if(alphas[i] == 0 and labelMatrix[i] * G_xi < 1-err):
                errAbsolu =  abs(1-err - labelMatrix[i] * G_xi)
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i
            if alphas[i] > 0 and alphas[i] < C and (labelMatrix[i] * G_xi < 1-err or labelMatrix[i] * G_xi > 1 + err):
                errAbsolu = max(abs(1 - err - labelMatrix[i] * G_xi), abs(1+e - labelMatrix[i] * G_xi))
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i
            if alphas[i] == C and labelMatrix[i] * G_xi > 1 + err:
                errAbsolu =  abs(1+err - labelMatrix[i] * G_xi)
                if errAbsolu > violatedMax and i not in i_blocked:
                    violatedMax = errAbsolu
                    I = i

        #第2个变量j选择
        J = -1
        if(I != -1):
            if(E[I] > 0):
                minE2 = max(E.values()) + 1
                for key in E:
                    if(E[key] < minE2 and key != I and key not in j_blocked):
                        minE2 = E[key]
                        J = key
            if(E[I] < 0):
                maxE2 = min(E.values()) - 1
                for key in E:
                    if (E[key] > maxE2 and key != I and key not in j_blocked):
                        maxE2 = E[key]
                        J = key
        if I == -1 and J == -1:
            break
        if I != -1 and J == -1:
            print("J work for All J in I:%d ---> choose another I" %I)
            j_blocked.clear()
            i_blocked[I] = 1
            continue
        # 第2个变量j选择
        alphaIold = alphas[I].copy()
        alphaJold = alphas[J].copy()
        if (labelMatrix[I] != labelMatrix[J]):
            L = max(0, alphaJold - alphaIold)
            H = min(C, C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold + alphaIold - C)
            H = min(C, alphaJold + alphaIold)
        if L == H:
            print("L == H for I:%d, J:%d --> choose another J" %(I, J))
            j_blocked[J] = 1
            continue
        eta = dataMatrix[I, :] * dataMatrix[I, :].T + dataMatrix[J, :] * dataMatrix[J, :].T
        eta += -2.0 * dataMatrix[I, :] * dataMatrix[J, :].T
        alphas_unCut = alphaJold + labelMatrix[J] * (E[I] - E[J]) / eta

        if (abs(clipAlpha(alphas_unCut, H, L) - alphaJold) < 0.0001):
            print("J not moving enough for I:%d, J:%d ---> choose another J" % (I, J))
            j_blocked[J] = 1
            continue

        #更新I，J
        alphas[J] = clipAlpha(alphas_unCut, H, L)
        alphas[I] = alphaIold + labelMatrix[J] * labelMatrix[I] * (alphaJold - alphas[J])

        #更新b
        b1 = b - E[I]
        b1 += - labelMatrix[I] * (alphas[I] - alphaIold) * dataMatrix[I,:] * dataMatrix[I,:].T
        b1 += - labelMatrix[J] * (alphas[J] - alphaJold) * dataMatrix[I,:] * dataMatrix[J,:].T

        b2 = b - E[J]
        b2 += - labelMatrix[I] * (alphas[I] - alphaIold) * dataMatrix[I,:] * dataMatrix[J,:].T
        b2 += - labelMatrix[J] * (alphas[J] - alphaJold) * dataMatrix[J,:] * dataMatrix[J,:].T

        if (0 < alphas[I]) and (C > alphas[I]):
            b = b1
        elif (0 < alphas[J]) and (C > alphas[J]):
            b = b2
        else:
            b = (b1 + b2) / 2.0
        print("iter: %d"%iter)
        print("i:%d from %f to %f"%(I, float(alphaIold), alphas[I]))
        print("j:%d from %f to %f" % (J, float(alphaJold), alphas[J]))
        iter += 1
        j_blocked.clear() #Reset Block list
        i_blocked.clear()
        print("iteration number: %d" % iter)
    return b, alphas

def draw(alpha, bet, data, label):
    plt.xlabel(u"x1")
    plt.xlim(0, 100)
    plt.ylabel(u"x2")
    for i in range(len(label)):
        if label[i] > 0:
            plt.plot(data[i][0], data[i][1], 'or')
        else:
            plt.plot(data[i][0], data[i][1], 'og')
    w1 = 0.0
    w2 = 0.0
    for i in range(len(label)):
        w1 += alpha[i] * label[i] * data[i][0]
        w2 += alpha[i] * label[i] * data[i][1]
    w = float(- w1 / w2)

    b = float(- bet / w2)
    r = float(1 / w2)
    lp_x1 = list([10, 90])
    lp_x2 = []
    lp_x2up = []
    lp_x2down = []
    for x1 in lp_x1:
        lp_x2.append(w * x1 + b)
        lp_x2up.append(w * x1 + b + r)
        lp_x2down.append(w * x1 + b - r)
    lp_x2 = list(lp_x2)
    lp_x2up = list(lp_x2up)
    lp_x2down = list(lp_x2down)
    plt.plot(lp_x1, lp_x2, 'b')
    plt.plot(lp_x1, lp_x2up, 'b--')
    plt.plot(lp_x1, lp_x2down, 'b--')
    plt.show()

if __name__ == '__main__':
    #Read Training data
    filestr = "dataset.txt"
    dataArr, labelArr = loadDataSet(filestr)
    print(dataArr)
    print(labelArr)
    # Reshape the input data
    dataMatrix = mat(dataArr)
    labelMatrix = mat(labelArr).transpose()
    b, alphas = smoComputation(dataMatrix, labelMatrix, 0.05, 0.001, 500)
    print(b)
    print(alphas)
    draw(alphas, b, dataArr, labelArr)