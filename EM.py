import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import os


def normPDF(x, mu, sigma):  # 输入向量x(size维)，输出x在给定高斯分布的解
    # x n 维，mu n 维 sigma n*n维
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            #raise NameError("The covariance matrix can't be singular")
            return 1

        try:
            norm_const = 1.0 / \
                (np.math.pow((2*np.pi), float(size)/2) * np.math.pow(det, 1.0/2))
            x_mu = np.matrix(x - mu).T
            inv_ = np.linalg.inv(sigma)
            result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
            #print(norm_const * result)
            return norm_const * result
        except:
            return 0
    else:
        raise NameError("The dimensions of the input don't match")
        return -1


def initForwardBackward(X, K, d, N):
    # Initialize the state transition matrix A. A (- R:KxK matrix
    # K is the implicit state number
    # element A_{jk} = p(Z_n = k | Z_{n-1} = j)
    # Therefore, the matrix will be row-wise normalized. IOW, Sum(Row) = 1
    # State transition probability is time independent.
    A = np.ones((K, K))
    A = A/np.sum(A, 1)[None].T

    # Initialize the marginal probability for the first hidden variable
    # PI (-  R:Kx1
    PI = np.ones((K, 1))/K

    # Initialize Emission Probability. We assume Gaussian distribution
    # for emission. So we just need to keep the mean and covariance. These
    # parameters are different for different states.
    # Mu is dxK where kth column represents mu for kth state
    # SIGMA是一个形状为dxd的K个矩阵的列表
    # 每个元素表示对应状态的协方差矩阵。
    # 给定当前潜变量状态，emission probability与时间无关

    MU = np.random.rand(d, K)
    SIGMA = [np.eye(d) for i in range(K)]  # np.eye(d) 是d乘d 对脚为1的矩阵

    return A, PI, MU, SIGMA


def buildAlpha(X, PI, A, MU, SIGMA):
    # We build up Alpha here using dynamic programming. Alpha (- R:KxN matrix
    # where the element ALPHA_{ij} represents the forward probability
    # for jth timestep (j = 1...N) and ith state. The columns of ALPHA are
    # normalized for preventing underflow problem as discussed in secion
    # 13.2.4 in Bishop's PRML book. So,sum(column) = 1
    # c_t is the normalizing costant
    N = np.size(X, 1)  # 得到输入数据X的列数
    K = np.size(PI, 0)  # K是初始数据的行数
    Alpha = np.zeros((K, N))  # Alpha (- R:KxN matrix 初始化全是0
    c = np.zeros(N)  # c 初始化 R:Nx1 vector 初始化全是0

    # Base case: build the first column of ALPHA
    for i in range(K):
        try:
            Alpha[i, 0] = PI[i]*normPDF(X[:, 0], MU[:, i], SIGMA[i])
        except:
            Alpha[i, 0] = 0
    c[0] = np.sum(Alpha[:, 0])
    Alpha[:, 0] = Alpha[:, 0]/c[0]

    # Build up the subsequent columns
    for t in range(1, N):
        for i in range(K):
            for j in range(K):
                Alpha[i, t] += Alpha[j, t-1]*A[j, i]  # sum part of recursion
            # product with emission prob
            Alpha[i, t] *= normPDF(X[:, t], MU[:, i], SIGMA[i])
        c[t] = np.sum(Alpha[:, t])
        Alpha[:, t] = Alpha[:, t]/c[t]   # for scaling factors
    return Alpha, c


def buildBeta(X, c, PI, A, MU, SIGMA):
    # Beta is KxN matrix where Beta_{ij} represents the backward probability
    # for jth timestamp and ith state. Columns of Beta are normalized using
    # the element of vector c.

    N = np.size(X, 1)
    K = np.size(PI, 0)
    Beta = np.zeros((K, N))

    # Base case: build the last column of Beta
    for i in range(K):
        Beta[i, N-1] = 1.

    # Build up the matrix backwards
    for t in range(N-2, -1, -1):
        for i in range(K):
            for j in range(K):
                Beta[i, t] += Beta[j, t+1]*A[i, j] * \
                    normPDF(X[:, t+1], MU[:, j], SIGMA[j])
        Beta[:, t] /= c[t+1]
    return Beta


def Estep(trainSet, PI, A, MU, SIGMA):
    # The goal of E step is to evaluate Gamma(Z_{n}) and Xi(Z_{n-1},Z_{n})
    # First, create the forward and backward probability matrices
    Alpha, c = buildAlpha(trainSet, PI, A, MU, SIGMA)
    Beta = buildBeta(trainSet, c, PI, A, MU, SIGMA)

    # Dimension of Gamma is equal to Alpha and Beta where nth column represents
    # posterior density of nth latent variable. Each row represents a state
    # value of all the latent variables. IOW, (i,j)th element represents
    # p(Z_j = i | X,MU,SIGMA)
    Gamma = Alpha*Beta

    # Xi is a KxKx(N-1) matrix (N is the length of data seq)
    # Xi(:,:,t) = Xi(Z_{t-1},Z_{t})
    N = np.size(trainSet, 1)
    K = np.size(PI, 0)
    Xi = np.zeros((K, K, N))
    for t in range(1, N):
        Xi[:, :, t] = (1/c[t])*Alpha[:, t-1][None].T.dot(Beta[:, t][None])*A
        # Now columnwise multiply the emission prob
        for col in range(K):

            Xi[:, col, t] *= normPDF(trainSet[:, t], MU[:, col], SIGMA[col])

    return Gamma, Xi, c


def Mstep(X, Gamma, Xi):
    # Goal of M step is to calculate PI, A, MU, and SIGMA while training
    # Gamma and Xi as constant
    K = np.size(Gamma, 0)
    d = np.size(X, 0)

    PI = (Gamma[:, 0]/np.sum(Gamma[:, 0]))[None].T
    tempSum = np.sum(Xi[:, :, 1:], axis=2)
    A = tempSum/np.sum(tempSum, axis=1)[None].T

    MU = np.zeros((d, K))
    GamSUM = np.sum(Gamma, axis=1)[None].T
    SIGMA = []
    for k in range(K):
        MU[:, k] = np.sum(Gamma[k, :]*X, axis=1)/GamSUM[k]
        X_MU = X - MU[:, k][None].T
        SIGMA.append(X_MU.dot(((X_MU*(Gamma[k, :][None])).T))/GamSUM[k])
    return PI, A, MU, SIGMA


def iteration(allData, K = 10, iter_number = 20, PI=0, A=0, MU=0, SIGMA=0):

    (m, n) = np.shape(allData)

    # Separating out dev and train set
    devSet = allData[np.math.ceil(m*0.9):, 0:].T
    trainSet = allData[:np.math.floor(m*0.9), 0:].T

    # Setting up total number of clusters which will be fixed
    # Initialization: Build a state transition matrix with uniform probability
    if type(PI) == int:
        A, PI, MU, SIGMA = initForwardBackward(trainSet, K, n, m)

    # Temporary variables. X, Y mesh for plotting
    iter = 0
    prevll = -999999
    count = 0

    ll_dif_ls= []
    while(True):
        past_A = A.copy()
        #print("Count>>", count)
        count += 1
        iter = iter + 1
        # E-Step
        Gamma, Xi, c = Estep(trainSet, PI, A, MU, SIGMA)

        # M-Step
        PI, A, MU, SIGMA = Mstep(trainSet, Gamma, Xi)

        # Calculate log likelihood. We use the c vector for log likelihood because
        # it already gives us p(X_1^N)
        ll_train = np.sum(np.log(c))
        Gamma_dev, Xi_dev, c_dev = Estep(devSet, PI, A, MU, SIGMA)
        ll_dev = np.sum(np.log(c_dev))

        if(iter > iter_number or abs(ll_train - prevll) < 0.005):
            break
        print(abs(ll_train - prevll))
        ll_dif_ls.append(ll_train - prevll)
        prevll = ll_train

    print("length",len(ll_dif_ls),count)
    print("PI>>", PI.shape)
    print("A>>", A.shape)
    print("MU>>", MU.shape)
    print("SIGMA>>", len(SIGMA),SIGMA[0].shape)
    return [PI, A, MU, SIGMA]


def unifyData(data):
    for i in range(data.shape[0]):
        minVal = min(data[i])
        maxVal = max(data[i])
        domain = (maxVal-minVal)
        for j in range(data.shape[1]):
            absolute = (data[i][j]-minVal) / domain
            data[i][j] = absolute//0.1*0.1  # From 0.0 to 0.9
        # Deliberately split the state into 10 levels, which can be changed to 100 or 1000
    return data

def getData(scr):
    f = open(scr, 'r', encoding="utf-8")
    reader = csv.reader(f)

    Ls = [row for row in reader]
    Ls = np.array(Ls)
    # It's the raw data with the first row
    print(len(Ls))
    pure_data = Ls[1:, :].astype(np.float32)

    return pure_data

def combineMFCCmatrices(read_path):
    read_path = r"D:\cat_meow_clasification\jdr\train\small_B"

    files = os.listdir(read_path)
    combinedM = np.zeros((13, 0))
    for file_name in files:
        # 读取单个文件内容
        file_data = getData(read_path+"\\"+file_name)
        unified_data = unifyData(file_data)
        combinedM = np.concatenate((combinedM, unified_data), axis=1)
    combinedM = combinedM[:, 1:]
    return combinedM

# main()
data_B = combineMFCCmatrices(r"D:\cat_meow_clasification\jdr\train\small_B")
data_F = combineMFCCmatrices(r"D:\cat_meow_clasification\jdr\train\small_F")
data_I = combineMFCCmatrices(r"D:\cat_meow_clasification\jdr\train\small_I")
data_I = getData("I_ANI01_MC_FN_SIM01_101.csv")
data_I = unifyData(data_F)
testData = getData("Deal_B_BRA01_MC_MN_SIM01_102.csv")
testData = unifyData(testData)
print("getB")
B_coeff = iteration(data_B,4)
F_coeff = iteration(data_F,4)
I_coeff = iteration(data_I,4)
t_coeff= iteration(testData,10,20)

def classify(t_coeff,B_coeff,F_coeff,I_coeff):
    dif_B = cal_diff(t_coeff,B_coeff)
    dif_F = cal_diff(t_coeff,F_coeff)
    dif_I = cal_diff(t_coeff,I_coeff)
    small_idx = [dif_B,dif_F,dif_I].index(min([dif_B,dif_F,dif_I]))
    print(small_idx)
    return small_idx    

def cal_diff(t_coeff, cmp_coeff):
    t_A = t_coeff[1]
    cmp_A = cmp_coeff[1]
    A_diff = 0
    for i in range(t_A.shape[0]):
        for j in range(t_A.shape[1]):
            A_diff += abs(t_A[i][j]-cmp_A[i][j])
    print("A Diff = ",A_diff)
    return A_diff

def test_correctness(B_coeff,F_coeff,I_coeff):
    read_path = r"D:\cat_meow_clasification\jdr\test\Test_B"
    files = os.listdir(read_path)

    correct = 0
    sum= 0
    for file_name in files:
        # 读取单个文件内容
        sum+=1
        testData = getData(read_path+"\\"+file_name)
        testData = unifyData(testData)
        t_coeff= iteration(testData,4)
        if classify(t_coeff,B_coeff,F_coeff,I_coeff) == 0:
            correct += 1
    print("Correctness>>", correct/sum)

test_correctness(B_coeff,F_coeff,I_coeff)
