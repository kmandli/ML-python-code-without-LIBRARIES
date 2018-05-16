
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA
from matplotlib.mlab import PCA

def d_PCA(x, corr_logic):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll

    # use mean_centered data or standardized mean_centered data
    if not corr_logic:
        dataForPca = xMeanCentered
    else:
        dataForPca = x

    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)

    # eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]

    # get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)

    # collect PCA results
    pcaResults = {'data': x,
                   'mean_centered_data': xMeanCentered,
                   'PC_variance': eigenValues,
                   'loadings': eigenVectors,
                   'scores': pcaScores}

    return pcaResults

def main():

    # ----------------------------------------------------------------
    # A first glimpse at PCA
    # ----------------------------------------------------------------

    # 1. generate the raw data
    x1 = np.arange(start=0, stop=20, step=0.1)
    x2 = 2 * x1 + np.random.normal(loc=0, scale=0.5, size=len(x1))

    # 2. visualize the raw data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x1, x2, color='blue')
    ax.set_aspect('equal', 'box')
    fig.show()

    # 3. do PCA
    dataForAnalysis = np.column_stack((x1, x2))

    useCorr = False
    myPCAResults = d_PCA(x=dataForAnalysis, corr_logic=useCorr)

    percentVarianceExplained = 100 * myPCAResults['PC_variance'][0] / sum(myPCAResults['PC_variance'])
    print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scree plot')
    ax.scatter(range(len(myPCAResults['PC_variance'])), myPCAResults['PC_variance'], color='blue')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scores plot')
    ax.scatter(myPCAResults['scores'][:,0], myPCAResults['scores'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('loadings plot')
    ax.scatter(myPCAResults['loadings'][:,0], myPCAResults['loadings'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('raw data and PC axis')
    ax.scatter(x1, x2, color='blue')
    ax.plot([0, -20*myPCAResults['loadings'][0,0]], [0, -20*myPCAResults['loadings'][1,0]],
            color='green', linewidth=3)
    ax.plot([0, 20 * myPCAResults['loadings'][0, 1]], [0, 20 * myPCAResults['loadings'][1, 1]],
            color='green',linewidth=3)
    ax.set_aspect('equal', 'box')
    fig.show()

    # keep only the first dimension
    dataReconstructed = np.matmul(myPCAResults['scores'][:, 0].reshape((200, 1)), myPCAResults['loadings'][:,0].reshape((1,2)))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('reconstructed data')
    ax.scatter(dataReconstructed[:, 0], dataReconstructed[:, 1], color='blue')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.show()

    # ----------------------------------------------------------------
    # PCA on completely random data
    # ----------------------------------------------------------------
    # 1. generate raw data
    x1 = np.random.normal(loc=0, scale=0.5, size=len(x1))
    x2 = np.random.normal(loc=0, scale=0.5, size=len(x1))

    # 2. visualize the raw data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x1, x2, color='blue')
    ax.set_aspect('equal', 'box')
    fig.show()

    # 3. do PCA
    dataForAnalysis = np.column_stack((x1, x2))

    useCorr = False
    myPCAResults = d_PCA(x=dataForAnalysis, corr_logic=False)

    percentVarianceExplained = 100 * myPCAResults['PC_variance'][0] / sum(myPCAResults['PC_variance'])
    print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scree plot')
    ax.scatter(range(len(myPCAResults['PC_variance'])), myPCAResults['PC_variance'], color='blue')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scores plot')
    ax.scatter(myPCAResults['scores'][:,0], myPCAResults['scores'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('loadings plot')
    ax.scatter(myPCAResults['loadings'][:,0], myPCAResults['loadings'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('raw data and PC axis')
    ax.scatter(x1, x2, color='blue')
    k=3
    ax.plot([0, (-1)*k*myPCAResults['loadings'][0,0]], [0, (-1)*k*myPCAResults['loadings'][1,0]],
            color='green', linewidth=3)
    ax.plot([0, k * myPCAResults['loadings'][0, 1]], [0, k * myPCAResults['loadings'][1, 1]],
            color='green',linewidth=3)
    ax.set_aspect('equal', 'box')
    fig.show()
    plt.close('all')

    # ----------------------------------------------------------------
    # PCA on toy data
    # ----------------------------------------------------------------
    # # 1. get the raw data
    # in_file_name = "dataset_1.csv"
    # dataIn = pd.read_csv(in_file_name)
    #
    # # 2. visualize the raw data
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("raw data")
    # ax.scatter(dataIn['x'], dataIn['y'], color='blue')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # fig.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("raw data")
    # ax.scatter(dataIn['x'], dataIn['z'], color='blue')
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')
    # fig.show()
    #
    #
    # # 3. do PCA
    # dataForAnalysis = dataIn.as_matrix()
    # useCorr = False
    # myPCAResults = d_PCA(x=dataForAnalysis, corr_logic=useCorr)
    #
    # percentVarianceExplained = 100 * myPCAResults['PC_variance'][0] / sum(myPCAResults['PC_variance'])
    # print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.set_title('scree plot')
    # ax.scatter(range(len(myPCAResults['PC_variance'])), myPCAResults['PC_variance'], color='blue')
    # fig.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.set_title('scores plot')
    # ax.scatter(myPCAResults['scores'][:,0], myPCAResults['scores'][:,1], color='blue')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # fig.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.set_title('loadings plot')
    # ax.scatter(myPCAResults['loadings'][:,0], myPCAResults['loadings'][:,1], color='blue')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # fig.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.set_title('raw data and PC axis')
    # ax.scatter(x1, x2, color='blue')
    # k=3
    # ax.plot([0, (-1)*k*myPCAResults['loadings'][0,0]], [0, (-1)*k*myPCAResults['loadings'][1,0]],
    #         color='green', linewidth=3)
    # ax.plot([0, k * myPCAResults['loadings'][0, 1]], [0, k * myPCAResults['loadings'][1, 1]],
    #         color='green',linewidth=3)
    # ax.set_aspect('equal', 'box')
    # fig.show()
    # plt.close('all')


if __name__ == '__main__':
    main()
