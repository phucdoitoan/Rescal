

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy import sparse
from rescal_wass import als_wass
from time import time

def predict_cross(T1, T2, alpha=5, rank=20, conv=1e-4):
    """predict values in the cross region
    Note that: we are assumming T1 and T2 have the same number of relations
    """

    t0 = time()
    A1, A2, R1, R2 = als_wass(T1, T2, rank=rank, alpha=alpha, conv=conv, lmbdaA1=5, lmbdaA2=5, lmbdaR1=5, lmbdaR2=5)
    print('als_wass in %.2f s' %(time() - t0))

    n1 = A1.shape[0]
    n2 = A2.shape[0]

    K1 = len(R1)
    K2 = len(R2)

    print('n1, n2: ', n1, n2)
    print('K1, K2: ', K1, K2)


    cross_P1 = np.zeros((n1, n1, K2))
    cross_P2 = np.zeros((n2, n2, K1))

    for k in range(K2):
        cross_P1[:, :, k] = A1 @ R2[k] @ A1.T

    for k in range(K1):
        cross_P2[:, :, k] = A2 @ R1[k] @ A2.T
    """

    cross_P1 = np.zeros((n1, n1, K1))
    cross_P2 = np.zeros((n2, n2, K2))

    for k in range(K1):
        cross_P1[:, :, k] = A1 @ R1[k] @ A1.T

    for k in range(K2):
        cross_P2[:, :, k] = A2 @ R2[k] @ A2.T
    """

    return cross_P1, cross_P2

def normalize_prediction_cross(cross_P1, cross_P2):
    """normalize the element in P1, P2"""

    n1, K1 = cross_P1.shape[0], cross_P1.shape[2]
    n2, K2 = cross_P2.shape[0], cross_P2.shape[2]

    for a in range(n1):
        for b in range(n1):
            nrm1 = np.linalg.norm(cross_P1[a, b, :])
            if nrm1 != 0:
                cross_P1[a, b, :] = np.round_(cross_P1[a, b, :] / nrm1, decimals=3)

    for a in range(n2):
        for b in range(n2):
            nrm2 = np.linalg.norm(cross_P2[a, b, :])
            if nrm2 != 0:
                cross_P2[a, b, :] = np.round_(cross_P2[a, b, :] / nrm2, decimals=3)

    return cross_P1, cross_P2



def main(K1, K2, cross_truth1, cross_truth2, alpha, rank):

    cross_P1, cross_P2 = predict_cross(K1, K2, alpha, rank)
    cross_P1, cross_P2 = normalize_prediction_cross(cross_P1, cross_P2)

    print('cross_P1: ', cross_P1.shape)
    print('cross_P1: ', cross_P2.shape)

    prec1, recall1, _ = precision_recall_curve(cross_truth1.reshape(-1), cross_P1.reshape(-1))
    prec2, recall2, _ = precision_recall_curve(cross_truth2.reshape(-1), cross_P2.reshape(-1))

    return auc(recall1, prec1), auc(recall2, prec2)

if __name__ == '__main__':

    mat = loadmat('data/alyawarradata.mat')
    #mat = loadmat('data/uml.mat')
    K = np.array(mat['Rs'], np.float32)

    n, d = K.shape[0], K.shape[2]
    n1 = int(n/2)
    d1 = int(d/2)

    K1 = K[:n1, :n1, :d1]
    K2 = K[n1:, n1:, d1:]
    cross_truth1 = K[:n1, :n1, d1:]
    cross_truth2 = K[n1:, n1:, :d1]
    #cross_truth1 = K[:n1, :n1, :d1]
    #cross_truth2 = K[n1:, n1:, d1:]

    print('K1: ', K1.shape)
    print('K2: ', K2.shape)
    print('cross_truth1: ', cross_truth1.shape)
    print('cross_truth2: ', cross_truth2.shape)

    k1 = K1.shape[2]
    k2 = K2.shape[2]
    T1 = [sparse.lil_matrix(K1[:,:,i]) for i in range(k1)]
    T2 = [sparse.lil_matrix(K2[:,:,i]) for i in range(k2)]

    rank = 30
    for alpha in [0, 5, 10, 15, 16, 17, 18, 19, 20]:
        print('\tAlpha: ', alpha)
        auc1, auc2 = main(T1, T2, cross_truth1, cross_truth2, alpha=alpha, rank=rank)

        print('Cross PR-AUC1: %.4f' %(auc1))
        print('Cross PR-AUC2: %.4f' %(auc2))

