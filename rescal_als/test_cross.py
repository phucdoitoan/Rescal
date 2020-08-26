

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
    A1, A2, R1, R2 = als_wass(T1, T2, rank=rank, alpha=alpha, conv=conv)
    print('als_wass in %.2f s' %(time() - t0))

    n1 = A1.shape[0]
    n2 = A2.shape[0]

    K = len(R1)
    assert K == len(R2), 'Both tensors need to have the same number of relations'

    cross_P1 = np.zeros((n1, n2, K))
    cross_P2 = np.zeros((n2, n1, K))

    for k in range(K):
        cross_P1[:, :, k] = A1 @ (R1[k] + R2[k]) @ A2.T / 2
        cross_P2[:, :, k] = A2 @ (R1[k] + R2[k]) @ A1.T / 2
        #cross_P1[:, :, k] = A1 @ R2[k] @ A2.T
        #cross_P2[:, :, k] = A2 @ R2[k] @ A1.T

    return cross_P1, cross_P2

def normalize_prediction_cross(cross_P1, cross_P2):
    """normalize the element in P1, P2"""

    n1, n2, K = cross_P1.shape
    assert (n2, n1, K) == cross_P2.shape, 'both predictions tensors need to have similar shape (n1, n2, K) and (n2, n1, K)'

    for a in range(n1):
        for b in range(n2):
            nrm1 = np.linalg.norm(cross_P1[a, b, :])
            if nrm1 != 0:
                cross_P1[a, b, :] = np.round_(cross_P1[a, b, :] / nrm1, decimals=3)

            nrm2 = np.linalg.norm(cross_P2[b, a, :])
            if nrm2 != 0:
                cross_P2[b, a, :] = np.round_(cross_P2[b, a, :] / nrm2, decimals=3)

    return cross_P1, cross_P2



def main(K1, K2, cross_truth1, cross_truth2, alpha, rank):

    cross_P1, cross_P2 = predict_cross(K1, K2, alpha, rank)
    cross_P1, cross_P2 = normalize_prediction_cross(cross_P1, cross_P2)


    """
    k = len(K1)
    n1, n2 = K1[0].shape
    SZ = n1*n2*k
    IDX = list(range(SZ))
    np.random.shuffle(IDX)
    idx_test1 = IDX[: int(SZ/10)]
    np.random.shuffle(IDX)
    idx_test2 = IDX[: int(SZ/10)]

    target_idx1 = np.unravel_index(idx_test1, (n1, n2, k))
    target_idx2 = np.unravel_index(idx_test2, (n2, n1, k))
    print('target_idx1: ', len(target_idx1))
    print('target_idx2: ', len(target_idx2))
    prec1, recall1, _ = precision_recall_curve(cross_truth1[target_idx1], cross_P1[target_idx1])
    prec2, recall2, _ = precision_recall_curve(cross_truth2[target_idx2], cross_P2[target_idx2])
    """

    prec1, recall1, _ = precision_recall_curve(cross_truth1.reshape(-1), cross_P1.reshape(-1))
    prec2, recall2, _ = precision_recall_curve(cross_truth2.reshape(-1), cross_P2.reshape(-1))

    return auc(recall1, prec1), auc(recall2, prec2)

if __name__ == '__main__':

    mat = loadmat('data/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)

    K1 = K[:52, :52, :]
    K2 = K[52:, 52:, :]
    cross_truth1 = K[:52, 52:, :]
    cross_truth2 = K[52:, :52, :]
    print('K1: ', K1.shape)
    print('K2: ', K2.shape)
    print('cross_truth1: ', cross_truth1.shape)
    print('cross_truth2: ', cross_truth2.shape)

    k = K.shape[2]
    T1 = [sparse.lil_matrix(K1[:,:,i]) for i in range(k)]
    T2 = [sparse.lil_matrix(K2[:,:,i]) for i in range(k)]

    rank = 20
    for alpha in [0, 5, 10, 15, 20]:
        print('\tAlpha: ', alpha)
        auc1, auc2 = main(T1, T2, cross_truth1, cross_truth2, alpha=alpha, rank=rank)

        print('Cross PR-AUC1: %.4f' %(auc1))
        print('Cross PR-AUC2: %.4f' %(auc2))

