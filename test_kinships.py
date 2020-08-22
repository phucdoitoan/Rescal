"""
Test the rescal with wasserstein on the Kinships dataset
"""

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import precision_recall_curve, auc
from scipy import sparse
from rescal_wass import als_wass
from time import time


def predict_rescal_als_wass(T1, T2, alpha=5, rank=100, conv=1e-4):
    """
    Predict the values of elements in T1, T2 from the factorizations
    Args:
        T1:
        T2:
        alpha:
        rank:

    Returns:
        Prediction P1, P2 for T1, T2
    """
    t0 = time()
    A1, A2, R1, R2 = als_wass(T1, T2, rank=rank, alpha=alpha, conv=conv, lmbdaA1=5, lmbdaA2=5, lmbdaR1=5, lmbdaR2=5)
    #print('\t\t In predict_rescal_als_wass: %.2f s' %(time() - t0))

    n1 = A1.shape[0]
    P1 = np.zeros((n1, n1, len(R1)))
    for k in range(len(R1)):
        P1[:, :, k] = A1 @ R1[k] @ A1.T

    n2 = A2.shape[0]
    P2 = np.zeros((n2, n2, len(R2)))
    for k in range(len(R2)):
        P2[:, :, k] = A2 @ R2[k] @ A2.T

    return P1, P2

def normalize_predictions_wass(P1, P2, e1, e2, k1, k2):
    """
    Normalize the elements in P1, P2 along the third axis (the relation axis)
    Args:
        P1: prediction of the first tensor
        P2:
        e1: num of entities in P1
        e2:
        k1: num of relations in P1
        k2:

    Returns:
        the normalized P1, P2 along the third axis
    """
    for a in range(e1):
        for b in range(e1):
            nrm = np.linalg.norm(P1[a, b, :k1])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P1[a, b, :k1] = np.round_(P1[a, b, :k1] / nrm, decimals=3)

    for a in range(e2):
        for b in range(e2):
            nrm = np.linalg.norm(P2[a, b, :k2])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P2[a, b, :k2] = np.round_(P2[a, b, :k2] / nrm, decimals=3)

    return P1, P2

def innerfold(GROUNDTRUTH1, GROUNDTRUTH2, T1, T2, mask_idx1, mask_idx2, target_idx1, target_idx2, e1, e2, k1, k2, sz1, sz2, alpha=5, rank=100):
    """
    Inner fold in cross-validation
    Args:
        T1: e1 x e1 x k1 tensor
        T2:
        mask_idx1: mask the elements of T1 in mask_idx1, i.e set them to be 0s
        mask_idx2:
        target_idx1: target elements of T1 to predict the values
        target_idx2:
        e1: number of entities in T1
        e2:
        k1: number of relations in T1
        k2:
        sz1:
        sz2:
        alpha: regularization term for the Wasserstein Loss
        rank: rank of the factorization

    Returns:
        AUC1, AUC2 for the predictions in T1 and T2
    """
    Tc1 = [Ti.copy() for Ti in T1]
    mask_idx1 = np.unravel_index(mask_idx1, (e1, e1, k1))
    target_idx1 = np.unravel_index(target_idx1, (e1, e1, k1))

    Tc2 = [Ti.copy() for Ti in T2]
    mask_idx2 = np.unravel_index(mask_idx2, (e2, e2, k2))
    target_idx2 = np.unravel_index(target_idx2, (e2, e2, k2))

    # set values to be predicted to zero
    for i in range(len(mask_idx1[0])):
        Tc1[mask_idx1[2][i]][mask_idx1[0][i], mask_idx1[1][i]] = 0

    for i in range(len(mask_idx2[0])):
        Tc2[mask_idx2[2][i]][mask_idx2[0][i], mask_idx2[1][i]] = 0

    # predict unknown values
    P1, P2 = predict_rescal_als_wass(Tc1, Tc2, alpha=alpha, rank=rank)
    P1, P2 = normalize_predictions_wass(P1, P2, e1, e2, k1, k2)

    # compute AUC
    prec1, recall1, _ = precision_recall_curve(GROUNDTRUTH1[target_idx1], P1[target_idx1])
    prec2, recall2, _ = precision_recall_curve(GROUNDTRUTH2[target_idx2], P2[target_idx2])

    return auc(recall1, prec1), auc(recall2, prec2)

def main(K1, K2, alpha, rank):

    e1, k1 = K1.shape[0], K1.shape[2]
    e2, k2 = K2.shape[0], K2.shape[2]

    SZ1 = e1 * e1 * k1
    SZ2 = e2 * e2 * k2

    # save the ground truth to prepare
    GROUNDTRUTH1 = K1.copy()
    GROUNDTRUTH2 = K2.copy()

    T1 = [sparse.lil_matrix(K1[:,:,i]) for i in range(k1)]
    T2 = [sparse.lil_matrix(K2[:,:,i]) for i in range(k2)]

    FOLDS = 10
    IDX1 = list(range(SZ1))
    IDX2 = list(range(SZ2))

    np.random.shuffle(IDX1)
    np.random.shuffle(IDX2)

    fsz1 = int(SZ1/FOLDS)
    fsz2 = int(SZ2/FOLDS)

    offset1, offset2 = 0, 0

    AUC_test1 = np.zeros(FOLDS)
    AUC_test2 = np.zeros(FOLDS)

    for f in range(FOLDS):
        idx_test1 = IDX1[offset1: offset1 + fsz1]
        idx_test2 = IDX2[offset2: offset2 + fsz2]

        AUC_test1[f], AUC_test2[f] = innerfold(GROUNDTRUTH1, GROUNDTRUTH2, T1, T2, idx_test1, idx_test2, idx_test1, idx_test2, e1, e2, k1, k2, SZ1, SZ2, alpha=alpha, rank=rank)

        offset1 += fsz1
        offset2 += fsz2

    return AUC_test1, AUC_test2


if __name__ == '__main__':

    mat = loadmat('data/uml.mat')
    #mat = loadmat('data/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    print('K: ', K.shape)

    # fill nan values with 0s if exist
    #K = np.nan_to_num(K)

    n, d = K.shape[0], K.shape[2]
    n1 = int(n/2)
    d1 = int(d/2)

    K1 = K[:n1, :n1, :d1]   # note: when lambda_A, lambda_R = 5, 5 => 52 x 52 x 26 tensors -> 0.95 AUCs
    K2 = K[n1:, n1:, d1:]
    print('K1: ', K1.shape)
    print('K2: ', K2.shape)




    """
    # load the data
    mat = loadmat('data/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)

    # shuffle the order of entities
    #idx = np.arange(K.shape[0])
    #np.random.shuffle(idx)
    #K = K[idx] # shuffle the rows
    #np.random.shuffle(idx)
    #K = K[:,idx,:] # shuffle the columns


    K1 = K[:52, :52, :]
    K2 = K[52:, 52:, :]
    print('K1: ', K1.shape)
    print('K2: ', K2.shape)
    """

    rank = 50
    with open('unoverlapped_UMLS_output%s.txt' %rank, 'w') as file:
        file.write('K1: %s\n' %(K1.shape,))
        file.write('K2: %s\n' % (K2.shape,))

        for alpha in [0, 5, 10, 11, 12, 13, 14, 15]: #[15, 16, 17, 18, 19, 20]: # [0, 5, 10, 15, 16, 17, 20]:

            print('\t ***** Alpha = %s, rank = %s *****' %(alpha, rank))
            file.write('\t ***** Alpha = %s, Rank = %s *****\n' %(alpha, rank))

            t0 = time()
            AUC1, AUC2 = main(K1, K2, alpha=alpha, rank=rank)
            print('\t      run in %.2f s' %(time() - t0))
            file.write('\t      run in %.2f s\n' %(time() - t0))

            auc1, std1 = AUC1.mean(), AUC1.std()
            auc2, std2 = AUC2.mean(), AUC2.std()

            print('AUC-PR Test 1: Mean : %f / Std : %f' % (auc1, std1))
            print('AUC-PR Test 2: Mean : %f / Std : %f' % (auc2, std2))
            file.write('AUC-PR Test 1: Mean : %f / Std : %f\n' % (auc1, std1))
            file.write('AUC-PR Test 2: Mean : %f / Std : %f\n' % (auc2, std2))




























