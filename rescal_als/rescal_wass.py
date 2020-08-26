"""
This is based on the codes published by rescal author
"""

import numpy as np
from scipy import sparse
import torch
from sinkhorn import SinkhornDistance
from time import time


def als_wass(X1, X2, rank, alpha=5, conv=1e-4, maxIter=500, lmbdaA1=10, lmbdaA2=10, lmbdaR1=10, lmbdaR2=10, dtype=np.float):
    """
    Compute the factorization A1, A2, R1, R2 of tensor X1 and X2
    Args:
        X1: n1 x n1 x m1 tensor
        X2: n2 x n2 x m2 tensor
        rank: latent representation rank
        alpha: regularization hyperparameter for Wasserstein Loss
        conv: thresh value to stop

    Returns:
        A1: latent representation of entities in tensor X1, size n1 x rank
        A2: latent representation of entities in tensor X2, size n2 x rank
        R1: latent component interaction matrix of tensor X1, size rank x rank
        R2: latent component interaction matrix of tensor X2, size rank x rank
    """
    fit = fitchange = fitold = f  =0

    n1, n2 = X1[0].shape[0], X2[0].shape[0]
    k1, k2 = len(X1), len(X2)

    # convert X1, X2 to csr
    for i in range(k1):
        if sparse.issparse(X1[i]):
            X1[i] = X1[i].tocsr()
            X1[i].sort_indices()           # ===============> WHY DO WE NEED sort_indices()

    for i in range(k2):
        if sparse.issparse(X2[i]):
            X2[i] = X2[i].tocsr()
            X2[i].sort_indices()

    # initialize A1, A2, X1, X2 and P
    S1 = sparse.csr_matrix((n1, n1), dtype=dtype)
    for i in range(k1):
        S1 = S1 + X1[i]
        S1 = S1 + X1[i].T

    _, A1 = sparse.linalg.eigsh(sparse.csr_matrix(S1, dtype=dtype, shape=(n1,n1)), rank)
    A1 = np.array(A1, dtype=dtype)

    S2 = sparse.csr_matrix((n2, n2), dtype=dtype)
    for i in range(k2):
        S2 = S2 + X2[i]
        S2 = S2 + X2[i].T

    _, A2 = sparse.linalg.eigsh(sparse.csr_matrix(S2, dtype=dtype, shape=(n2,n2)), rank)
    A2 = np.array(A2, dtype=dtype)

    P = np.ones((n1,n2), dtype=dtype) / (n1 * n2)

    # initialize R1, R2
    R1 = updateR(X1, A1, lmbdaR1)
    R2 = updateR(X2, A2, lmbdaR2)

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=50, device='cpu')
    # factorization
    for itr in range(maxIter):
        fitold = fit

        # update A1, A2, P
        A1_old = A1
        A1 = updateA_rescal(X1, A1, R1, lmbdaA1) + alpha * updateA_wass(A1, A2, P)
        A2 = updateA_rescal(X2, A2, R2, lmbdaA2) + alpha * updateA_wass(A2, A1_old, P.T)

        P = updateP(A1, A2, sinkhorn) # update P with sinkhorn

        # update R1, R2
        R1 = updateR(X1, A1, lmbdaR1)
        R2 = updateR(X2, A2, lmbdaR2)

        fit = compute_fit(X1, X2, A1, A2, R1, R2, lmbdaA1, lmbdaA2, lmbdaR1, lmbdaR2)

        fitchange = abs(fitold - fit)

        if itr > 0 and fitchange < conv:
            print('\t\t\t\t in als_wass: itr = %d' %(itr))
            break

    return A1, A2, R1, R2

def updateR(X, A, lmbdaR):
    """
    Update R step as described in RESCAL
    Args:
        X: tensor data n x n x m
        A: latent representation n x rank
        lmbdaR: regularization term for R

    Returns:
        updated R
    """
    rank = A.shape[1]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Shat = np.kron(S, S)
    Shat = (Shat / (Shat ** 2 + lmbdaR)).reshape(rank, rank)
    R = []
    for i in range(len(X)):
        Rn = Shat * (U.T @ X[i] @ U)
        Rn = Vt.T @ Rn @ Vt

        R.append(Rn)

    return R

def updateA_rescal(X, A, R, lmbdaA):
    """
    Update A step as described in RESCAL
    Args:
        X: list of m frontal slices X_k of size n x n
        A: latent representation n x rank
        R: list of m latent-component interaction matrix rank x rank
        lmbdaA: regularization term of A

    Returns:
        update part for A duo to the loss in original RESCAL
    """
    n, rank = A.shape
    F = np.zeros((n, rank), dtype=A.dtype)
    E = np.zeros((rank, rank), dtype=A.dtype)

    AtA = A.T @ A

    for i in range(len(X)):
        F += X[i] @ A @ R[i].T + X[i].T @ A @ R[i]
        E += R[i] @ AtA @ R[i].T + R[i].T @ AtA @ R[i]

    # regularization
    I = lmbdaA * np.eye(rank, dtype=A.dtype)

    # compute upate for A
    A = F @ np.linalg.inv(I + E)

    return A

def updateA_wass(A, otherA, P):
    """
    Update A step: this part is due to Wasserstein Loss term
    Args:
        A: latent representation of one tensor, n1 x rank
        otherA: latent representation of other tensor, n2 x rank
        P: optimal transport plan between the entities in the 1st tensor to the 2nd tensor,
           transport cost defined as the Euclidiean distance between the rows

    Returns:
        update part for A due to the Wassertein Loss
    """
    rank = A.shape[1]
    n = P.shape[1]
    P1 = (P @ np.ones(n)).reshape(-1, 1)
    P1_extend = np.tile(P1, reps=(1, rank))

    return 2 * (A * P1_extend - P @ otherA)

# <================= need to implement in a same platform, all in torch or all in numpy
def updateP(A1, A2, sinkhorn=SinkhornDistance(eps=0.1, max_iter=50, device='cpu')):
    """
    Update optimal transport P between the entities of the two tensors
    Args:
        A1: latent representation of the entities in the 1st tensor
        A2: latent representation of the entities in the 2nd tensor

    Returns:
        updated optimal transport plan P
    """
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    mu1 = torch.ones(n1) / n1
    mu2 = torch.ones(n2) / n2

    _, P, _ = sinkhorn(torch.from_numpy(A1), torch.from_numpy(A2), mu1, mu2)

    return P.numpy()

def compute_fit_rescal(X, A, R, lmbdaA, lmbdaR):
    """
    Compute the fit score of factorization A, R with X, the same as in RESCAL codes
    Args:
        X:
        A:
        R:
        lmbdaA:
        lmbdaR:

    Returns:
        fit score for the factorization
    """
    f = 0

    # compute norm of X
    normX = [sum(M.data**2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = A @ R[i] @ A.T
        f += np.linalg.norm(X[i] - ARAt) ** 2

    return 1 - f / sumNorm

def compute_fit(X1, X2, A1, A2, R1, R2, lmbdaA1, lmbdaA2, lmbdaR1, lmbdaR2):
    """
    Compute the average fit score for the factorization of tensor X1 and X2
    Args:
        X1:
        X2:
        A1:
        A2:
        R1:
        R2:
        lmbdaA1:
        lmbdaA2:
        lmbdaR1:
        lmbdaR2:

    Returns:
        average fit score of the two factorization
    """
    f1 = compute_fit_rescal(X1, A1, R1, lmbdaA1, lmbdaR1)
    f2 = compute_fit_rescal(X2, A2, R2, lmbdaA2, lmbdaR2)

    return (f1 + f2) / 2


















































