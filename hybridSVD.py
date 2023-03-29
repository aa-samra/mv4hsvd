import numpy as np
import scipy
from scipy.linalg import cholesky


def hsvd(source_mtx, rank=None, left_similarity=None, right_similarity=None, alpha=0, beta=0, ret='both'):
    R = source_mtx
    if not left_similarity is None:
        assert left_similarity.shape[0] == left_similarity.shape[1], 'Similarity must be square'
        assert left_similarity.shape[0] == source_mtx.shape[0]
        left_similarity_weighted = (1-alpha) * np.eye(left_similarity.shape[0]) + alpha * left_similarity
        left_cholesky_factor = cholesky(left_similarity_weighted)
        R = left_cholesky_factor.T @ R
    if not right_similarity is None:
        assert right_similarity.shape[0] == right_similarity.shape[1], 'Similarity must be square'
        assert right_similarity.shape[0] == source_mtx.shape[1]
        right_similarity_weighted = (1-beta) * np.eye(right_similarity.shape[0]) + beta * right_similarity
        right_cholesky_factor = cholesky(right_similarity_weighted)
        R = R @ right_cholesky_factor

    U_ , S , V_ = np.linalg.svd(R)
    if not left_similarity is None:
        U = np.linalg.inv(left_cholesky_factor).T @ U_
    else:
        U = U_
    if not right_similarity is None:
        V = V_ @ scipy.linalg.inv(right_cholesky_factor).T
    else:
        V = V_
    
    if ret=='left':
        U = np.ascontiguousarray(U[:,:rank])
        return U, S
    if ret == 'right':
        V = np.ascontiguousarray(V.T[:,:rank])
        return S, V
    if ret == 'both':
        U = np.ascontiguousarray(U[:,:rank])
        V = np.ascontiguousarray(V.T[:,:rank])
        return U, S, V
    
