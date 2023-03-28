import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import svds
from tqdm import tqdm
from dataprep import transform_indices, reindex_data, generate_interactions_matrix
from evaluation import downvote_seen_items, topn_recommendations
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, LinearOperator
from scipy.linalg import cholesky
from polara import get_movielens_data
from polara.preprocessing.dataframes import leave_one_out, reindex
from dataprep import transform_indices
from evaluation import topn_recommendations, model_evaluate, downvote_seen_items


def hsvd(source_mtx, rank, left_similarity=None, right_similarity=None, alpha=0, beta=0, ret='both'):
    R = source_mtx
    if left_similarity:
        assert left_similarity.shape[0] == left_similarity.shape[1], 'Similarity must be square'
        assert left_similarity.shape[0] == source_mtx.shape[0]
        left_similarity_weighted = (1-alpha) * np.eye(left_similarity.shape[0]) + alpha * left_similarity
        left_cholesky_factor = cholesky(left_similarity_weighted)
        R = left_cholesky_factor.T @ R
    if right_similarity:
        assert right_similarity.shape[0] == right_similarity.shape[1], 'Similarity must be square'
        assert right_similarity.shape[0] == source_mtx.shape[1]
        right_similarity_weighted = (1-beta) * np.eye(right_similarity.shape[0]) + beta * right_similarity
        right_cholesky_factor = cholesky(right_similarity_weighted)
        R = R @ right_cholesky_factor

    U_ , S , V_ = svds(R, k=rank)
    if left_similarity:
        U = np.linalg.inv(left_cholesky_factor).T @ U_
    else:
        U = U_
    if right_similarity:
        V = V_ @ np.linalg.inv(right_cholesky_factor).T
    else:
        V = V_
    
    if ret=='left':
        U = np.ascontiguousarray(U[: ,::-1])
        return U, S
    if ret == 'right':
        V = np.ascontiguousarray(V[::-1, :].T)
        return S, V
    if ret == 'both':
        U = np.ascontiguousarray(U[: ,::-1])
        V = np.ascontiguousarray(V[::-1, :].T)
        return U, S, V

    


    
    
