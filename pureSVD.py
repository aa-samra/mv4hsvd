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
from polara import get_movielens_data
from polara.preprocessing.dataframes import leave_one_out, reindex
from dataprep import transform_indices
from evaluation import topn_recommendations, model_evaluate, downvote_seen_items


def build_ssvd_model(config, data, data_description):
    source_matrix = generate_interactions_matrix(data, data_description, rebase_users = False)
    scaled_matrix, scaling_weights = rescale_matrix(source_matrix, config['scaling'])
    *_, vt = svds(scaled_matrix, k=config['rank'], return_singular_vectors='vh')
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors, scaling_weights

def rescale_matrix(matrix, scaling_factor):
    frequencies = matrix.getnnz(axis=0)
    scaling_weights = np.power(frequencies, 0.5*(scaling_factor-1))
    return matrix.dot(diags(scaling_weights)), scaling_weights


def ssvd_model_scoring(params, data, data_description):
    item_factors, scaling_weights = params
    test_matrix = generate_interactions_matrix(data, data_description, rebase_users = True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)
    return scores