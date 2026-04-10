import numpy as np
import torch
import pickle
import os
from torch_geometric.data import Data
import scipy.sparse as ss
from dataLoader import Options, Read_all_cascade
import networkx as nx

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([col, row])
    data = torch.FloatTensor(coo.data)
    return Data(edge_index=index, edge_attr=data)

def ConRelationGraph(data):
    options = Options(data)
    _u2idx = {}
    _idx2u = {}

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    _idx2u = {idx: uid for uid, idx in _u2idx.items()}

    edges_list = []
    if not os.path.exists(options.net_data):
        return [], {}, None

    with open(options.net_data, 'r') as handle:
        relation_list = handle.read().strip().split("\n")
        relation_list = [edge.split(',') for edge in relation_list]
        relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                         edge[0] in _u2idx and edge[1] in _u2idx]
        relation_list_reverse = [edge[::-1] for edge in relation_list]
        edges_list += relation_list_reverse

    row, col, entries = [], [], []
    for pair in edges_list:
        row += [pair[0]]
        col += [pair[1]]
        entries += [1.0]
    social_mat = ss.csr_matrix((entries, (row, col)), shape=(len(_u2idx), len(_u2idx)), dtype=np.float32)

    social_matrix = social_mat.dot(social_mat)
    social_matrix = social_matrix.multiply(social_mat) + ss.eye(len(_u2idx), dtype=np.float32)
    social_matrix = social_matrix.tocoo()
    social_matrix = _convert_sp_mat_to_sp_tensor(social_matrix)

    social_out_degree = social_mat.sum(axis=1).A.flatten()
    social_in_degree = social_mat.sum(axis=0).A.flatten()

    G = nx.from_scipy_sparse_array(social_mat, create_using=nx.DiGraph())
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=1000, tol=1e-6)

    return social_matrix, pagerank, social_out_degree, social_in_degree


def compute_temporal_similarity(cas_times):

    start_time = cas_times[0]
    total_cas_times = cas_times[-1] - cas_times[0]
    if total_cas_times != 0:
        delta_ts = np.array([(t - start_time)/(total_cas_times) for t in cas_times])
    else:
        delta_ts = np.ones(len(cas_times))

    exp_deltas = np.exp(delta_ts)
    sum_exp_deltas = np.sum(exp_deltas)
    if sum_exp_deltas == 0:
        g = np.ones_like(delta_ts) / len(delta_ts)
    else:
        g = exp_deltas / sum_exp_deltas  # g_o^n

    exp_neg_g = np.exp(-g)
    sum_exp_neg_g = np.sum(exp_neg_g)
    if sum_exp_neg_g == 0:
        s = np.ones_like(g) / len(g)
    else:
        s = exp_neg_g / sum_exp_neg_g  # s_o^n

    return s

def ConHypergraph(data_name, user_size, window):

    user_size, all_cascade, all_time = Read_all_cascade(data_name)
    num_cascades = len(all_cascade)

    user_cont = {}
    for i in range(user_size):
        user_cont[i] = []

    win = window
    for i in range(len(all_cascade)):
        cas = all_cascade[i]

        if len(cas) < win:
            for idx in cas:
                user_cont[idx] = list(set(user_cont[idx] + cas))
            continue
        for j in range(len(cas)-win+1):
            if (j+win) > len(cas):
                break
            cas_win = cas[j:j+win]
            for idx in cas_win:
                user_cont[idx] = list(set(user_cont[idx] + cas_win))

    indptr, indices, data = [], [], []
    indptr.append(0)
    idx = 0

    for j in user_cont.keys():
        if len(user_cont[j])==0:
            idx =  idx + 1
            continue
        source = np.unique(user_cont[j])

        length = len(source)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(source[i])
            data.append(1)

    H_U = ss.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys())-idx, user_size))
    HG_User = H_U.tocoo()

    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])

        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))
    HG_Item = H_T.tocoo()

    win_size = window
    edge_weight = {}
    for cas in all_cascade:
        cas_unique = []
        [cas_unique.append(u) for u in cas if u not in cas_unique]
        for pos in range(1, len(cas_unique)):
            cur_user = cas_unique[pos]
            start_idx = max(0, pos - win_size)
            prev_users = cas_unique[start_idx:pos]
            for prev_user in prev_users:
                if cur_user == prev_user:
                    continue
                edge_key = (cur_user, prev_user)
                edge_weight[edge_key] = edge_weight.get(edge_key, 0) + 1

    adj_indptr = [0]
    adj_indices = []
    adj_data = []
    for cur_user in range(user_size):
        cur_edges = [(prev, w) for (cur, prev), w in edge_weight.items() if cur == cur_user]
        cur_edges = sorted(cur_edges, key=lambda x: x[0])
        for prev_user, weight in cur_edges:
            adj_indices.append(prev_user)
            adj_data.append(weight)
        adj_indptr.append(len(adj_indices))

    adj_cascade = ss.csr_matrix(
        (adj_data, adj_indices, adj_indptr),
        shape=(user_size, user_size),
        dtype=np.float32
    )
    degree = np.array(adj_cascade.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    degree_inv_sqrt = np.power(degree, -0.5)
    D_inv_sqrt = ss.diags(degree_inv_sqrt)
    adj_cascade_norm = D_inv_sqrt @ adj_cascade @ D_inv_sqrt

    adj_cascade_tensor = _convert_sp_mat_to_sp_tensor(adj_cascade_norm)

    cas_sim = []
    for j in range(num_cascades):
        cas = all_cascade[j]
        cas_time = all_time[j]

        if len(cas) != len(cas_time):
            raise ValueError(f"级联{j}的用户数({len(cas)})与时间数({len(cas_time)})不匹配！")

        cas_unique = []
        cas_time_unique = []
        seen_users = set()
        for u, t in zip(cas, cas_time):
            if u not in seen_users:
                seen_users.add(u)
                cas_unique.append(u)
                cas_time_unique.append(t)

        if len(cas_unique) == 0:
            cas_sim.append(([], []))
        elif len(cas_unique) == 1:
            cas_sim.append((cas_unique, [1.0]))
        else:
            cas_sim_list = compute_temporal_similarity(cas_time_unique)
            cas_sim.append((cas_unique, cas_sim_list))

    data_temporal = []
    for j in range(num_cascades):
        cas_unique, cas_sim_list = cas_sim[j]
        items = np.unique(cas_unique)

        sim_dict = dict(zip(cas_unique, cas_sim_list))
        for u in items:
            data_temporal.append(sim_dict[u])

        assert len(items) == (indptr[j + 1] - indptr[j]), f"级联{j}的长度不匹配！"

    temporal_similarity = ss.csr_matrix(
        (data_temporal, indices, indptr),
        shape=(num_cascades, user_size),
        dtype=np.float32
    )
    temporal_similarity_coo = temporal_similarity.tocoo()

    temporal_similarity_tensor = _convert_sp_mat_to_sp_tensor(temporal_similarity_coo)
    HG_Item = _convert_sp_mat_to_sp_tensor(HG_Item)
    HG_User = _convert_sp_mat_to_sp_tensor(HG_User)


    return HG_Item, HG_User, adj_cascade_tensor, temporal_similarity_tensor


