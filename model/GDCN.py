import numpy as np
import scipy.sparse as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # 仅导入Data（兼容所有PyG版本）

def pyg_to_scipy_csr(pyg_data, num_nodes):
    if not hasattr(pyg_data, 'edge_index') or pyg_data.edge_index is None:
        raise ValueError("Object edge_index doesn't exist or None！")
    edge_index = pyg_data.edge_index
    if edge_index.device.type != 'cpu':
        edge_index = edge_index.cpu()
    edge_index = edge_index.numpy()

    edge_weight = None
    if hasattr(pyg_data, 'edge_weight'):
        edge_weight = pyg_data.edge_weight
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    else:
        if edge_weight.device.type != 'cpu':
            edge_weight = edge_weight.cpu()
        edge_weight = edge_weight.numpy()

    csr_mat = ss.csr_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return csr_mat

def _convert_sp_mat_to_sp_tensor(sp_mat, device="cpu"):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(device)
    values = torch.from_numpy(coo.data).to(device)
    shape = torch.Size(coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class CascadeGDCN0(nn.Module):
    def __init__(self, num_hops=2, in_channels=64, user_size=1000,
                 adj_cascade_norm=None, out_degree=None, in_degree=None, device="cuda"):
        super().__init__()
        self.num_hops = num_hops
        self.embed_dim = in_channels
        self.user_size = user_size
        self.device = self._validate_device(device)

        self.hop_attention = nn.Parameter(torch.ones(num_hops, device=self.device))
        self.Theta = nn.Parameter(torch.randn(in_channels, in_channels, device=self.device))
        nn.init.xavier_uniform_(self.Theta)

        self.theta_out = nn.Parameter(
            torch.full((num_hops, in_channels, in_channels), 0.5, device=self.device)
        )
        for k in range(num_hops):
            nn.init.xavier_uniform_(self.theta_out[k])
        
        self.theta_in = nn.Parameter(
            torch.full((num_hops, in_channels, in_channels), 0.5, device=self.device)
        )
        for k in range(num_hops):
            nn.init.xavier_uniform_(self.theta_in[k])

        if isinstance(adj_cascade_norm, Data):
            adj_csr = pyg_to_scipy_csr(adj_cascade_norm, user_size)
        elif isinstance(adj_cascade_norm, ss.csr_matrix):
            adj_csr = adj_cascade_norm
        else:
            raise TypeError(
                f"adj_cascade_norm type：{type(adj_cascade_norm)}"
            )

        self.multi_hop_A = []
        A_k = adj_csr.copy()
        for k in range(self.num_hops):
            A_k_tensor = _convert_sp_mat_to_sp_tensor(A_k, self.device)
            self.multi_hop_A.append(A_k_tensor)
            if k < self.num_hops - 1:
                A_k = A_k @ adj_csr

        if out_degree is None or in_degree is None:
            raise ValueError("out_degree and in_degree vector error！")
        self.out_degree = torch.tensor(out_degree, dtype=torch.float32, device=self.device)
        self.out_degree = torch.clamp(self.out_degree, min=1e-8)
        self.in_degree = torch.tensor(in_degree, dtype=torch.float32, device=self.device)
        self.in_degree = torch.clamp(self.in_degree, min=1e-8)

    def _validate_device(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is unavailable, change CPU")
            return "cpu"
        return device if device in ["cpu", "cuda"] else "cpu"

    def _spmm(self, sparse_mat, dense_mat):
        if torch.__version__ >= "2.0.0":
            return torch.spmm(sparse_mat, dense_mat)
        else:
            return torch.mm(sparse_mat.to_dense(), dense_mat)

    def forward(self, H_l):
        if H_l.shape != (self.user_size, self.embed_dim):
            raise ValueError(
                f"input dim {H_l.shape} error，should ({self.user_size}, {self.embed_dim})"
            )
        H_l = H_l.to(self.device)

        alpha = F.softmax(self.hop_attention, dim=0)
        sum_term = torch.zeros_like(H_l, device=self.device)

        for k in range(self.num_hops):
            A_k = self.multi_hop_A[k]
            theta_k1 = self.theta_out[k]
            theta_k2 = self.theta_in[k]

            D_out_H = self.out_degree.unsqueeze(1) * H_l
            D_out_H_transformed = torch.matmul(D_out_H, theta_k1)
            term_out = self._spmm(A_k, D_out_H_transformed)

            A_k_T = A_k.transpose(0, 1)
            D_in_H = self.in_degree.unsqueeze(1) * H_l
            D_in_H_transformed = torch.matmul(D_in_H, theta_k2)
            term_in = self._spmm(A_k_T, D_in_H_transformed)

            sum_term += alpha[k] * (term_out + term_in)

        conv_term = torch.sigmoid(torch.matmul(sum_term, self.Theta))
        H_l_plus_1 = conv_term + H_l

        return H_l_plus_1

class CascadeGDCN(nn.Module):
    def __init__(self, num_hops=2, in_channels=64, user_size=1000,
                 adj_cascade_norm=None, out_degree=None, in_degree=None, device="cuda"):
        super().__init__()
        self.num_hops = num_hops
        self.embed_dim = in_channels
        self.user_size = user_size
        self.device = self._validate_device(device)

        self.hop_attention = nn.Parameter(torch.ones(num_hops, device=self.device))
        self.theta_out = nn.Parameter(torch.full((num_hops,), 0.5, device=self.device))
        self.theta_in = nn.Parameter(torch.full((num_hops,), 0.5, device=self.device))
        self.Theta = nn.Parameter(torch.randn(in_channels, in_channels, device=self.device))
        nn.init.xavier_uniform_(self.Theta)

        if isinstance(adj_cascade_norm, Data):
            adj_csr = pyg_to_scipy_csr(adj_cascade_norm, user_size)
        elif isinstance(adj_cascade_norm, ss.csr_matrix):
            adj_csr = adj_cascade_norm
        else:
            raise TypeError(
                f"adj_cascade_norm type：{type(adj_cascade_norm)}"
            )

        self.multi_hop_A = []
        A_k = adj_csr.copy()
        for k in range(self.num_hops):
            A_k_tensor = _convert_sp_mat_to_sp_tensor(A_k, self.device)
            self.multi_hop_A.append(A_k_tensor)
            if k < self.num_hops - 1:
                A_k = A_k @ adj_csr

        if out_degree is None or in_degree is None:
            raise ValueError("out_degree and in_degree vector error！")
        self.out_degree = torch.tensor(out_degree, dtype=torch.float32, device=self.device)
        self.out_degree = torch.clamp(self.out_degree, min=1e-8)
        self.in_degree = torch.tensor(in_degree, dtype=torch.float32, device=self.device)
        self.in_degree = torch.clamp(self.in_degree, min=1e-8)

    def _validate_device(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is unavailable，change CPU")
            return "cpu"
        return device if device in ["cpu", "cuda"] else "cpu"

    def _spmm(self, sparse_mat, dense_mat):
        if torch.__version__ >= "2.0.0":
            return torch.spmm(sparse_mat, dense_mat)
        else:
            return torch.mm(sparse_mat.to_dense(), dense_mat)

    def forward(self, H_l):
        if H_l.shape != (self.user_size, self.embed_dim):
            raise ValueError(
                f"imput dim {H_l.shape} error，should ({self.user_size}, {self.embed_dim})"
            )
        H_l = H_l.to(self.device)

        alpha = F.softmax(self.hop_attention, dim=0)
        sum_term = torch.zeros_like(H_l, device=self.device)

        for k in range(self.num_hops):
            A_k = self.multi_hop_A[k]
            theta_k1 = self.theta_out[k]
            theta_k2 = self.theta_in[k]

            D_out_H = self.out_degree.unsqueeze(1) * H_l
            term_out = theta_k1 * self._spmm(A_k, D_out_H)

            A_k_T = A_k.transpose(0, 1)
            D_in_H = self.in_degree.unsqueeze(1) * H_l
            term_in = theta_k2 * self._spmm(A_k_T, D_in_H)

            sum_term += alpha[k] * (term_out + term_in)

        conv_term = torch.sigmoid(torch.matmul(sum_term, self.Theta))
        H_l_plus_1 = conv_term + H_l

        return H_l_plus_1