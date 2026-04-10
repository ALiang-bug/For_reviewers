import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from model.HGAT import HypergraphConv
from model.GDCN import CascadeGDCN
import torch.nn.init as init
import Constants
from model.BYOL_net import OnlineNetwork, TargetNetwork
from torch.autograd import Variable

from Optim import ScheduledOptim
from utils.util import *


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)

    return masked_seq.cuda()

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred1, proj2, pred2, proj1):
        pred1 = nn.functional.normalize(pred1, dim=-1)
        proj2 = nn.functional.normalize(proj2, dim=-1)
        pred2 = nn.functional.normalize(pred2, dim=-1)
        proj1 = nn.functional.normalize(proj1, dim=-1)

        loss1 = nn.functional.mse_loss(pred1, proj2, reduction='sum')
        loss2 = nn.functional.mse_loss(pred2, proj1, reduction='sum')
        return (loss1 + loss2) / 2


class SNEP(nn.Module):
    def __init__(self, opt, social_graph, hypergraphs, temporal_similarity, dropout=0.3, reverse=False, attn=False):
        super(SNEP, self).__init__()
        self.n_node = opt.n_node
        self.initial_feature = opt.d_model
        self.hidden_size = opt.d_model
        self.pos_dim = opt.pos_dim
        self.drop_r  = opt.dropout
        self.layers = opt.graph_layer
        self.diffcov_layers = opt.diffcov_layer
        self.att_head = opt.att_head
        self.rela_hop = opt.rela_hop
        self.mask_pa = opt.mask_pa
        self.mask_pr = opt.mask_pr

        self.reverse= reverse
        self.attn=attn
        self.n_channel = len(hypergraphs) + 1  

        self.dropout = nn.Dropout(dropout)
        self.social_centrality = social_graph[1]
        self.out_degree = social_graph[2]
        self.in_degree = social_graph[3]
        
        self.temporal_similarity = temporal_similarity
        self.adj_cascade_norm = hypergraphs[2]

        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.hidden_size)) for _ in range(self.n_channel)])
        self.att = nn.Parameter(torch.zeros(1, self.hidden_size))
        self.att_m = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.HGAT_layers = nn.ModuleList()
        self.GDCN_layers = nn.ModuleList()

        for i in range(self.diffcov_layers):
            self.GDCN_layers.append(CascadeGDCN(self.rela_hop, self.hidden_size, self.n_node, self.adj_cascade_norm, self.out_degree, self.in_degree))
        for i in range(self.layers):
            self.HGAT_layers.append(HypergraphConv(in_channels = self.hidden_size, out_channels = self.hidden_size, heads = self.att_head))
        self.online_ATT = OnlineNetwork(input_size=self.hidden_size, att_heads=self.att_head, attn_dropout = self.drop_r)
        self.target_ATT = TargetNetwork(input_size=self.hidden_size, att_heads=self.att_head, attn_dropout=self.drop_r)
        self.reset_parameters()
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps= 1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.hidden_size, opt.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

        self.criterion = BYOLLoss()
        for param in self.target_ATT.parameters():
            param.requires_grad = False

        self.ema_tau = opt.ema_tau if hasattr(opt, 'ema_tau') else 0.99

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))
    
    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score
    
    def _dropout_graph(self, graph, keep_prob):
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index.t()
        
        random_index = torch.rand(edge_attr.shape[0]) + keep_prob
        random_index = random_index.int().bool()

        edge_index = edge_index[random_index]
        edge_attr = edge_attr[random_index]
        return Data(edge_index=edge_index.t(), edge_attr=edge_attr)

    def pgrank_mask_optimize(self, centrality, raw_user_emb, p_a=0.5, p_r=0.6):
        N = raw_user_emb.shape[0]
        phi = trans_to_cuda(torch.tensor([centrality[idx] for idx in range(N)], dtype=torch.float32))  # [N]

        X = raw_user_emb
        d = X.shape[1]
        a_i = []
        for i in range(d):
            feat_sum = torch.sum(torch.abs(X[:, i]) * phi)
            a_i.append(torch.log(feat_sum + 1e-8))
        a_i = trans_to_cuda(torch.tensor(a_i, dtype=torch.float32))   # [d]

        a_max = torch.max(a_i)
        u_a = torch.mean(a_i)
        denominator = (a_max - u_a) + 1e-8

        p_i = ((a_max - a_i) / denominator) * p_a  # [d]
        p_r_tensor = trans_to_cuda(torch.tensor(p_r, dtype=torch.float32))
        p_i = torch.min(p_i, p_r_tensor)
        p_i = torch.nan_to_num(p_i, nan=0.0, posinf=1.0, neginf=0.0)
        p_i = torch.clamp(p_i, 0.0, 1.0)

        M = trans_to_cuda(torch.bernoulli(p_i.unsqueeze(1).repeat(1, N)))
        X_T = X.T
        masked_X_T = (1 - M) * X_T
        optimized_user_em = masked_X_T.T

        return optimized_user_em

    def history_cas_learning(self):

        if self.training:
            temporal_similarity = self._dropout_graph(self.temporal_similarity, keep_prob=1-self.drop_r)
        else:
            temporal_similarity = self.temporal_similarity

        temporal_similarity = trans_to_cuda(temporal_similarity)

        u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c1 = self.pgrank_mask_optimize(self.social_centrality, u_emb_c1, p_a=self.mask_pa, p_r=self.mask_pr)
        u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)
        all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]


        for k in range(self.diffcov_layers):
            u_emb_c1 = self.GDCN_layers[k](u_emb_c1)
            normalize_c1 = F.normalize(u_emb_c1, p=2, dim=-1)
            all_emb_c1 += [normalize_c1]

        for k in range(self.layers):
            u_emb_c2 = self.HGAT_layers[k](u_emb_c2, temporal_similarity.edge_index)
            normalize_c2 = F.normalize(u_emb_c2, p=2, dim=-1)
            all_emb_c2 += [normalize_c2]

        u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        u_emb_c1 = torch.mean(u_emb_c1, dim=1)
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.mean(u_emb_c2, dim=1)
        high_embs, contrib_score = self.channel_attention(u_emb_c1, u_emb_c2)
        return high_embs, u_emb_c1, u_emb_c2, contrib_score


    def forward(self, input_original, label):

        input = input_original

        mask = (input == Constants.PAD)
        mask_label = (label == Constants.PAD)   
        HG_user, _, _, _ = self.history_cas_learning()

        online_emb = F.embedding(input, HG_user)
        target_emb = F.embedding(label, HG_user)
    
        online_att_out, _, online_pred, _ = self.online_ATT(online_emb, mask=mask.cuda())
        couple_out, _, couple_pred, _ = self.online_ATT(target_emb, mask=mask.cuda())
        with torch.no_grad():
            _, target_proj = self.target_ATT(target_emb, mask=mask_label.cuda())
            _, couple_proj = self.target_ATT(online_emb, mask=mask_label.cuda())
    
        online_output = torch.matmul(online_att_out, torch.transpose(HG_user, 1, 0))
    
        mask = get_previous_user_mask(input.cuda(), self.n_node)
        output_online = (online_output + mask).view(-1, online_output.size(-1))
    
        return output_online, online_pred, target_proj, couple_pred, couple_proj

    def model_prediction(self, input_original, _):

        input = input_original

        mask = (input == Constants.PAD)
        HG_user, u_emb_c1, u_emb_c2, contrib_score = self.history_cas_learning()

        online_emb = F.embedding(input, HG_user)

        online_att_out, _, online_pred, att_score = self.online_ATT(online_emb, mask=mask.cuda())
    
        online_output = torch.matmul(online_att_out, torch.transpose(HG_user, 1, 0))
    
        mask = get_previous_user_mask(input.cuda(), self.n_node)
        output_online = (online_output + mask).view(-1, online_output.size(-1))
    
        return output_online, att_score, HG_user, u_emb_c1, u_emb_c2, contrib_score

    @torch.no_grad()
    def update_target_ema(self):
        for online_param, target_param in zip(self.online_ATT.parameters(), self.target_ATT.parameters()):
            target_param.data = self.ema_tau * target_param.data + (1 - self.ema_tau) * online_param.data

    
   

