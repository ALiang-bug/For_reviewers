import torch
import torch.nn as nn
from model.TransformerBlock import TransformerBlock

class OnlineNetwork(nn.Module):
    def __init__(self, input_size=64, att_heads=8, attn_dropout=0.2, proj_hidden_dim=64, pred_dim=64):
        super().__init__()
        self.hidden_size = input_size
        self.att_heads = att_heads
        self.attn_dropout = attn_dropout
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_dim = pred_dim

        self.encoder = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_heads, attn_dropout = self.attn_dropout)

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.proj_hidden_dim),
            nn.LayerNorm(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
            nn.LayerNorm(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.pred_dim),
            nn.LayerNorm(self.pred_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.proj_hidden_dim, self.pred_dim),
            nn.LayerNorm(self.pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pred_dim, self.pred_dim)
        )

    def forward(self, x, mask):
        feat, att_score = self.encoder(x, x, x, mask)  # (48, 199, 64)
        proj = self.projector(feat)  # (9552, 64)
        pred = self.predictor(proj)

        return feat, proj, pred, att_score

class TargetNetwork(nn.Module):
    def __init__(self, input_size=64, att_heads=8, attn_dropout=0.2, proj_hidden_dim=64, pred_dim=64):
        super().__init__()
        self.hidden_size = input_size
        self.att_heads = att_heads
        self.attn_dropout = attn_dropout
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_dim = pred_dim

        self.encoder = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_heads, attn_dropout=self.attn_dropout)

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.proj_hidden_dim),
            nn.LayerNorm(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
            nn.LayerNorm(self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.pred_dim),
            nn.LayerNorm(self.pred_dim)
        )

    def forward(self, x, mask):
        feat, _ = self.encoder(x, x, x, mask)
        B, L, D = feat.shape  # B=48, L=199, D=64

        feat_trans = feat.transpose(1, 2).contiguous()  # (48, 64, 199)
        feat_flat = feat_trans.view(-1, D)  # (48*199=9552, 64)

        proj_flat = self.projector(feat_flat)  # (9552, 64)

        proj = proj_flat.view(B, L, -1)  # (48, 199, 64)

        return feat, proj

    @torch.no_grad()
    def update_from_online(self, online_net, tau=0.996):
        for target_param, online_param in zip(self.parameters(), online_net.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data