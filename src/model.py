import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from tango.integrations.torch import Model
from torch_geometric.nn import HypergraphConv, HeteroConv, GATConv

from .utils import quantizer

from typing import Union, Callable
import math


@Model.register("position")
class PositionalEncoding(Model):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: Tensor) -> Tensor:
        return self.pe[t-1]


@Model.register("dummy")
class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 10)

    def forward(self, *args):
        return args


@Model.register("cuboid")
class CuboidModel(Model):
    def __init__(self, hidden_dim: int = 6):
        super().__init__()
        self.node_embedder = nn.Linear(3, hidden_dim)
        self.edge_embedder = nn.Embedding(2, hidden_dim)
        self.layers = nn.Sequential(
            *(
                [nn.TransformerEncoderLayer(16 * hidden_dim, 4, 128, batch_first=True) for _ in range(2)]
              + [nn.Linear(16 * hidden_dim, 8)]
              ),
        )

    def forward(self, X, H):
        X_org = X.clone().detach()
        X = quantizer.dequantize(X)
        X = self.node_embedder(X)
        # [bs, n_hyper, n_nodes, hidden_dim]
        H = self.edge_embedder(H)
        # [bs, n_hyper, n_nodes, hidden_dim]
        X = X.unsqueeze(1).expand(-1, H.size(1), -1, -1)
        # [bs, n_hyper, n_nodes, 2*hidden_dim]
        out = torch.cat([H, X], dim=-1)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.layers(out)
        return X_org, out


class BipartiteDenseGATConv(Model):
    def __init__(
            self,
            hidden_dim: int,
            heads: int = 8,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        assert hidden_dim % heads == 0
        
        self.lin_src = nn.Linear(hidden_dim, hidden_dim//heads)
        self.lin_tgt = nn.Linear(hidden_dim, hidden_dim//heads)
        self.lin_edge = nn.Linear(hidden_dim, hidden_dim//heads)
        
        self.attn = nn.Linear(3 * hidden_dim // heads, heads, bias=False)

        self.lin_root = nn.Linear(hidden_dim, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        # self.feed_forward = nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0)
        
    
    def forward(self, x_tgt, x_src, edge_attr, mask=None):
        """
        x_tgt: [bs, n_tgt_nodes, n_features]
        x_src: [bs, n_src_nodes, n_features]
        edge_attr: [bs, n_tgt_nodes, n_src_nodes, n_features]
        """
        x_root = x_tgt
        x_src = self.lin_src(x_src)
        x_tgt = self.lin_tgt(x_tgt)
        edge_attr = self.lin_edge(edge_attr)
        
        # attention coefficients [bs, n_tgt_nodes, n_src_nodes, heads]
        alpha = self.attn(torch.cat([x_src.unsqueeze(1).expand(-1, x_tgt.size(1), -1, -1), edge_attr, x_tgt.unsqueeze(2).expand(-1, -1, x_src.size(1), -1)], dim=-1))
        # attention coefficients [bs, heads, n_tgt_nodes, n_src_nodes]
        alpha = alpha.permute(0, 3, 1, 2)
        alpha = F.leaky_relu(alpha)
        if mask is not None:
            mask = ((mask - 1) * 1e10)[:, None, None, :]
            alpha = alpha + mask
            
        alpha = torch.softmax(alpha, dim=-1)
        
        # [bs, heads, n_tgt_nodes, n_features] = [bs, heads, n_tgt_nodes, n_src_nodes, 1] * [bs, 1, 1, n_src_nodes, n_features]
        src_msg = (alpha.unsqueeze(-1) * x_src[:, None, None, :, :]).sum(-2)
        # [bs, n_tgt_nodes, heads, n_features] 
        src_msg = src_msg.transpose(1, 2)
        src_msg = src_msg.reshape(src_msg.size(0), src_msg.size(1), -1)
        x_root = self.lin_root(x_root) + self.lin_msg(src_msg)
        x_root = self.feed_forward(x_root)
        
        return x_root
    
                
@Model.register("hyper")
class HyperModel(Model):
    def __init__(
            self,
            hidden_dim: int,
            position_encoder: Model,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder

        self.node_embedder = nn.Linear(3, hidden_dim)
        self.edge_embedder = nn.Embedding(2, hidden_dim)
        
        self.convs = nn.ModuleList([
            nn.ModuleList([BipartiteDenseGATConv(hidden_dim),BipartiteDenseGATConv(hidden_dim)])
            for _ in range(2)
        ])

        self.node_head = nn.Linear(hidden_dim, 256 * 3)
        self.edge_head = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        # X_ = F.one_hot(X, 256).float() * 2e10 - 1e10
        # H_ = H.float() * 2e10 - 1e10
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        H = self.edge_embedder(H)

        X = quantizer.dequantize(X).to(H.dtype)
        X = self.node_embedder(X)

        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = (X + pos_emb[:, None, :])# * mask[..., None]
        face_emb = pos_emb[:, None, :].expand(-1, n_hyper, -1)
        edge_emb = H# * mask[:, None, :, None]
        
        # [bs, n_hyper, hidden_dim]
        for conv in self.convs:
            face_emb = conv[0](face_emb, vertex_emb, edge_emb, mask) + face_emb
            vertex_emb = conv[1](vertex_emb, face_emb, edge_emb.transpose(1, 2)) + vertex_emb

        # [bs, n_nodes, 3 * hidden_dim]
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3, -1)
        H = face_emb @ self.edge_head.weight @ vertex_emb.transpose(-1, -2)
    
        return X, H
    

@Model.register("separate")
class SeparateModel(Model):
    def __init__(
            self,
            hidden_dim: int,
            position_encoder: Model,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder
        
        self.node_embedder = nn.Linear(3, hidden_dim)
        self.edge_embedder = nn.Embedding(2, hidden_dim)

        heads = 4
        self.node_head = nn.Sequential(
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.Linear(hidden_dim, 256 * 3)
        )
        self.edge_head = nn.Sequential(
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        H = self.edge_embedder(H)

        X = quantizer.dequantize(X).to(H.dtype)
        X = self.node_embedder(X)
        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = X + pos_emb[:, None, :].expand(-1, n_nodes, -1)
        edge_emb = H + pos_emb[:, None, None, : ].expand(-1, n_hyper, n_nodes, -1)
        
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3, -1)
        H = self.edge_head(edge_emb.view(bs, n_hyper * n_nodes, -1)).squeeze().view(bs, n_hyper, n_nodes)
        
        return X, H