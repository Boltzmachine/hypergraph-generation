import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from tango.integrations.torch import Model
from torch_geometric.nn import HypergraphConv, HeteroConv, GATConv

from .utils import quantizer

from typing import Union, Callable


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

        self.lin_root = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn = nn.Linear(3 * hidden_dim // heads, heads, bias=False)
        
    
    def forward(self, x_tgt, x_src, edge_attr):
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
        alpha = torch.softmax(alpha, dim=-1)
        
        # [bs, heads, n_tgt_nodes, n_features] = [bs, heads, n_tgt_nodes, n_src_nodes, 1] * [bs, 1, 1, n_src_nodes, n_features]
        src_msg = (alpha.unsqueeze(-1) * x_src[:, None, None, :, :]).sum(-2)
        # [bs, n_tgt_nodes, heads, n_features] 
        src_msg = src_msg.transpose(1, 2)
        src_msg = src_msg.reshape(src_msg.size(0), src_msg.size(1), -1)
        x_root = self.lin_root(x_root) + src_msg
        
        return x_root
    
                
@Model.register("hyper")
class HyperModel(Model):
    def __init__(
            self,
            hidden_dim: int = 128,
        ) -> None:
        super().__init__()
        self.node_embedder = nn.Linear(3, hidden_dim)
        self.edge_embedder = nn.Embedding(2, hidden_dim)
        
        self.convs = nn.ModuleList([
            nn.ModuleList([BipartiteDenseGATConv(hidden_dim),BipartiteDenseGATConv(hidden_dim)]),
            nn.ModuleList([BipartiteDenseGATConv(hidden_dim),BipartiteDenseGATConv(hidden_dim)]),
        ])

        self.node_head = nn.Linear(hidden_dim, 256 * 3)
        self.edge_head = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, X, H):
        X_org = F.one_hot(X, 256).float() * 2e8 - 1e8
        X = quantizer.dequantize(X)
        X = self.node_embedder(X)
        H = self.edge_embedder(H)
        
        # [bs, n_hyper, hidden_dim]
        face_emb = torch.ones(H.size(0), H.size(1), H.size(-1), device=H.device)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = X
        
        for conv in self.convs:
            face_emb = conv[0](face_emb, vertex_emb, H)
            face_emb = F.leaky_relu(face_emb)
            vertex_emb = conv[1](vertex_emb, face_emb, H.transpose(1, 2))
            vertex_emb = F.leaky_relu(vertex_emb)

        X = self.node_head(vertex_emb)
        H = face_emb @ self.edge_head.weight @ vertex_emb.transpose(-1, -2)
    
        return X_org, H
