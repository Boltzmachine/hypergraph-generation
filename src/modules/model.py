import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from tango.integrations.torch import Model
from torch_geometric.nn import HypergraphConv, HeteroConv, GATConv
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from .layer import MeshConv, BipartiteDenseGATConv2, BipartiteDenseGATConv
from ..utils import get_offset, create_mask_if_not_exist

from typing import Union, Callable
import math


@Model.register("sin")
class SinusoidalPositionalEncoding(Model):
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


@Model.register("mlp")
class MLPPositionalEncoding(Model):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, t):
        return self.layers(t.float().unsqueeze(-1))

@Model.register("dummy")
class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 10)

    def forward(self, *args):
        return args


@Model.register("flat")
class FlatModel(Model):
    def __init__(self,
                 hidden_dim: int,
                 position_encoder: Model,
            ) -> None:
        super().__init__()
        self.position_encoder = position_encoder
        self.node_embedder = nn.Linear(3, 3 * hidden_dim)
        self.edge_embedder = nn.Linear(1, hidden_dim)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim * (24 + 48 + 10), 100 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 100, 100 * hidden_dim),
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim * 100, 100 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 100, 8 * 3),
        )
        
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * (24 + 48 + 10), 100 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 100, 100 * hidden_dim),
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim * 100, 100 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 100, 6 * 8),
        )

    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_nodes, n_hyper]
        """
        bs = X.size(0)
        # X = quantizer.dequantize(X)
        # [bs, n_nodes, 3, hidden_dim]
        X = self.node_embedder(X)
        # [bs, n_hyper, n_nodes, hidden_dim]
        H = self.edge_embedder(H.unsqueeze(-1).float())
        
        X = X.view(bs, -1)
        H = H.view(bs, -1)
        t = self.position_encoder(t)
        f = torch.cat([X, H, t], dim=-1)
        X = self.node_head(f).view(bs, 8, 3)
        H = self.edge_head(f).view(bs, 6, 8)
        return X, H
    
    
class PyGWrapper(Model):
    def __init__(self, conv) -> None:
        super().__init__()
        self.conv = conv
        
    def forward(self, x_src, x_tgt, adj, src_mask, tgt_mask):
        bs, n_tgt_nodes, hidden_dim = x_tgt.size()
        _, n_src_nodes, hidden_dim = x_src.size()
        
        def adj_to_edge_index(adj):
            edge_index = adj.nonzero()
            src_offset = get_offset(edge_index[:, 0], src_mask)
            tgt_offset = get_offset(edge_index[:, 0], tgt_mask)
            edge_index = edge_index[:, 1:]
            edge_index[:, 0] += src_offset
            edge_index[:, 1] += tgt_offset
            edge_index = edge_index.T
            # edge_index = torch.sort(edge_index, 1)[0]
            return edge_index
                
        edge_index = adj_to_edge_index(adj)
        x_tgt = x_tgt.reshape(-1, hidden_dim)
        x_src = x_src.reshape(-1, hidden_dim)
        # src_mask = create_mask_if_not_exist(src_mask, x_src)
        # tgt_mask = create_mask_if_not_exist(tgt_mask, x_tgt)
        # # adj [bs, n_src, n_tgt]
        # adj1 = adj * src_mask[:, :, None] * tgt_mask[:, None, :]
        # adj2 = (1 - adj) * src_mask[:, :, None] * tgt_mask[:, None, :]
        # data_list = [
        #     BipartiteData(edge_index1=edge_index1.nonzero().T, edge_index2=edge_index2.nonzero().T, x_s=x_s[mask_s.bool()], x_t=x_t[mask_t.bool()])
        #     for x_s, x_t, mask_s, mask_t, edge_index1, edge_index2 in zip(x_src, x_tgt, src_mask, tgt_mask, adj1, adj2)
        #     ]
        # batch = Batch.from_data_list(data_list, follow_batch=["x_s", "x_t"])
        x_tgt = self.conv((x_src, x_tgt), edge_index)
        x_tgt = x_tgt.view(bs, n_tgt_nodes, hidden_dim)
        return x_tgt
    

class EdgeGNNConv(Model):
    def __init__(
        self,
        hidden_dim: int,
        n_channels: int = 2,
        ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        aggr = ['mean', 'sum', 'std', 'max', 'min']
        # aggr = 'mean'
        # self.conv = pyg_nn.GENConv(hidden_dim, hidden_dim)#, aggr=aggr)
        self.conv = MeshConv(hidden_dim, hidden_dim, aggr=aggr)
        self.norm = pyg_nn.norm.LayerNorm(hidden_dim)
        self.edge_transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_src, x_tgt, adj, src_mask, tgt_mask):
        """
        x_tgt - [bs, n_tgt_nodes, hidden_dim]
        x_src - [bs, n_src_nodes, hidden_dim]
        """
        bs, n_tgt_nodes, hidden_dim = x_tgt.size()
        _, n_src_nodes, hidden_dim = x_src.size()
        
        src_mask = create_mask_if_not_exist(src_mask, x_src)
        tgt_mask = create_mask_if_not_exist(tgt_mask, x_tgt)
        
        adj_mask = src_mask[:, :, None] * tgt_mask[:, None, :]
        edge_attr = adj.masked_select(adj_mask.bool())
        edge_attr = self.edge_transform(edge_attr.unsqueeze(1).float())
        
        adj = torch.ones_like(adj) * adj_mask
        edge_index = adj.nonzero()
        src_offset = get_offset(edge_index[:, 0], src_mask)
        tgt_offset = get_offset(edge_index[:, 0], tgt_mask)
        edge_index = edge_index[:, 1:]
        edge_index[:, 0] += src_offset
        edge_index[:, 1] += tgt_offset
        edge_index = edge_index.T
        x_src = x_src[src_mask.bool()]
        x_tgt = x_tgt[tgt_mask.bool()]
        
        batch = torch.arange(bs, device=x_tgt.device).repeat_interleave(tgt_mask.sum(1))
        
        x_tgt = self.norm(self.conv((x_src, x_tgt), edge_index, edge_attr), batch) + x_tgt

        x_tgt, mask = to_dense_batch(x_tgt, batch)
        assert (mask == tgt_mask).all()

        return x_tgt


class MultiChannelGNNConv(Model):
    def __init__(
        self,
        hidden_dim: int,
        n_channels: int = 2,
        ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        aggr = ['mean', 'sum', 'std', 'max', 'min']
        # aggr = 'mean'
        self.conv1 = PyGWrapper(pyg_nn.SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        self.conv2 = PyGWrapper(pyg_nn.SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        # self.conv1 = pyg_nn.GENConv(hidden_dim, hidden_dim)
        # self.conv2 = pyg_nn.GENConv(hidden_dim, hidden_dim)
        # self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        # self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        
        self.downscale = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(), 
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

    def forward(self, x_src, x_tgt, adj, src_mask, tgt_mask):
        """
        x_tgt - [bs, n_tgt_nodes, hidden_dim]
        x_src - [bs, n_src_nodes, hidden_dim]
        """
        src_mask = create_mask_if_not_exist(src_mask, x_src)
        tgt_mask = create_mask_if_not_exist(tgt_mask, x_tgt)
        
        adj1 = adj * src_mask[:, :, None] * tgt_mask[:, None, :]
        adj2 = (1-adj) * src_mask[:, :, None] * tgt_mask[:, None, :]
        
        x_tgt_1 = self.conv1(x_src, x_tgt, adj1, src_mask, tgt_mask)
        x_tgt_2 = self.conv2(x_src, x_tgt, adj2, src_mask, tgt_mask)

        x_tgt = self.downscale(torch.cat([x_tgt_1, x_tgt_2], dim=-1))
        return x_tgt
                

@Model.register("hyper")
class HyperModel(Model):
    def __init__(
            self,
            hidden_dim: int,
            position_encoder: Model,
            backbone: str,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder

        self.node_embedder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.face_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        node_types = ["vertex", "face"]
        edge_types = [("vertex", "in", "face"), ("face", "has", "vertex")]
        meta_data = (node_types, edge_types)
        
        backbone_class = {
            "pyg": EdgeGNNConv,
            "dense": BipartiteDenseGATConv2
        }[backbone]
        self.convs = nn.ModuleList([
            nn.ModuleList([
                backbone_class(hidden_dim),
                backbone_class(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                ),
            ])
            for _ in range(2)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(1 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, 3)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        # X_ = F.one_hot(X, 256).float() * 2e10 - 1e10
        # X_ = X
        # H_ = H.float() * 2e10 - 1e10
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        # [bs, n_nodes, 3]
        # X = quantizer.dequantize(X)
        # [bs, n_nodes, hidden_dim]
        X = self.node_embedder(X)

        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = (X + pos_emb[:, None, :])# * mask[..., None]

        face_emb = self.face_embedder(H.sum(-1, keepdim=True).float())
        face_emb = face_emb + pos_emb[:, None, :]
        
        for conv in self.convs:
            # [bs, n_hyper, hidden_dim]
            face_emb = conv[0](vertex_emb, face_emb, H.transpose(1, 2), src_mask=mask, tgt_mask=None)# + face_emb
            face_emb = conv[2](face_emb) + face_emb
            # [bs, n_node, hidden_dim]
            vertex_emb = conv[1](face_emb, vertex_emb, H, src_mask=None, tgt_mask=mask)# + vertex_emb
            vertex_emb = conv[3](vertex_emb) + vertex_emb
            
        # [bs, n_hyper, n_nodes, 2 * hidden_dim]
        edge_emb = torch.cat([
            face_emb[:, :, None, :].expand(-1, -1, n_nodes, -1),
            vertex_emb[:, None, :, :].expand(-1, n_hyper, -1, -1)
            ], dim=-1)

        # vertex_emb = torch.cat([vertex_emb, X], dim=-1)
        # [bs, n_nodes, 3 * hidden_dim]
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)
        # [bs, n_hyper, n_nodes, hidden_dim]
        # edge_emb = torch.cat([edge_emb, H[..., None].expand(-1, -1, -1, self.hidden_dim).to(edge_emb.dtype)], dim=-1)
        H = self.edge_head(edge_emb).view(bs, n_hyper, n_nodes)
    
        return X, H


@Model.register("debug")
class DebugModel(Model):
    def __init__(
            self,
            hidden_dim: int,
            position_encoder: Model,
            backbone: str,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder

        self.node_embedder = nn.Sequential(
            nn.Linear(3, hidden_dim),
        )
        self.face_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
        )
        
        node_types = ["vertex", "face"]
        edge_types = [("vertex", "in", "face"), ("face", "has", "vertex")]
        meta_data = (node_types, edge_types)
        
        backbone_class = {
            "pyg": MultiChannelGNNConv,
            "dense": BipartiteDenseGATConv2
        }[backbone]
        self.convs = nn.ModuleList([
            nn.ModuleList([
                backbone_class(hidden_dim),
                backbone_class(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                ),
            ])
            for _ in range(1)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(1 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, 3)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        # X_ = F.one_hot(X, 256).float() * 2e10 - 1e10
        # X_ = X
        # H_ = H.float() * 2e10 - 1e10
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        # [bs, n_nodes, 3]
        # X = quantizer.dequantize(X)
        # [bs, n_nodes, hidden_dim]
        X = self.node_embedder(X)

        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = X

        face_emb = self.face_embedder(H.sum(-1, keepdim=True).float())
        face_emb = face_emb

        for conv in self.convs:
            # [bs, n_hyper, hidden_dim]
            face_emb = conv[0](vertex_emb, face_emb, H.transpose(1, 2), src_mask=mask, tgt_mask=None) + face_emb
            face_emb = conv[2](face_emb) + face_emb
            # [bs, n_node, hidden_dim]
            vertex_emb = conv[1](face_emb, vertex_emb, H, src_mask=None, tgt_mask=mask) + vertex_emb
            vertex_emb = conv[3](vertex_emb) + vertex_emb
            
        # [bs, n_hyper, n_nodes, 2 * hidden_dim]
        edge_emb = torch.cat([
            face_emb[:, :, None, :].expand(-1, -1, n_nodes, -1),
            vertex_emb[:, None, :, :].expand(-1, n_hyper, -1, -1)
            ], dim=-1)

        # vertex_emb = torch.cat([vertex_emb, X], dim=-1)
        # [bs, n_nodes, 3 * hidden_dim]
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)
        # [bs, n_hyper, n_nodes, hidden_dim]
        # edge_emb = torch.cat([edge_emb, H[..., None].expand(-1, -1, -1, self.hidden_dim).to(edge_emb.dtype)], dim=-1)
        H = self.edge_head(edge_emb).view(bs, n_hyper, n_nodes)
    
        return X, H


@Model.register("hyper_initial")
class HyperInitialModel(Model):
    def __init__(
            self,
            hidden_dim: int,
            position_encoder: Model,
            backbone: str,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder

        self.node_embedder = nn.Sequential(
            nn.Linear(3 * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.face_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        node_types = ["vertex", "face"]
        edge_types = [("vertex", "in", "face"), ("face", "has", "vertex")]
        meta_data = (node_types, edge_types)
        
        backbone_class = {
            "pyg": MultiChannelGNNConv,
            "dense": BipartiteDenseGATConv2
        }[backbone]
        self.convs = nn.ModuleList([
            nn.ModuleList([backbone_class(hidden_dim), backbone_class(hidden_dim)])
            for _ in range(2)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(1 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, 3)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, H, t, mask, XT):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        # X_ = F.one_hot(X, 256).float() * 2e10 - 1e10
        
        # H_ = H.float() * 2e10 - 1e10
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        # [bs, n_nodes, 3]
        # X = quantizer.dequantize(X)
        # [bs, n_nodes, hidden_dim]
        assert XT is not None
        X = self.node_embedder(torch.cat([X, XT], dim=-1))

        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = (X + pos_emb[:, None, :])# * mask[..., None]

        face_emb = self.face_embedder(H.sum(-1, keepdim=True).float())
        face_emb = face_emb + pos_emb[:, None, :]
        
        for conv in self.convs:
            # [bs, n_hyper, hidden_dim]
            face_emb = conv[0](vertex_emb, face_emb, H.transpose(1, 2), mask)
            # [bs, n_node, hidden_dim]
            vertex_emb = conv[1](face_emb, vertex_emb, H)
            
        # [bs, n_hyper, n_nodes, 2 * hidden_dim]
        edge_emb = torch.cat([
            face_emb[:, :, None, :].expand(-1, -1, n_nodes, -1),
            vertex_emb[:, None, :, :].expand(-1, n_hyper, -1, -1)
            ], dim=-1)

        # vertex_emb = torch.cat([vertex_emb, X], dim=-1)
        # [bs, n_nodes, 3 * hidden_dim]
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)
        # [bs, n_hyper, n_nodes]
        # edge_emb = torch.cat([edge_emb, H[..., None].expand(-1, -1, -1, self.hidden_dim).to(edge_emb.dtype)], dim=-1)
        H = self.edge_head(edge_emb).squeeze(-1)
    
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
            # nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.edge_head = nn.Sequential(
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            # nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, batch_first=True, dropout=0),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, X, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        """
        H_ = H.float() * 2e10 - 1e10
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        H = self.edge_embedder(H)

        X = self.node_embedder(X)
        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        vertex_emb = X + pos_emb[:, None, :].expand(-1, n_nodes, -1)
        edge_emb = H + pos_emb[:, None, None, : ].expand(-1, n_hyper, n_nodes, -1)
        
        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)
        H = self.edge_head(edge_emb.view(bs, n_hyper * n_nodes, -1)).squeeze().view(bs, n_hyper, n_nodes)
        
        return X, H_

from copy import deepcopy

@Model.register("multi-stage")
class MultiStageModel(Model):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model1 = model
        self.model2 = deepcopy(model)
        self.model3 = deepcopy(model)

    def forward(self, X, H, t, mask):
        X1, H1 = self.model1(X, H, t, mask)
        X2, H2 = self.model2(X, H, t, mask)
        X3, H3 = self.model3(X, H, t, mask)

        X = self.mask_add(t, X1, X2, X3)
        H = self.mask_add(t, H1, H2, H3)
        return X, H

    def mask_add(self, t, out1, out2, out3):
        out = (t <= 25)[:, None, None] * out1 + ((25 < t) & (t <= 125))[:, None, None] * out2 + (125 < t)[:, None, None] * out3
        return out
