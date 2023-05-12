import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from tango.integrations.torch import Model
from torch_geometric.nn import HypergraphConv, HeteroConv, GATConv
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, dense_to_sparse

from .layer import MeshConv, BipartiteDenseGATConv2, BipartiteDenseGATConv
from .transformer import TransformerEncoder
from ..utils import get_offset, create_mask_if_not_exist

from typing import Union, Callable
import math


@Model.register("sin")
class SinusoidalPositionalEncoding(Model):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
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
        self.conv = pyg_nn.SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        # self.conv = pyg_nn.TransformerConv(hidden_dim, hidden_dim//8, heads=8, edge_dim=2)
        # self.conv = MeshConv(hidden_dim, hidden_dim, aggr=aggr)

        self.norm1 = pyg_nn.norm.LayerNorm(hidden_dim)
        self.norm2 = pyg_nn.norm.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
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
        edge_attr = F.one_hot(edge_attr).float()
        
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
        
        x_tgt = self.conv((x_src, x_tgt), edge_index, edge_attr) + x_tgt
        x_tgt = self.norm1(x_tgt, batch)

        x_tgt = self.norm2(self.feed_forward(x_tgt) + x_tgt)

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
        
        self.pos_embedder = nn.Embedding(1000, hidden_dim)
        
        backbone_class = MultiChannelGNNConv
        
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

    def forward(self, X, E, H, t, mask):
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
        vertex_emb = vertex_emb + self.pos_embedder(100+torch.arange(vertex_emb.size(1), device=vertex_emb.device))

        face_emb = self.face_embedder((H * mask[:, None, :]).sum(-1, keepdim=True).to(vertex_emb.dtype))
        face_emb = face_emb + pos_emb[:, None, :]
        face_emb = face_emb + self.pos_embedder(torch.arange(face_emb.size(1), device=face_emb.device)) 
        
        for conv in self.convs:
            # [bs, n_hyper, hidden_dim]
            face_emb = conv[0](vertex_emb, face_emb, H.transpose(1, 2), src_mask=mask, tgt_mask=None)# + face_emb
            # face_emb = conv[2](face_emb) + face_emb
            # [bs, n_node, hidden_dim]
            vertex_emb = conv[1](face_emb, vertex_emb, H, src_mask=None, tgt_mask=mask)# + vertex_emb
            # vertex_emb = conv[3](vertex_emb) + vertex_emb
            
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
    
        return X, F.one_hot(E, num_classes=2).float() * 2e8 - 1e8, H



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
    

class PyGTransformerEncoderLayer(Model):
    def __init__(self, d_model, d_inner, n_head) -> None:
        super().__init__()
        self.conv = pyg_nn.TransformerConv(d_model, d_model//n_head, heads=n_head, edge_dim=6)

        self.norm1 = pyg_nn.norm.LayerNorm(d_model)
        self.norm2 = pyg_nn.norm.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, d_model)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv(x, edge_index, edge_attr) + x
        x = self.norm1(x, batch)
        x = self.feed_forward(x) + x
        x = self.norm2(x)
        return x
    

class PyGTransformerEncoder(Model):
    def __init__(self,         
            n_layers: int, 
            n_head: int, 
            d_k: int,
            d_v:int, 
            d_model: int, 
            d_inner: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            PyGTransformerEncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head)
         for _ in range(n_layers)])


    def forward(self, x, mask, relation):
        bs, n_nodes, _ = x.size()
        x = x.view(-1, x.size(-1))
        edge_index, edge_attr = dense_to_sparse(relation)
        batch = torch.arange(bs, device=x.device).repeat_interleave(n_nodes)
        edge_attr = F.one_hot(edge_attr).float()

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        
        x, mask = to_dense_batch(x, batch)

        return x
        


@Model.register("transformer")
class Transformer(Model):
    def __init__(
        self,
        hidden_dim: int,
        position_encoder: Model,
        n_layer: int = 3
        ) -> None:
        super().__init__()
        
        self.position_encoder = position_encoder
        
        # self.rank_embedder = nn.ModuleList([nn.Embedding(1000, hidden_dim) for _ in range(3)])
        self.L = 10
        self.node_embedder = nn.Sequential(
            # nn.Linear(3 * self.L * 2, hidden_dim),
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.face_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        hidden_dim = hidden_dim + self.position_encoder.d_model
        self.type_embedder = nn.Embedding(2, hidden_dim)
        self.face_pos_embedder = nn.Embedding(100, hidden_dim)
        self.vertex_pos_embedder = nn.Embedding(100, hidden_dim)
        
        n_head = 8
        self.relation_embedder = nn.Embedding(6, n_head)
        self.transformer = TransformerEncoder(n_layers=n_layer, n_head=n_head, d_k=hidden_dim//n_head, d_v=hidden_dim//n_head, d_model=hidden_dim, d_inner=2 * hidden_dim)
        
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
        
    def forward(self, X, E, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        mask - [bs, n_nodes]
        """
        bs, n_nodes, _ = X.size()
        _, n_hyper, _ = H.size()
        # [bs, n_nodes, 3]
        # X = quantizer.dequantize(X)
        # [bs, n_nodes, hidden_dim]
        pos_emb = self.position_encoder(t)
        
        # rank = torch.sort(X, 1)[1]
        # rank_emb = torch.stack([
        #     self.rank_embedder[i](rank[..., i]) for i in range(3)
        # ], dim=-1).sum(-1)
        
        # arange = torch.arange(self.L, device=X.device)
        # # pe - [bs, n_nodes, 3, 2 * L]
        # pe_sin = torch.sin(2**arange * torch.pi * X.unsqueeze(-1))
        # pe_cos = torch.cos(2**arange * torch.pi * X.unsqueeze(-1))
        # pe = torch.stack([pe_sin, pe_cos], dim=-1).view(bs, n_nodes, 3, -1).view(bs, n_nodes, -1)  

        vertex_emb = X
        vertex_emb = self.node_embedder(vertex_emb)# + rank_emb

        # [bs, n_nodes, hidden_dim]
        vertex_emb = torch.cat([vertex_emb, pos_emb[:, None, :].expand(-1, n_nodes, -1)], dim=-1) + self.vertex_pos_embedder(torch.arange(vertex_emb.size(1), device=vertex_emb.device))

        # H = H * mask[:, None, :]
        # face_feats = (X.unsqueeze(1) * H.unsqueeze(-1))
        # face_feats_max = face_feats.masked_fill(~H.unsqueeze(-1).bool(), -10).max(-2)[0]
        # face_feats_min = face_feats.masked_fill(~H.unsqueeze(-1).bool(), 10).min(-2)[0]
        # face_feats = face_feats.masked_fill(~H.unsqueeze(-1).bool(), torch.nan)
        # face_feats_sum = torch.nan_to_num(face_feats.nansum(-2))
        # face_feats_mean = torch.nan_to_num(face_feats.nanmean(-2))
        # face_emb = torch.cat([face_feats_max, face_feats_min, face_feats_sum, face_feats_mean], dim=-1)

        face_emb = self.face_embedder((H * mask[:, None, :]).sum(-1, keepdim=True).to(vertex_emb.dtype))
        # face_emb = self.face_embedder(face_emb)
        face_emb = torch.cat([face_emb, pos_emb[:, None, :].expand(-1, n_hyper, -1)], dim=-1)
        
        face_emb = face_emb + self.face_pos_embedder(torch.arange(face_emb.size(1), device=face_emb.device)) 
        input_emb = torch.cat([vertex_emb, face_emb], dim=1)

        type_ids = torch.cat([torch.zeros(n_nodes, device=X.device), torch.ones(n_hyper, device=H.device)]).long()
        type_emb = self.type_embedder(type_ids)
        
        input_emb = input_emb + type_emb
        attn_mask = torch.cat([mask, torch.ones(bs, n_hyper, dtype=torch.long, device=mask.device)], dim=1)
        
        top_left = torch.zeros(bs, n_nodes, n_nodes, dtype=torch.long, device=X.device)
        btn_right = torch.ones(bs, n_hyper, n_hyper, dtype=torch.long, device=X.device)
        top_right = H.transpose(-1, -2) + 2
        btn_left = H + 4
        
        relation = torch.cat([
            torch.cat([ top_left, top_right ], dim=2),
            torch.cat([ btn_left, btn_right ], dim=2),
        ], dim=1)

        relation = self.relation_embedder(relation).permute(0, 3, 1, 2).contiguous()
        attn_mask = attn_mask[..., None] * attn_mask[:, None, :]

        output = self.transformer(input_emb, attn_mask, relation)
        
        vertex_emb = output[:, :n_nodes, :]
        face_emb = output[:, n_nodes:, :]
        
        # [bs, n_hyper, n_nodes, 2 * hidden_dim]
        edge_emb = torch.cat([
            face_emb[:, :, None, :].expand(-1, -1, n_nodes, -1),
            vertex_emb[:, None, :, :].expand(-1, n_hyper, -1, -1)
            ], dim=-1)

        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)        
        H = self.edge_head(edge_emb).view(bs, n_hyper, n_nodes)
        return X, F.one_hot(E, num_classes=2).float() * 2e8 - 1e8, H
    
from ..digress.models.transformer_model import GraphTransformer

@Model.register("graph_tf")
class GraphTransformerW(Model):
    def __init__(self, position_encoder: Model):
        super().__init__()
        self.position_encoder = position_encoder
        y_dim = position_encoder.d_model
        self.model = GraphTransformer(
            n_layers=5,
            input_dims={"X": 3, "E": 2, "y": y_dim},
            hidden_mlp_dims={'X': 256, 'E': 128, 'y': 128},
            hidden_dims={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
            output_dims={"X": 3, "E": 2, "y": 0},
        )
    
    def forward(self, X, E, H, t, mask):
        y = self.position_encoder(t)
        E = F.one_hot(E, num_classes=2).float()

        outputs = self.model(X, E, y, mask)
        
        return outputs.X, outputs.E, H
    

@Model.register("gb_tf")
class GlobalTransformer(Model):
    def __init__(
        self,
        position_encoder: Model,
        hidden_dim: int = 256,
        ) -> None:
        super().__init__()
        
        self.position_encoder = position_encoder
        
        self.node_embedder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.face_embedder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        n_head = 8
        self.relation_embedder = nn.Embedding(6, n_head)
        self.transformer = TransformerEncoder(n_layers=4, n_head=n_head, d_k=hidden_dim//n_head, d_v=hidden_dim//n_head, d_model=hidden_dim, d_inner=2 * hidden_dim)
        
        self.node_head = nn.Sequential(
            nn.Linear(1 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, 3)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, X, E, H, t, mask):
        """
        X - [bs, n_nodes, 3]
        H - [bs, n_hyper, n_nodes]
        mask - [bs, n_nodes]
        """
        bs, n_nodes, _ = X.size()

        X = self.node_embedder(X)

        pos_emb = self.position_encoder(t)
        # [bs, n_nodes, hidden_dim]
        input_emb = (X + pos_emb[:, None, :])


        attn_mask = mask
        relation= E

        relation = self.relation_embedder(relation).permute(0, 3, 1, 2).contiguous()
        attn_mask = attn_mask[..., None] * attn_mask[:, None, :]

        vertex_emb = self.transformer(input_emb, None, relation)

        # [bs, n_hyper, n_nodes, 2 * hidden_dim]
        edge_emb = torch.cat([
            vertex_emb[:, :, None, :].expand(-1, -1, n_nodes, -1),
            vertex_emb[:, None, :, :].expand(-1, n_nodes, -1, -1)
            ], dim=-1)

        X = self.node_head(vertex_emb).view(bs, n_nodes, 3)        
        E = self.edge_head(edge_emb).view(bs, n_nodes, n_nodes, 2)

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return X, E, H