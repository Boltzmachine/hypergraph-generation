import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from tango.integrations.torch import Model
from torch_geometric.nn import HypergraphConv, HeteroConv, GATConv
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch
from .dataset import BipartiteData

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
    def __init__(self,
                 hidden_dim: int,
                 position_encoder: Model,
            ) -> None:
        super().__init__()
        self.position_encoder = position_encoder
        self.node_embedder = nn.Linear(3, 3 * hidden_dim)
        self.edge_embedder = nn.Embedding(2, hidden_dim)
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
        H = self.edge_embedder(H)
        
        X = X.view(bs, -1)
        H = H.view(bs, -1)
        t = self.position_encoder(t)
        f = torch.cat([X, H, t], dim=-1)
        X = self.node_head(f).view(bs, 8, 3)
        H = self.edge_head(f).view(bs, 6, 8)
        return X, H


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
    
    
class BipartiteDenseGATConv2(Model):
    def __init__(
            self,
            hidden_dim: int,
            heads: int = 8,
        ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        assert hidden_dim % heads == 0
        
        self.lin_src_pos = nn.Linear(hidden_dim, hidden_dim//heads)
        self.lin_src_neg = nn.Linear(hidden_dim, hidden_dim//heads)
        self.lin_src_att = nn.Linear(hidden_dim, hidden_dim//heads)
        
        self.lin_tgt_att = nn.Linear(hidden_dim, hidden_dim//heads)
        
        self.attn = nn.Linear(2 * hidden_dim // heads, heads, bias=False)

        self.lin_root = nn.Linear(hidden_dim, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        
    
    def forward(self, x_src, x_tgt, edge_attr, mask=None):
        """
        x_tgt: [bs, n_tgt_nodes, n_features]
        x_src: [bs, n_src_nodes, n_features]
        edge_attr: [bs, n_src_nodes, n_tgt_nodes]
        """
        x_root = x_tgt
        edge_attr = edge_attr.transpose(1, -1)
        
        # [bs, n_src_nodes, hidden_dim//head]
        x_src_pos = self.lin_src_pos(x_src)
        x_src_neg = self.lin_src_neg(x_src)
        
        x_tgt_att = self.lin_tgt_att(x_tgt)
        x_src_att = self.lin_src_att(x_src)
        
        # attention coefficients [bs, n_tgt_nodes, n_src_nodes, heads]
        alpha = self.attn(torch.cat([x_src_att.unsqueeze(1).expand(-1, x_tgt_att.size(1), -1, -1), x_tgt_att.unsqueeze(2).expand(-1, -1, x_src_att.size(1), -1)], dim=-1))
        # attention coefficients [bs, heads, n_tgt_nodes, n_src_nodes]
        alpha = alpha.permute(0, 3, 1, 2)
        alpha = F.leaky_relu(alpha)
        if mask is not None:
            mask = ((mask - 1) * 1e10)[:, None, None, :]
            alpha = alpha + mask
            
        alpha = torch.softmax(alpha, dim=-1)
        
        # [bs, n_tgt_nodes, n_src_nodes, hidden_dim//head]
        x_src_msg = (edge_attr == 1).unsqueeze(-1) * x_src_pos[:, None, :, :] + (edge_attr == 0).unsqueeze(-1) * x_src_neg[:, None, :, :]
        
        # [bs, heads, n_tgt_nodes, n_features] = [bs, heads, n_tgt_nodes, n_src_nodes, 1] * [bs, 1, n_tgt_nodes, n_src_nodes, n_features]
        src_msg = (alpha.unsqueeze(-1) * x_src_msg[:, None, ...]).sum(-2)
        # [bs, n_tgt_nodes, heads, n_features] 
        src_msg = src_msg.transpose(1, 2)
        src_msg = src_msg.reshape(src_msg.size(0), src_msg.size(1), -1)
        x_root = self.lin_root(x_root) + src_msg
        x_root = self.feed_forward(x_root)
        
        return x_root
    

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
        self.conv1 = pyg_nn.SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
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
        bs, n_tgt_nodes, hidden_dim = x_tgt.size()
        _, n_src_nodes, hidden_dim = x_src.size()
        
        def create_mask_if_not_exist(mask, x_ref):
            if mask is not None:
                return mask
            else:
                mask = torch.ones(x_ref.size(0), x_ref.size(1), dtype=torch.long, device=x_ref.device)
                return mask
        
        src_mask = create_mask_if_not_exist(src_mask, x_src)
        tgt_mask = create_mask_if_not_exist(tgt_mask, x_tgt)
        
        def get_offset(batch_idx, indicator):
            if isinstance(indicator, int):
                offset = batch_idx * indicator
            elif isinstance(indicator, torch.Tensor):
                assert indicator.dim() == 2
                indicator = indicator.sum(1).cumsum(0)
                indicator = torch.roll(indicator, 1)
                indicator[0] = 0
                offset = indicator[batch_idx]
            return offset
        
        def adj_to_edge_index(adj):
            edge_index = adj.nonzero()
            src_offset = get_offset(edge_index[:, 0], src_mask if src_mask is not None else n_src_nodes)
            tgt_offset = get_offset(edge_index[:, 0], tgt_mask if tgt_mask is not None else n_tgt_nodes)
            edge_index = edge_index[:, 1:]
            edge_index[:, 0] += src_offset
            edge_index[:, 1] += tgt_offset
            edge_index = edge_index.T
            # edge_index = torch.sort(edge_index, 1)[0]
            return edge_index


        adj1 = adj * src_mask[:, :, None] * tgt_mask[:, None, :]
        adj2 = (1 - adj) * src_mask[:, :, None] * tgt_mask[:, None, :]
        
        edge_index1 = adj_to_edge_index(adj1)
        edge_index2 = adj_to_edge_index(adj2)
        
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
        x_tgt_1 = self.conv1((x_src, x_tgt), edge_index1)
        x_tgt_2 = self.conv2((x_src, x_tgt), edge_index2)

            
        x_tgt = self.downscale(torch.cat([x_tgt_1, x_tgt_2], dim=-1))

        # x_tgt, _ = to_dense_batch(x_tgt, batch.x_t_batch)
        x_tgt = x_tgt.view(bs, n_tgt_nodes, hidden_dim)

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
        # [bs, n_hyper, n_nodes]
        # edge_emb = torch.cat([edge_emb, H[..., None].expand(-1, -1, -1, self.hidden_dim).to(edge_emb.dtype)], dim=-1)
        H = self.edge_head(edge_emb).squeeze(-1)
    
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