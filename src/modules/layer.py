import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tango.integrations.torch import Model
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_sparse import SparseTensor, matmul

from typing import List, Optional, Tuple, Union


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

@Model.register("mesh_conv")
class MeshConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggr,
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)
            
        self.msg_mlp = nn.Sequential(
            Linear(in_channels[0] * 3, in_channels[0] * 3, bias=True),
            nn.LeakyReLU(),
            Linear(in_channels[0] * 3, in_channels[0], bias=True),
        )

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(x_ij)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
