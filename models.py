"""Graph network models for vehicle drag regression from surface meshes.

Main (and only) model: HGPTrans -- GIN message passing followed by
Transolver-style physics attention (learned slice tokens for irregular meshes)
and TopK hierarchical graph pooling, with a multi-scale pooled readout head.

Note: the forward passes use dense reshapes that assume every batch contains
exactly one graph, so keep batch_size=1 in the DataLoader.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GINConv, TopKPooling, global_max_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'lrelu': partial(nn.LeakyReLU, negative_slope=0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}


def scatter_add_pytorch(src, index, dim=0, dim_size=None):
    """Scatter-add helper (plain PyTorch implementation)."""
    if dim_size is None:
        dim_size = index.max() + 1
    output = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    output.index_add_(dim, index, src)
    return output


class HGPTrans(torch.nn.Module):
    """Hierarchical graph pooling + physics attention drag regressor."""

    def __init__(self, input_dim=6, hidden_dim=256, out_dim=1, pooling_ratio=0.8,
                 dropout_ratio=0, n_layers=3, n_head=8, slice_num=32):
        super(HGPTrans, self).__init__()

        # blocks
        self.layers = n_layers
        blocks_inputdims = [input_dim] + [hidden_dim for _ in range(self.layers - 1)]

        # slice_num kept fixed at 32 to stay compatible with released checkpoints
        self.blocks = nn.ModuleList([
            HGPAttenBlock(blocks_inputdims[i], ratio=pooling_ratio, transolver=True,
                          hidden_dim=hidden_dim, num_heads=n_head, slice_num=32, act='gelu')
            for i in range(self.layers)
        ])

        # linear head
        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, out_dim)
        self.dropout_ratio = dropout_ratio

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.04)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x_list = []
        for i in range(self.layers):
            x, edge_index, edge_attr, batch = self.blocks[i](x, edge_index, edge_attr, batch)
            x_list.append(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
        x = sum(F.leaky_relu(x_i) for x_i in x_list)

        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)
        return x


class HGPAttenBlock(torch.nn.Module):
    """One hierarchy level: GIN conv -> physics attention -> TopK pooling."""

    def __init__(self, in_channels, ratio=0.8, transolver=True, hidden_dim=256,
                 num_heads=8, slice_num=32, act='gelu'):
        super(HGPAttenBlock, self).__init__()
        self.conv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU()
            )
        )
        # custom information score & pooling
        self.in_channels = in_channels
        self.pool = TopKPooling(hidden_dim, ratio)
        self.ratio = ratio
        self.calc_information_score = NodeInformationScore()

        # Transolver physics attention
        self.transolver = transolver
        if self.transolver:
            self.transblock = Transolver_block(num_heads, hidden_dim, act=act, slice_num=slice_num)

        self.downsample = True

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = self.conv(x, edge_index, edge_attr)

        if self.transolver:
            # NOTE: dense reshape assumes batch_size=1
            x = x.view(batch[-1] + 1, -1, x.shape[-1])  # batch, N, C
            x = self.transblock(x)  # batch, N, C
            x = x.view(-1, x.shape[-1])

        if self.downsample:
            x_information_score = self.calc_information_score(x, edge_index, edge_attr)
            score = torch.sum(torch.abs(x_information_score), dim=1)

            x, induced_edge_index, induced_edge_attr, batch, perm, _ = self.pool(
                x, edge_index, batch=batch, attn=score)
            return x, induced_edge_index, induced_edge_attr, batch

        return x, edge_index, edge_attr, batch


class NodeInformationScore(MessagePassing):
    """Node importance score used as TopKPooling attention input."""

    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add_pytorch(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Transolver physics attention: learned slice tokens for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        # (1) slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)  # B, H, G, C
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # (2) attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # (3) deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    """Small MLP with optional residual connections."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            activation = ACTIVATION[act]()
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation) for _ in range(n_layers)])
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block built around the physics attention."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout=0,
            act='gelu',
            mlp_ratio=2,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads,
                                                     dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


def build_model(name, input_dim=6, hidden_dim=256, out_dim=1, pooling_ratio=0.8,
                n_layers=4, n_head=8, slice_num=32):
    """Instantiate a model by name (replaces the previous eval()-based lookup)."""
    if name == 'HGPTrans':
        return HGPTrans(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                            pooling_ratio=pooling_ratio, n_layers=n_layers,
                            n_head=n_head, slice_num=slice_num)
    raise ValueError(f"Unknown model '{name}'. Supported model: HGPTrans")
