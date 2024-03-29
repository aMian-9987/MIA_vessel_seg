import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import itertools
from collections.abc import Sequence
### the KRFormer with Gaussian relative position and dependency pruning

np.set_printoptions(threshold=1000)
#from utils.visualization import featuremap_visual, featuremap1d_visual

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, prob, **kwargs):
        return self.fn(self.norm(x), prob, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
### relative_position_distance
def relative_pos_index_dis(height=96, weight=96, depth=96):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords_d = torch.arange(depth)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 
    coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]- coords_flatten[None, :, :]   # 3, Wh*Ww, Wh*Ww, Wh*Wd
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, Wh*Wd 3
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2+ (relative_coords[:, :, 1].float()/depth) ** 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 2] += depth - 1  # shift to start from 0 # the relative pos in z axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww, Wh*Wd
    return relative_position_index, dis



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        ## here is the batch, sequence length in 1d, head number and size of each attention, arrange them for 3D data
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), torch.cat((q, k, v), dim=-1), attn #attn


class AttentionPruneKV(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width, depth = int(np.sqrt(num_patches)), int(np.sqrt(num_patches)),  int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # ------------------Gaussian relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width, depth)
        self.relative_position_index = relative_position_index  # hwd
        self.dis = dis.cuda()  # hwd
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.relative_position_bias_table2 = nn.Parameter(torch.zeros((2 * depth - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width
        self.depth  = depth
        # -----------------add function-----------------
        self.gate = nn.Parameter(torch.tensor(-2.0), requires_grad=False)  #
        self.neg_thresh = 0.9
        self.thresh_for_kv = nn.Linear(dim_head, 1, bias=False)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, prob, rpe=True):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.depth * self.weight, -1)*self.relative_position_bias_table2[self.relative_position_index.view(-1)].view(
                self.depth * self.weight, self.depth * self.weight, -1)  # n n h
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            factor = 1 / (2 * self.headsita ** 2 + 1e-6)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            pos_embed = torch.exp(-exponent)  # g hw hw
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01 * pos_embed[None, :, :, :]
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        # '''
        b, g, n, _ = attn.shape
        attn_max = torch.max(attn, dim=-1)[0]  # b g n
        attn_min = torch.min(attn, dim=-1)[0]  # b g n
        # q = rearrange(q, 'b g n d -> b n g d')
        # q[prob >= self.neg_thresh, :, :] = 0
        # q = rearrange(q, 'b n g d -> (b g) n d')
        q = rearrange(q, 'b g n d -> (b g) n d')
        thresh = self.sig(self.thresh_for_kv(q)) * self.sig(self.gate)  # bg n 1
        thresh = rearrange(thresh, '(b g) n d -> b g (n d)', b=b)
        thresh = attn_min + thresh * (attn_max - attn_min)

        record = attn - thresh[:, :, :, None]  # b g n n
        record[record > 0] = 1
        record[record <= 0] = 0
        prob = prob[:, None, :].repeat(1, g, 1)
        record[prob >= self.neg_thresh, :] = 0

        deno = torch.einsum('bcik,bcik->bci', [attn, record])
        attn = torch.mul(attn, record) / (deno[:, :, :, None] + 1e-6)
        # '''
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), self.attend(dots0)


class AttentionPruneKV_inference(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        height, width, depth = int(np.sqrt(num_patches)), int(np.sqrt(num_patches)),  int(np.sqrt(num_patches))

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        # ------------------Gaussian relative position embedding -------------------
        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index  # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads) * 0.1, requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.relative_position_bias_table2 = nn.Parameter(torch.zeros((2 * depth - 1) * (2 * width - 1), heads),
                                                         requires_grad=True)
        self.height = height
        self.weight = width
        self.depth  = depth
        self.height = height
        self.weight = width
        self.dim_head = dim_head

        # -----------------add function-----------------
        self.gate = nn.Parameter(torch.tensor(-2.0), requires_grad=False)  #
        self.neg_thresh = 0.9
        self.thresh_for_kv = nn.Linear(dim_head, 1, bias=False)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, prob, rpe=True):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        b, n, _ = x.shape

        q = rearrange(q, 'b g n d -> b n g d')
        q = q[:, prob[0, :] < self.neg_thresh, :, :]
        q = rearrange(q, 'b n g d -> (b g) n d')

        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.weight, self.depth * self.weight, -1)*self.relative_position_bias_table2[self.relative_position_index.view(-1)].view(
                self.depth * self.weight, self.depth * self.weight, -1)  # n n H
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = relative_position_bias[:, prob[0, :] < self.neg_thresh, :]
            factor = 1 / (2 * self.headsita ** 2 + 1e-6)  # g
            exponent = factor[:, None, None] * self.dis[None, :, :]  # g hw hw
            exponent = exponent[:, prob[0, :] < self.neg_thresh, :]
            pos_embed = torch.exp(-exponent)  # g hwd
            dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]
        else:
            dots = dots0

        attn = self.attend(dots)  # b g n n
        if q.shape[1] > 0:
            #'''
            attn_max = torch.max(attn, dim=-1)[0]  # b g n
            attn_min = torch.min(attn, dim=-1)[0]  # b g n

            thresh = self.sig(self.thresh_for_kv(q)) * self.sig(self.gate)  # bg n 1
            thresh = rearrange(thresh, '(b g) n d -> b g (n d)', b=b)
            thresh = attn_min + thresh * (attn_max - attn_min)

            record = attn - thresh[:, :, :, None]  # b g n n
            record[record > 0] = 1
            record[record <= 0] = 0
            #print(torch.sum(record) / (b * g * n * n + 0.0000001))

            deno = torch.einsum('bcik,bcik->bci', [attn, record])
            attn = torch.mul(attn, record) / (deno[:, :, :, None] + 1e-6)
        #'''
        out0 = torch.matmul(attn, v)
        out0 = rearrange(out0, 'b h n d -> b n (h d)')
        out0 = self.to_out(out0)
        out = x * 0
        out[:, prob[0, :] < self.neg_thresh, :] = out0

        return out, self.attend(dots0)


class Transformer(nn.Module):
    def __init__(self, dim, layer_depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layer_depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        qkvs, attns = [], []
        for attn, ff in self.layers:
            ax, qkv, attn = attn(x)
            qkvs.append(qkv)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, qkvs, attns
    
class TransformerDown(nn.Module):
    """used in before downsampling without prune"""

    def __init__(self, in_channels, out_channels, image_size, layer_depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth // patch_depth ==0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth) 
        self.patch_dim = in_channels * patch_height * patch_width * patch_depth
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, layer_depth, heads, dim_head, self.mlp_dim, dropout, num_patches)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=image_height//patch_height, w=image_width//patch_width, d=image_depth // patch_depth),
        )


    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, n, h, w, d) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        
        x = self.dropout(x)
        # transformer layer
        ax, qkvs, attns = self.transformer(x)
        out = self.recover_patch_embedding(ax)
        return out, qkvs, attns
##


"""Dependency pruning by gaussian"""
class TransformerSPrune(nn.Module):
    def __init__(self, dim, layer_depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layer_depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionPruneKV(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, prob):
        attns = []
        for attn, ff in self.layers:
            ax, attn = attn(x, prob)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, attns

class TransformerSPrune_Test(nn.Module):
    def __init__(self, dim, layer_depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layer_depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, AttentionPruneKV_inference(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, prob):
        attns = []
        for attn, ff in self.layers:
            ax, attn = attn(x, prob)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, attns

class KRtransformer(nn.Module):
    """Dependency pruning by gaussian and following downsampling"""

    def __init__(self, in_channels, out_channels, image_size, layer_depth=2, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth // patch_depth ==0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth) 
        self.patch_dim = in_channels * patch_height * patch_width * patch_depth
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4
        image_size: Sequence[int] | int
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerSPrune(self.dmodel, layer_depth, heads, dim_head, self.mlp_dim, dropout, num_patches)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=image_height//patch_height, w=image_width//patch_width, d=image_depth // patch_depth),
        )

        self.pred_class = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        sx = self.pred_class(x)  # b n d h w -> b 2 d h w 
        sxp = self.softmax(sx)
        sxp_neg_1d = rearrange(sxp[:, 0, :, :], 'b h w d-> b (h w d)')  # b n

        x = self.to_patch_embedding(x)  # (b, n, h, w, d) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n]
        
        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x, sxp_neg_1d)
        out = self.recover_patch_embedding(ax)
        return out, sx, attns

class KRTransformer_Test(nn.Module):
    """Dependency pruning by gaussian"""

    def __init__(self, in_channels, out_channels, image_size, layer_depth=2,  patch_size=2, heads=6,
                 dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth // patch_depth ==0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth) 
        self.patch_dim = in_channels * patch_height * patch_width * patch_depth
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerSPrune_Test(self.dmodel, layer_depth, heads, dim_head, self.mlp_dim, dropout, num_patches)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=image_height//patch_height, w=image_width//patch_width, d=image_depth // patch_depth),
        )

        self.pred_class = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        sx = self.pred_class(x)  # b n h w -> b 2 h w
        sxp = self.softmax(sx)
        sxp_neg_1d = rearrange(sxp[:, 0, :, :], 'b h w d-> b (h w d)')  # b n

        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        # x += self.pos_embedding[:, :n]

        x = self.dropout(x)
        # transformer layer
        ax, attns = self.transformer(x, sxp_neg_1d)
        out = self.recover_patch_embedding(ax)
        return out, sx, attns