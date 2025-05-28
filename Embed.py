import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np



    
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer





class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, t_patch_size = 1, hour_size=48, weekday_size = 7):
        super(TemporalEmbedding, self).__init__()

        hour_size = hour_size
        weekday_size = weekday_size

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.timeconv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=t_patch_size, stride=t_patch_size)

    def forward(self, x):

        x = x.long()
        hour_x = self.hour_embed(x[:,:,1])
        weekday_x = self.weekday_embed(x[:,:,0])
        timeemb = self.timeconv(hour_x.transpose(1,2)+weekday_x.transpose(1,2)).transpose(1,2)

        return timeemb


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(TokenEmbedding, self).__init__()
        kernel_size = [t_patch_size,patch_size, patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
#        self.alignment= nn.Linear(d_model, d_model, bias=True)
#        self.layernorm1 = nn.LayerNorm(d_model)
#        nn.init.constant_(self.alignment.weight, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        # x = F.relu(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*C*H*W, C]
#        x = self.alignment(x)
#        x = self.layernorm1(x)
        return x


class ObsEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(ObsEmbedding, self).__init__()
        kernel_size = [t_patch_size, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        #        self.alignment= nn.Linear(d_model, d_model, bias=True)
        #        self.layernorm1 = nn.LayerNorm(d_model)
        #        nn.init.constant_(self.alignment.weight, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        # x = F.relu(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*C*H*W, C]
        #        x = self.alignment(x)
        #        x = self.layernorm1(x)
        return x

class MaskEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(MaskEmbedding, self).__init__()
        kernel_size = [t_patch_size,patch_size, patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*C*H*W, C]
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7 ):
        super(DataEmbedding, self).__init__()
        self.args = args
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)
        self.temporal_embedding = TemporalEmbedding(t_patch_size = args.t_patch_size, d_model=d_model, hour_size  = size1, weekday_size = size2) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, is_time=1):

        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        N, T, C, H, W = x.shape
        TokenEmb = self.value_embedding(x[:,0].unsqueeze(1))
        TimeEmb = self.temporal_embedding(x_mark)
        assert TokenEmb.shape[1] == TimeEmb.shape[1] * H // self.args.patch_size * W // self.args.patch_size
        TimeEmb = torch.repeat_interleave(TimeEmb, TokenEmb.shape[1]//TimeEmb.shape[1], dim=1)
        assert TokenEmb.shape == TimeEmb.shape
        return  TimeEmb



class DataEmbedding2(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7 ):
        super(DataEmbedding2, self).__init__()
        self.args = args
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)
        self.obs_embedding = ObsEmbedding(c_in=c_in, d_model=d_model, t_patch_size=args.t_patch_size,
                                              patch_size=args.patch_size)
        self.mask_embedding = MaskEmbedding(c_in=c_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)

    def forward(self, x, obs, mask):
        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        N, T, C, H, W = x.shape
        TokenEmb = self.value_embedding(x)
        ObsEmb = self.obs_embedding(obs)
        MaskEmb = self.mask_embedding(mask)
        return TokenEmb, ObsEmb, MaskEmb


class DataEmbedding_u(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1=48, size2=7):
        super(DataEmbedding_u, self).__init__()
        self.args = args
        self.user_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, t_patch_size=args.t_patch_size,
                                             patch_size=args.patch_size)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层

    def forward(self, users):
        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''

        UseEmb = self.user_embedding(users)
        UseEmb = F.relu(UseEmb)
        UseEmb = self.dropout(UseEmb)  # 在激活函数后应用 Dropout

        return UseEmb


class Poiemb(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(Poiemb, self).__init__()
        kernel_size = [t_patch_size,patch_size, patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        x = F.relu(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*C*H*W, C]

        return x

class DataEmbedding_poi(nn.Module):
   def __init__(self, c_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7 ):
       super(DataEmbedding_poi, self).__init__()
       self.args = args
       self.poi_convert = Poiemb(c_in=c_in, d_model=d_model, t_patch_size=args.t_patch_size,
                                              patch_size=args.patch_size)
       self.fc0 = nn.Sequential(nn.Linear(d_model + d_model, d_model), nn.Sigmoid())
       self.fc1 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
       self.fc2 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
       self.fc_out = nn.Sequential(nn.Linear(d_model, d_model))
       self.norm_poi = nn.LayerNorm(d_model)


   def forward(self, x, time):
       '''
       x: N, T, C, H, W
       x_mark: N, T, D
       '''
       N, _, C, H, W = x.shape
       PoiEmb = self.poi_convert(x)# B, N, C
       emb_init = torch.cat((PoiEmb, time), dim = -1)
       h = self.fc0(emb_init)
       h = self.fc2(self.fc1(h) + h)
       h_out = self.fc_out(h) # 应该是B, N, C
       return self.norm_poi(h_out)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed





def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb








def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

