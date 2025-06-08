# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import  Attention, Mlp
from Embed import DataEmbedding, DataEmbedding2, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import copy
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.linear = nn.Linear(hidden_size,  out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        args = None,
        input_size=32,
        patch_size= 2,
        in_channels= 2*2*2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.args = args
        self.hidden_size = hidden_size




        self.cond_projection = Conv1d_with_init(2 * hidden_size, hidden_size, 1)

        self.Embedding_H = DataEmbedding(1, self.hidden_size, args=self.args, size1 = 24, size2 = 7)
        self.Embedding_halfH = DataEmbedding(1, self.hidden_size, args=self.args, size1=48, size2=7)
        self.Embedding_qartH = DataEmbedding(1, self.hidden_size, args=self.args, size1=96, size2=7)

        self.Embedding_plus_mask = DataEmbedding2(1, hidden_size, args=self.args)

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, hidden_size)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, hidden_size)
        )

        #---------------------------------------------------

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights_trivial()




    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        torch.nn.init.trunc_normal_(self.Embedding_H.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_H.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding_H.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.trunc_normal_(self.Embedding_halfH.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_halfH.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding_halfH.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.trunc_normal_(self.Embedding_qartH.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_qartH.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding_qartH.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:  # Check if elementwise_affine is True
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        T, H, W = self.args.info
        # c = self.out_channels
        t = T//self.args.t_patch_size
        h = H // self.args.patch_size
        w = W // self.args.patch_size
        sigma_split = 2 if self.learn_sigma else 1


        x = x.reshape(x.shape[0], t, h, w, self.args.t_patch_size, self.args.patch_size, self.args.patch_size,  sigma_split)
        # x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nthwabcs->nsatbhcw', x)
        imgs = x.reshape(x.shape[0],sigma_split, T,H, W)
        return imgs

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.hidden_size)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.hidden_size)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def pos_embed_enc(self, batch, input_size):
        pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])
        pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
            1, input_size[0], 1
        ) + torch.repeat_interleave(
            pos_embed_temporal[:,:input_size[0]],
            input_size[1] * input_size[2],
            dim=1,
        )
        pos_embed = pos_embed
        pos_embed = pos_embed.expand(batch, -1, -1)
        return pos_embed

    def forward(self, x, mask_origin, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """


        N, imput_dim, T, H, W = x.shape


        T = T // self.args.t_patch_size
        input_size = (T, H // self.args.patch_size, W // self.args.patch_size)
        pos_embed_sort = self.pos_embed_enc( N, input_size)


        if self.args.name_id == 'TrafficNJ':
            TimeEmb = self.Embedding_qartH(x, y, is_time=True)
        elif self.args.name_id == 'TrafficNC':
            TimeEmb = self.Embedding_halfH(x, y, is_time=True)
        else:
            TimeEmb = self.Embedding_H(x, y, is_time=True)


        x_noise_mask = x[:,1].unsqueeze(1)
        x_obs = x[:,0].unsqueeze(1)
        asdff = torch.cat([mask_origin, x_obs, x_noise_mask], dim=1)

        x_mask_emb, obs_embed, mask_embed = self.Embedding_plus_mask(asdff, x_obs, mask_origin)

        _, L, C = x_mask_emb.shape
        assert x_mask_emb.shape == pos_embed_sort.shape

        x_mask_emb_comb = x_mask_emb

        t = self.t_embedder(t)                   # (N, D)
        x_mask_emb_comb = x_mask_emb_comb + pos_embed_sort.to(device = t.device) +  TimeEmb
        c =  t.unsqueeze(1)  #+ mask_embed
        for block in self.blocks:
            x_mask_emb_comb = block(x_mask_emb_comb, c)                      # (N, T, D)
        x = self.final_layer(x_mask_emb_comb, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, mask_origin





def DiT_S_8(args=None,**kwargs):
    return DiT(args = args, patch_size=2, num_heads=8,  **kwargs)

DiT_models = { 'DiT-S/8':  DiT_S_8}


