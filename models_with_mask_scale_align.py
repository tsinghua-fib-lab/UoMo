
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
from Embed import DataEmbedding, DataEmbedding2, DataEmbedding_u, DataEmbedding_poi, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import copy



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
        approx_gelu = lambda: nn.GELU(approximate="tanh")
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


class UoMo_model(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        args = None,
        input_size=32,
        patch_size=2,
        in_channels=2 * 2 * 2,
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
        #added_test-------------------------------------------------
        self.Embedding_H = DataEmbedding(1, self.hidden_size, args=self.args, size1 = 24, size2 = 7)
        self.Embedding_halfH = DataEmbedding(1, self.hidden_size, args=self.args, size1=48, size2=7)
        self.Embedding_qartH = DataEmbedding(1, self.hidden_size, args=self.args, size1=96, size2=7)

        self.userEmbedding = DataEmbedding_u(1, self.hidden_size, args=self.args)
        self.Embedding_plus_mask = DataEmbedding2(1, hidden_size, args=self.args)
        self.Embedding_poi= DataEmbedding_poi(15, hidden_size, args=self.args)
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





    def unpatchify(self, x):

        T, H, W = self.args.info
        t = T//self.args.t_patch_size
        h = H // self.args.patch_size
        w = W // self.args.patch_size
        sigma_split = 2 if self.learn_sigma else 1
        x = x.reshape(x.shape[0], t, h, w, self.args.t_patch_size, self.args.patch_size, self.args.patch_size,  sigma_split)
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

    def forward(self, x, user, poi, mask_origin, t, y):

        
        N, imput_dim, T, H, W = x.shape
        _,_,K,_,_ = poi.shape
        poi_trans = poi.permute(0,2,1,3,4).expand(-1,-1,T, -1, -1)[:,:15]

        if self.args.name_id == 'TrafficNJ':
            TimeEmb = self.Embedding_qartH(x, y, is_time=True)
        elif self.args.name_id == 'TrafficNC':
            TimeEmb = self.Embedding_halfH(x, y, is_time=True)
        else:
            TimeEmb = self.Embedding_H(x, y, is_time=True)

        T = T // self.args.t_patch_size
        input_size = (T, H // self.args.patch_size, W // self.args.patch_size)
        pos_embed_sort = self.pos_embed_enc( N, input_size)

        poi_emb = self.Embedding_poi(poi_trans, TimeEmb)

        x_noise_mask = x[:,1].unsqueeze(1)
        x_obs = x[:,0].unsqueeze(1)
        three_input = torch.cat([mask_origin, x_obs, x_noise_mask], dim=1)
        x_mask_emb, _, _ = self.Embedding_plus_mask(three_input, x_obs, mask_origin)
        users_embed = self.userEmbedding(user)

        _, L, C = x_mask_emb.shape
        assert x_mask_emb.shape == pos_embed_sort.shape

        t = self.t_embedder(t)

        x_mask_emb = x_mask_emb + pos_embed_sort.to(device=t.device) +  TimeEmb + poi_emb 

        c = t.unsqueeze(1) + users_embed
        #####-----------------------------------------------------------------------####
        for block in self.blocks:
            x_mask_emb = block(x_mask_emb, c)                   
        x = self.final_layer(x_mask_emb, c)
        x = self.unpatchify(x)
        return x





def UoMo_S(args=None,**kwargs):
    return UoMo_model(args = args, patch_size=2, num_heads=8,  **kwargs)

UoMo_models = { 'UoMo-S':  UoMo_S}



