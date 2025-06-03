# models/changeformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Note: The full ChangeFormer architecture is complex and typically involves
# cloning the entire `wgcban/ChangeFormer` repository and importing its
# specific `networks.py` and `config.py` modules.
# For this project, a simplified version based on common Transformer-based
# U-Net like architectures for segmentation is provided.
# For the exact official ChangeFormerV6, you would need to integrate
# the `ChangeFormer/models/networks.py` content directly.

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ChangeFormer(nn.Module):
    """
    A simplified ChangeFormer-like model for binary change detection.
    This architecture mimics a U-Net with Transformer blocks in the encoder/decoder.
    For the exact official ChangeFormerV6, please refer to the original GitHub repository.
    """
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256,
                 depths=[2, 2, 2, 2], num_heads=[4, 8, 16, 32],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False,
                 decoder_embed_dim=256, decoder_depths=[2, 2, 2, 2],
                 decoder_num_heads=[4, 8, 16, 32], align_corners=True):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.embed_dim = embed_dim
        self.align_corners = align_corners

        # Encoder (similar to Swin Transformer or hierarchical Vision Transformer)
        # We will use simple conv layers for downsampling and then Transformer blocks
        self.downsample1 = nn.Sequential(
            nn.Conv2d(input_nc, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        # Simplified: one block for each level of encoder
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads[0], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(embed_dim, num_heads[1], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(embed_dim, num_heads[2], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(embed_dim, num_heads[3], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        ])
        
        # Decoder (upsampling with Transformer blocks)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, decoder_embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim // 4, decoder_embed_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 8),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim // 8, decoder_embed_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 16),
            nn.ReLU(inplace=True)
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim // 2, decoder_num_heads[0], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(decoder_embed_dim // 4, decoder_num_heads[1], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(decoder_embed_dim // 8, decoder_num_heads[2], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer),
            Block(decoder_embed_dim // 16, decoder_num_heads[3], mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        ])

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(decoder_embed_dim // 16, output_nc, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is assumed to be (B, 6, H, W) where 6 comes from concatenated RGB images
        
        # Encoder
        # Downsample to get features at different scales
        feat1 = self.downsample1(x) # (B, embed_dim, H/4, W/4)
        
        # Apply Transformer blocks
        # Flatten and apply transformer blocks
        B, C, H_feat, W_feat = feat1.shape
        feat1_flat = feat1.flatten(2).transpose(1, 2) # (B, N_patches, C)
        
        # Apply encoder blocks sequentially (simplified)
        for block in self.encoder_blocks:
            feat1_flat = block(feat1_flat)
        
        feat_encoded = feat1_flat.transpose(1, 2).reshape(B, C, H_feat, W_feat) # Reshape back to (B, C, H/4, W/4)

        # Decoder
        # Upsample and apply decoder blocks
        x = self.upsample1(feat_encoded) # (B, decoder_embed_dim // 2, H/2, W/2)
        B, C, H_up, W_up = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.decoder_blocks[0](x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H_up, W_up)

        x = self.upsample2(x) # (B, decoder_embed_dim // 4, H, W)
        B, C, H_up, W_up = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.decoder_blocks[1](x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H_up, W_up)
        
        x = self.upsample3(x) # (B, decoder_embed_dim // 8, 2*H, 2*W) - assuming original H,W was 256
        B, C, H_up, W_up = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.decoder_blocks[2](x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H_up, W_up)

        x = self.upsample4(x) # (B, decoder_embed_dim // 16, 4*H, 4*W) - assuming original H,W was 256
        B, C, H_up, W_up = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.decoder_blocks[3](x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H_up, W_up)


        # Final segmentation head
        logits = self.segmentation_head(x) # (B, output_nc, H_orig, W_orig)

        return logits