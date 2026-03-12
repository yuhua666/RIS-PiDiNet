import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

import os
import numpy as np
import torch.nn.functional as F

class PDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(PDC, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
    
class LieGroupRotationPDC(nn.Module):
    """S-PDC: Symmetry-aware Pixel Difference Convolution (kernel_size=5).

    Modulates PDC weights with a fixed PHT (Polar Harmonic Transform) symmetry
    kernel H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i), scaled by a trainable
    coefficient alpha, then averages responses over 8 SO(2) rotations.

    Paper eq.:
        y = alpha * sum_{i != c} w_i * H_i^(n,l) * (x_i - q)  (q absorbed into PDC diff)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LieGroupRotationPDC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        # Trainable harmonic order coefficient alpha_{n,l} (initialised to 1.0)
        self.alpha = nn.Parameter(torch.ones(1))
        # Pre-calibrated per-position spectral weights sw_i = target_i / H_i^{(n,l)}
        # such that H_i * sw_i recovers the centre-surround symmetry kernel exactly
        self.register_buffer('_pht_spectral_weights', self._precompute_spectral_weights())

    def _precompute_spectral_weights(self):
        """Offline: solve sw_i = target_i / H_i^{(n=2,l=0)} on the 5x5 grid (k=2).

        H = cos(2*pi*2*r^2):  centre->1, cross-neighbours->cos(pi)=-1, others->+/-1
        sw = target / H    :  centre->-4, cross-neighbours->-1, others->0
        => H * sw = target  (exact recovery of the symmetry kernel)
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        H = torch.cos(2.0 * math.pi * 2 * r2 + 0 * theta)   # PHT order (n=2, l=0)
        target = torch.zeros(N, N)
        target[k, k]     = -4.0
        target[k - 1, k] =  1.0
        target[k + 1, k] =  1.0
        target[k, k - 1] =  1.0
        target[k, k + 1] =  1.0
        sw = torch.where(target != 0, target / H, torch.zeros_like(target))
        return sw  # [N, N]

    def forward(self, x):
        weights = self.conv.weight
        device = x.device
        weights = weights.to(device)

        # PHT harmonic kernel H_i^{(n,l)}, scaled by trainable alpha_{n,l}
        pht_kernel = self.create_pht_kernel(x.device)

        out = self.apply_lie_group_rotation(x, weights * self.alpha * pht_kernel, device, 2) #1 #9
        
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = self.apply_lie_group_rotation(x, kernel_diff, device, 0)
        
        '''if out.shape[2] != x.shape[2] or out.shape[3] != x.shape[3]:
            out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        if out_diff.shape[2] != x.shape[2] or out_diff.shape[3] != x.shape[3]:
            out_diff = F.interpolate(out_diff, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)'''
            
        output = out - self.theta * out_diff

        return output

    def create_pht_kernel(self, device):
        """Evaluate H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i)  [PHT order n=2, l=0]
        on the discrete 5x5 grid, then apply pre-calibrated spectral weights so that
        the element-wise product H_i * sw_i recovers the target centre-surround
        symmetry kernel (Eq.\,(6) of the paper).
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32, device=device) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        # H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i),  order (n=2, l=0)
        H = torch.cos(2.0 * math.pi * 2 * r2 + 0 * theta)
        # Combine with pre-calibrated spectral weights -> target symmetry kernel
        pht_kernel = H * self._pht_spectral_weights.to(device)
        pht_kernel = pht_kernel + eps
        pht_kernel = pht_kernel.unsqueeze(0).unsqueeze(0)
        pht_kernel = pht_kernel.repeat(self.in_channels, 1, 1, 1)
        return pht_kernel

    def apply_lie_group_rotation(self, x, weights, device, pad):
        rotation_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        rotated_weights = []
        output = []
        
        for angle in rotation_angles:
            rotation_matrix = self.rotation_matrix_2d(angle).to(device)
            rotated_weight = self.rotate_kernel(weights, rotation_matrix, device)
            rotated_weights.append(rotated_weight)
            
            output.append(F.conv2d(x, rotated_weight, stride=self.stride, padding=pad, dilation=self.dilation, groups=self.groups))

        return torch.stack(output).mean(0)

    def rotation_matrix_2d(self, angle):
        return torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]
        ])

    def rotate_kernel(self, weights, rotation_matrix, device):
        _, _, kernel_height, kernel_width = weights.shape
        rotated_kernel = torch.zeros_like(weights)
        
        for h in range(kernel_height):
            for w in range(kernel_width):
                rotated_h = int(h * rotation_matrix[0, 0] + w * rotation_matrix[0, 1])
                rotated_w = int(h * rotation_matrix[1, 0] + w * rotation_matrix[1, 1])
                
                if 0 <= rotated_h < kernel_height and 0 <= rotated_w < kernel_width:
                    rotated_kernel[:, :, rotated_h, rotated_w] = weights[:, :, h, w]
        
        return rotated_kernel.to(device)

class LieGroupRotationPDC2(nn.Module):
    """S-PDC: Symmetry-aware Pixel Difference Convolution (kernel_size=7, dilated).

    Modulates PDC weights with a fixed PHT (Polar Harmonic Transform) symmetry
    kernel H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i), scaled by a trainable
    coefficient alpha, then averages responses over 8 SO(2) rotations.

    Paper eq.:
        y = alpha * sum_{i != c} w_i * H_i^(n,l) * (x_i - q)  (q absorbed into PDC diff)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LieGroupRotationPDC2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        # Trainable harmonic order coefficient alpha_{n,l} (initialised to 1.0)
        self.alpha = nn.Parameter(torch.ones(1))
        # Pre-calibrated per-position spectral weights sw_i = target_i / H_i^{(n,l)}
        # such that H_i * sw_i recovers the centre-surround symmetry kernel exactly
        self.register_buffer('_pht_spectral_weights', self._precompute_spectral_weights())

    def _precompute_spectral_weights(self):
        """Offline: solve sw_i = target_i / H_i^{(n=1,l=0)} on the 7x7 grid (k=3).

        H = cos(2*pi*r^2): centre->1, cross-neighbours->cos(2*pi/9)~0.766, others vary
        sw = target / H  : centre->-4, cross-neighbours->1/cos(2*pi/9)~1.305, others->0
        => H * sw = target  (exact recovery of the symmetry kernel)
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        H = torch.cos(2.0 * math.pi * 1 * r2 + 0 * theta)   # PHT order (n=1, l=0)
        target = torch.zeros(N, N)
        target[k, k]     = -4.0
        target[k - 1, k] =  1.0
        target[k + 1, k] =  1.0
        target[k, k - 1] =  1.0
        target[k, k + 1] =  1.0
        sw = torch.where(target != 0, target / H, torch.zeros_like(target))
        return sw  # [N, N]

    def forward(self, x):
        weights = self.conv.weight
        device = x.device
        weights = weights.to(device)
        
        # PHT harmonic kernel H_i^{(n,l)}, scaled by trainable alpha_{n,l}
        pht_kernel = self.create_pht_kernel(x.device)
        out = self.apply_lie_group_rotation(x, weights * self.alpha * pht_kernel, device, 9) #1 #9
        
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = self.apply_lie_group_rotation(x, kernel_diff, device, 0)
        
        output = out - self.theta * out_diff

        return output

    def create_pht_kernel(self, device):
        """Evaluate H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i)  [PHT order n=1, l=0]
        on the discrete 7x7 grid, then apply pre-calibrated spectral weights so that
        the element-wise product H_i * sw_i recovers the target centre-surround
        symmetry kernel (Eq.\,(6) of the paper).
        """
        N = self.kernel_size
        k = N // 2
        eps = 1e-6
        coords = torch.arange(N, dtype=torch.float32, device=device) - k
        grid_v, grid_u = torch.meshgrid(coords, coords)
        r2 = (grid_u ** 2 + grid_v ** 2) / (k ** 2 + eps)
        theta = torch.atan2(grid_v, grid_u + eps)
        # H_i^{(n,l)} = cos(2*pi*n*r_i^2 + l*theta_i),  order (n=1, l=0)
        H = torch.cos(2.0 * math.pi * 1 * r2 + 0 * theta)
        # Combine with pre-calibrated spectral weights -> target symmetry kernel
        pht_kernel = H * self._pht_spectral_weights.to(device)
        pht_kernel = pht_kernel + eps
        pht_kernel = pht_kernel.unsqueeze(0).unsqueeze(0)
        pht_kernel = pht_kernel.repeat(self.in_channels, 1, 1, 1)
        return pht_kernel

    def apply_lie_group_rotation(self, x, weights, device, pad):
        rotation_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        rotated_weights = []
        output = []
        
        for angle in rotation_angles:
            rotation_matrix = self.rotation_matrix_2d(angle).to(device)
            rotated_weight = self.rotate_kernel(weights, rotation_matrix, device)
            rotated_weights.append(rotated_weight)

            output.append(F.conv2d(x, rotated_weight, stride=self.stride, padding=pad, dilation=self.dilation, groups=self.groups))

        return torch.stack(output).mean(0)

    def rotation_matrix_2d(self, angle):
        return torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]
        ])

    def rotate_kernel(self, weights, rotation_matrix, device):
        _, _, kernel_height, kernel_width = weights.shape
        rotated_kernel = torch.zeros_like(weights)
        
        for h in range(kernel_height):
            for w in range(kernel_width):
                rotated_h = int(h * rotation_matrix[0, 0] + w * rotation_matrix[0, 1])
                rotated_w = int(h * rotation_matrix[1, 0] + w * rotation_matrix[1, 1])
                
                if 0 <= rotated_h < kernel_height and 0 <= rotated_w < kernel_width:
                    rotated_kernel[:, :, rotated_h, rotated_w] = weights[:, :, h, w]
        
        return rotated_kernel.to(device)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = LieGroupRotationPDC(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = LieGroupRotationPDC2(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):  
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        
        return x * attn



class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class LSKNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)



    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet, self).init_weights()
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict