from functools import partial
from typing import Callable, Optional, Tuple, Union
import math
import logging
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from pathlib import Path

from .ops.modules import MSDeformAttn
from .adaptor_modules import DeformNeck
from .swin import SwinTransformer
from .hybrid_backbone import Hybrid


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not (stride == 1 and in_planes == planes):
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class Backbone(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d):
        super(Backbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1/2
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_layer=norm_layer) # 1/2
        self.layer2 = self._make_layer(96, stride=2, norm_layer=norm_layer) # 1/4
        self.layer3 = self._make_layer(128, stride=1, norm_layer=norm_layer) # 1/4

        self.conv2 = nn.Conv2d(128, output_dim, 1, 1, 0)

        self.output_dim = output_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = 2 * (x / 255.0) - 1.0
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/4
        x = self.conv2(x)

        out = [x, F.avg_pool2d(x, kernel_size=2, stride=2)]  # high to low res

        return out


class SwinAdaptor(nn.Module):
    def __init__(self, out_channels, drop_path_rate=0.):
        super().__init__()
        self.backbone = SwinTransformer(
            depths=(2, 2, 6, 2),
            drop_path_rate=drop_path_rate,
            embed_dim=96,
            num_heads=(3, 6, 12, 24),
        )

        self.neck = DeformNeck(
            dim=out_channels,
            in_channel_list=[96, 192, 384, 768],
            drop_path=0.0,
            deform_ratio=0.5,
            with_cp=False
        )
        self.output_dim = self.neck.dim

        self.neck.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        self.register_buffer(
            "mean", torch.as_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[None, :, None, None]
        )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x):
        # x.sub_(self.mean).div_(self.std)
        outs = self.backbone.forward(x)
        features = [outs[name] for name in ["p0", "p1", "p2", "p3"]]
        out = self.neck(x, features)  # 4s
        out = [out, F.avg_pool2d(out, kernel_size=2, stride=2)]  # high to low res

        return out


def checkpoint_filter_fn(state_dict):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        if "attn_mask" in k:
            continue  # skip buffers that should not be persistent

        if any([k.startswith(n) for n in ('norm', 'head')]):
            continue

        out_dict[k] = v
    return out_dict


class RepVitBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        model = timm.create_model('repvit_m2_3', pretrained=True)
        self.stem = model.stem
        self.stage0 = model.stages[0]
        self.stage1 = model.stages[1]
        self.stage2 = model.stages[2]
        self.stage3 = model.stages[3]

        self.neck = DeformNeck(
            dim=out_channels,
            in_channel_list=[80, 160, 320, 640],
            drop_path=0.0,
            deform_ratio=0.5,
            with_cp=False
        )
        self.output_dim = self.neck.dim

        self.neck.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x):
        c2 = self.stem(x)  # [bz, 48, H/4, W/4]
        c2 = self.stage0(c2)  # [bz, 64, H/4, W/4]
        c3 = self.stage1(c2)  # [bz, 128, H/8, W/8]
        c4 = self.stage2(c3)  # [bz, 256, H/16, W/16]
        c5 = self.stage3(c4)  # [bz, 512, H/32, W/32]
        features = [c2, c3, c4, c5]
        out = self.neck(x, features)  # 4s
        out = [out, F.avg_pool2d(out, kernel_size=2, stride=2)]  # high to low res

        return out


class MpVitBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        from torch.hub import load_state_dict_from_url
        from stereo.modeling.backbones.mpvit import mpvit_tiny, mpvit_xsmall

        model = mpvit_xsmall()
        checkpoint = load_state_dict_from_url('mpvit_xsmall.pth', map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)

        self.stem = model.stem
        self.patch_embed_stages = model.patch_embed_stages
        self.mhca_stages = model.mhca_stages

        self.neck = DeformNeck(
            dim=out_channels,
            in_channel_list=[128, 192, 256, 256],
            drop_path=0.0,
            deform_ratio=0.5,
            with_cp=False
        )
        self.output_dim = self.neck.dim

        self.neck.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, inputs):
        x = self.stem(inputs)
        feats = []
        for idx in range(4):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)
            feats.append(x)
        # [128, 192, 256, 256]
        out = self.neck(inputs, feats)  # 4s
        out = [out, F.avg_pool2d(out, kernel_size=2, stride=2)]  # high to low res

        return out


def create_backbone(model_type, norm_fn, out_channels, drop_path):
    model_type = model_type
    if model_type == "resnet":
        if norm_fn == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm_fn == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f'Invalid backbone normalization type: {norm_fn}')
        backbone = Backbone(out_channels, norm_layer)
    elif model_type == "swin":
        backbone = SwinAdaptor(out_channels=out_channels, drop_path_rate=drop_path)
        pretrained = True
        if pretrained:
            current_dir = Path(__file__).resolve()
            weight_url = current_dir.parent.parent.parent.parent.parent / 'ckpt/swin_tiny_patch4_window7_224.pth'
            if weight_url:
                weight = torch.load(weight_url, map_location="cpu")
                weight = checkpoint_filter_fn(weight)
                backbone.backbone.load_state_dict(weight)
                logger = logging.getLogger(__name__)
                print("Load pretrained backbone weights from {}".format(weight_url))
    elif model_type == 'hybrid':
        backbone = Hybrid()
    elif model_type == 'repvit':
        backbone = RepVitBackbone()
    elif model_type == 'mpvit':
        backbone = MpVitBackbone()
    else:
        raise ValueError(f"Do not find {model_type}")

    return backbone
