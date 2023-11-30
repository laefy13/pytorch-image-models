
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import create_conv2d, create_classifier, get_norm_act_layer, GroupNormAct
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from ._features import FeatureInfo, FeatureHooks
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['EfficientNet', 'EfficientNetFeatures']


class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(
            self,
            block_args,
            num_classes=1000,
            num_features=1280,
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg',
            **kwargs
    ):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
        )
        # print('block args',block_args)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_act_layer(self.num_features, inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head|bn2', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_effnet(variant, pretrained=False, **kwargs):
    features_mode = ''
    model_cls = EfficientNet
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        if 'feature_cfg' in kwargs:
            features_mode = 'cfg'
        else:
            kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
            model_cls = EfficientNetFeatures
            features_mode = 'cls'

    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        features_only=features_mode == 'cfg',
        pretrained_strict=features_mode != 'cls',
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    if features_mode == 'cls':
        model.pretrained_cfg = model.default_cfg = pretrained_cfg_for_features(model.pretrained_cfg)
    return model


def _gen_efficientnet(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, channel_divisor=8,
        group_size=None, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    # BREH CHANGE THE FUCKING C FOR THE CHANNELS SO THIS SHIT CAN ADJUST ACCORDING  TO THE MULTIPLIER
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, group_size=group_size),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model



@register_model
def efficientnet_b0(pretrained=False, **kwargs) -> EfficientNet:
    chann_mult = kwargs.get('chann_mult', 1.0) 
    dep_mult = kwargs.get('dep_mult', 1.0)
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=chann_mult, depth_multiplier=dep_mult, pretrained=pretrained, **kwargs)
    return model



register_model_deprecations(__name__, {
    'tf_efficientnet_b0_ap': 'tf_efficientnet_b0.ap_in1k',
    'tf_efficientnet_b1_ap': 'tf_efficientnet_b1.ap_in1k',
    'tf_efficientnet_b2_ap': 'tf_efficientnet_b2.ap_in1k',
    'tf_efficientnet_b3_ap': 'tf_efficientnet_b3.ap_in1k',
    'tf_efficientnet_b4_ap': 'tf_efficientnet_b4.ap_in1k',
    'tf_efficientnet_b5_ap': 'tf_efficientnet_b5.ap_in1k',
    'tf_efficientnet_b6_ap': 'tf_efficientnet_b6.ap_in1k',
    'tf_efficientnet_b7_ap': 'tf_efficientnet_b7.ap_in1k',
    'tf_efficientnet_b8_ap': 'tf_efficientnet_b8.ap_in1k',
    'tf_efficientnet_b0_ns': 'tf_efficientnet_b0.ns_jft_in1k',
    'tf_efficientnet_b1_ns': 'tf_efficientnet_b1.ns_jft_in1k',
    'tf_efficientnet_b2_ns': 'tf_efficientnet_b2.ns_jft_in1k',
    'tf_efficientnet_b3_ns': 'tf_efficientnet_b3.ns_jft_in1k',
    'tf_efficientnet_b4_ns': 'tf_efficientnet_b4.ns_jft_in1k',
    'tf_efficientnet_b5_ns': 'tf_efficientnet_b5.ns_jft_in1k',
    'tf_efficientnet_b6_ns': 'tf_efficientnet_b6.ns_jft_in1k',
    'tf_efficientnet_b7_ns': 'tf_efficientnet_b7.ns_jft_in1k',
    'tf_efficientnet_l2_ns_475': 'tf_efficientnet_l2.ns_jft_in1k_475',
    'tf_efficientnet_l2_ns': 'tf_efficientnet_l2.ns_jft_in1k',
    'tf_efficientnetv2_s_in21ft1k': 'tf_efficientnetv2_s.in21k_ft_in1k',
    'tf_efficientnetv2_m_in21ft1k': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'tf_efficientnetv2_l_in21ft1k': 'tf_efficientnetv2_l.in21k_ft_in1k',
    'tf_efficientnetv2_xl_in21ft1k': 'tf_efficientnetv2_xl.in21k_ft_in1k',
    'tf_efficientnetv2_s_in21k': 'tf_efficientnetv2_s.in21k',
    'tf_efficientnetv2_m_in21k': 'tf_efficientnetv2_m.in21k',
    'tf_efficientnetv2_l_in21k': 'tf_efficientnetv2_l.in21k',
    'tf_efficientnetv2_xl_in21k': 'tf_efficientnetv2_xl.in21k',
})
