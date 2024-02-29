"""Vision Transformer from ross wightman(https://github.com/rwightman/pytorch-image-models)
modify PatchEmbed for oct multi image.

TODO:
* Need a neat modification
"""

""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial
from itertools import repeat
# from torch._six import container_abcs

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs,int_classes
else:
    import collections.abc as container_abcs
    int_classes = int

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        # x shape: [Batch_size, Num_patches, embed_dim]
        B, N, C = x.shape

        # Linear2threedim
        # qkv shape: [B, N, dim] -> [B, N, 3*dim] -> [B, N, 3, num_head, dim/num_head] ->
        #            [3, B, num_head, N, dim/num_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 对应着3个QKV，
        # q, k, v: [B, num_head, N, dim/num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Query @ Key -> Attn * scale -> attn
        # scale的作用是出现过大的值导致，softmax后的值太小了
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 经过一个softmax获得注意力系数
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attn和V做矩阵乘法得到对应的结果
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        self.save_attention_map(attn)

        # 保存此时的attn 就是 Q和K的结果，把attention 求梯度的时候的对应值保存到对应的attn gradients
        if register_hook:
            # 此处的绑定过了save_attn_gradients，随后在对attn求梯度的时候，梯度值会作为参数传入到save_attn_gradients这个函数中。
            attn.register_hook(self.save_attn_gradients)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CoAttention(nn.Module):
    """
    Coattention, Specific calculation

    FORWARD:
        INPUT:
            x1, modality q,     [B, N1, D]
            x2, modality k&v,   [B, N2, D]
        RETURN:
            x1, modality q,     [B, N1, D]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x1, x2, register_hook=False):
        """
        :param x1: Q modality
        :param x2: K, V modality
        :param register_hook:
        :return: x1
        """

        B, N1, C1 = x1.shape
        B, N2, C2 = x2.shape

        # q shape: [B, N, dim] -> [B, N, 1, num_head, dim/num_head] ->
        #            [1, B, num_head, N, dim/num_head]
        q = x1.reshape(B, N1, 1, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        # kv shape: [B, N, dim] -> [B, N, 2*dim] -> [B, N, 2, num_head, dim/num_head] ->
        #            [2, B, num_head, N, dim/num_head]
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)

        # q [B, num_head, N1, dim / num_head]
        # v [B, num_head, N2, dim / num_head]
        q = q[0]
        k, v = kv[0], kv[1]

        # Query @ Key -> Attn * scale -> attn
        # scale的作用是出现过大的值导致，softmax后的值太小了
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 经过一个softmax获得注意力系数
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attn和V做矩阵乘法得到对应的结果
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C1)

        self.save_attention_map(attn)

        # 保存此时的attn 就是 Q和K的结果，把attention 求梯度的时候的对应值保存到对应的attn gradients
        if register_hook:
            # 此处的绑定过了save_attn_gradients，随后在对attn求梯度的时候，梯度值会作为参数传入到save_attn_gradients这个函数中。
            attn.register_hook(self.save_attn_gradients)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CoAttention_Block(nn.Module):
    """
    Coattention_Encoder_Block: Confirm the different two modality and using co-attention encoder block.
    Two modalities need to have independent LayerNorm, Feed Forward.
    It seems like that the changed is little

    Forward:
        INPUT:
            x1: modality 1  [B, N1, Dim1]
            x2: modality 2  [B, N2, Dim2]
            register_hook: visualize
        RETURN:
            x1: modality 1 [B, N1, Dim1]
            x2: modality 2 [B, N2, Dim1]
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm_1_1 = norm_layer(dim)
        self.co_attn_1 = CoAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_1_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_2_1 = norm_layer(dim)
        self.co_attn_2 = CoAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2, register_hook):
        x1 = self.norm_1_1(x1)
        x2 = self.norm_2_1(x2)

        v1 = x1 + self.drop_path_1(self.co_attn_1(x1, x2, register_hook=register_hook))
        v2 = x2 + self.drop_path_2(self.co_attn_2(x2, x1, register_hook=register_hook))

        x1 = v1 + self.drop_path_1(self.mlp_1(self.norm_1_2(v1)))
        x2 = v2 + self.drop_path_2(self.mlp_2(self.norm_2_2(v2)))

        return x1, x2


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
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


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    # 这样操作有利于完成dim的转换。不同的特征提取器得到不同的特征，得到的特征可以视为(B, channel, feature_size, feature_size)
    # 随后通过一个Conv转换到对应维度，(B, feature_dim, feature_size, feature_size)
    # 再取一个1*1的像素级patch，这其实确实，不能将feature map视为一个拼接的图，因为事实上他们不是。
    # 这样子才更合理，才是视觉。
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, cls_token_type='first'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.cls_token_type = cls_token_type

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, register_hook=False):
        B = x.shape[0]

        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, register_hook=register_hook)

        x = self.norm(x)
        if self.cls_token_type == 'first':
            return x[:, 0]
        elif self.cls_token_type == 'mean':
            return torch.mean(x, dim=1)

    def forward(self, x, register_hook=False):
        x = self.forward_features(x, register_hook=register_hook)
        x = self.head(x)
        return x
    

class VisionTransformer_Coattention(nn.Module):
    """ 
    MMRAF based on Vision Transformer
    ViT Default Config:
        - img_size, the size of input image: 224
        - patch_size, the size of patch: 16
        - in_channels, input channels: 3
        - num_classes, num_classes: 1000
        - embed_dim, the dimension of embeded token: 768
        - depth, the depth of feature extraction block: 3
        - num_heads, the number of head incide each attention block: 12
        - mlp_ratio: 4
        - qkv_bias: False
        - qk_scale: None
        - drop_rate: 0.
        - attn_drop_rate: 0.
        - drop_path_rate: 0.
        - hybrid_backbone: None
        - norm_layer: nn.LayerNorm
        - cls_token_type, cls token type: 'mean'
    MM_RAF Config:
        - haf_ca_depth, the depth of cross attention in HAF: 3
        - haf_sa_depth, the depth of merged attention in HAF: 3        
        - patches_num_m_1, the nums of tokens for the first modality: 196
        - need_cls_token, whether need the class token in the last MLP: True
        - sep_mean, whether choose the seperate mean strategy: False
        - trm_feature_extractor: DEPRECATED
        - BCA strategy: DEPRECATED
        - contrastive_token: DEPRECATED
        - MILR: DEPRECATED
    
    """
    

    def __init__(self, 
                 # ViT Default Config
                 img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, cls_token_type='mean',
                 # MM_RAF Config
                 haf_ca_depth=3, haf_sa_depth=3, patches_num_m_1=196, need_cls_token=False, sep_mean=False
                ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.cls_token_type = cls_token_type
        self.patch_num_m_1 = patches_num_m_1
        self.need_cls_token = need_cls_token
        self.sep_mean = sep_mean
        
        # Define the patch_embedding
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                               embed_dim=embed_dim)
        self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                               embed_dim=embed_dim)
        
        # get the num_patches of modal CFP
        num_patches = self.patch_embed1.num_patches
        
        # Define the CLS token for each modal
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Define the learnable positional encoding for each modal
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))        
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # ViT stochastic depth decay rule
        haf_sa_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, haf_sa_depth)]  # SA stochastic depth decay rule
        haf_ca_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, haf_ca_depth)]  # CA stochastic depth decay rule
        
        # Define the basic feature extractor for each modal
        self.SA_blocks_1 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.SA_blocks_2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        # Define the HAF module, namely the self-attention and the cross-attention blocks.
        self.haf_self_attn_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
            for i in range(haf_sa_depth)])

        self.haf_cross_attn_blocks = nn.ModuleList([
            CoAttention_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_ca_dpr[i], norm_layer=norm_layer)
            for i in range(haf_ca_depth)])

        self.norm = norm_layer(embed_dim)
        
        # 按照sep_mean和need_cls_token，调整进入最终MLP的输入通道
        linear_in = embed_dim * (1 + self.sep_mean + (self.sep_mean and self.need_cls_token))
        self.head = nn.Linear(linear_in, num_classes) if num_classes > 0 else nn.Identity()


        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x1, x2, register_hook=False):
        B = x1.shape[0]
        N = x1.shape[1] + x2.shape[1]

        N1 = 196
        N2 = 196 * (N - 1)
        # x: [batch_num, pic_pair, channel, h, w]
        # example: [1, 12, 3, 224, 224]
        
        # For m1:
        x1 = self.patch_embed1(x1)
        cls_token_1 = self.cls_token1.expand(B, -1, -1)
        x1 = torch.cat((cls_token_1, x1), dim=1)
        x1 = x1 + self.pos_embed1
        x1 = self.pos_drop1(x1)

        for blk in self.SA_blocks_1:
            x1 = blk(x1, register_hook=register_hook)
            
        # For m2:
        x2 = self.patch_embed2(x2)
        cls_token_2 = self.cls_token2.expand(B, -1, -1)
        x2 = torch.cat((cls_token_2, x2), dim=1)

        x2 = x2 + self.pos_embed2
        x2 = self.pos_drop1(x2)

        for blk in self.SA_blocks_2:
            x2 = blk(x2, register_hook=register_hook)
        
        # 去掉OCT模态的的CLS token，此时cat得到两个模态总的结构x
        x2 = x2[:, 1:, :]
        x = torch.cat([x1, x2], dim=1)

        # HAF-SA，首先经过一次SA，保留中间结果
        for blk in self.haf_self_attn_blocks:
            x = blk(x, register_hook=register_hook)
        mid_res = x
        
        # HAF-CA，拆开两个模态，经过一次CA
        x1, x2 = x[:, 1:N1 + 1, :], x[:, N1 + 1:, :]
        for blk in self.haf_cross_attn_blocks:
            x1, x2 = blk(x1, x2, register_hook=register_hook)
        
        # 将输出合起来
        x = torch.cat((x[:, 0, :].unsqueeze(1), x1, x2), dim=1)
        x = self.norm(x)

        if self.cls_token_type == 'first':
            # 当使用cls first作为分类依据的时候，只需要返回CLS TOKEN即可
            # 对于MM-RAF没有意义
            # example: [1, 768]
            return x[:, 0]
        elif self.cls_token_type == 'mean':
            # 使用整体的平均值作为分类语义结果，对其做平均
            # return shape: [batch_shape, embed_dim]
            
            if self.need_cls_token:
                # 如果需要把CFP的cls_token也单独拎出来处理
                if self.sep_mean:
                    # 如果针对两个模态的token单独取mean                    
                    cls_token_value = x[:, 0, :]
                    m1_mean = torch.mean(x[:, 1:N1 + 1, :], dim=1)
                    m2_mean = torch.mean(x[:, N1 + 1:, :], dim=1)
                    res = torch.cat([cls_token_value, m1_mean, m2_mean], dim=1)
                else:
                    # 整体取mean
                    res = torch.mean(x, dim=1)
                return res
            else:
                # 如果不需要CFP的CLS token
                if self.sep_mean:
                    # 如果针对两个模态的token单独取mean    
                    m1_mean = torch.mean(x[:, 1:N1 + 1, :], dim=1)
                    m2_mean = torch.mean(x[:, N1 + 1:, :], dim=1)
                    res = torch.cat([m1_mean, m2_mean], dim=1)
                else:
                    # 整体取mean
                    res = torch.mean(x[:, 1:, :], dim=1)
                return res


    def forward(self, x1, x2, register_hook=False):
        x = self.forward_features(x1, x2, register_hook=register_hook)
        x = self.head(x)
        return x



def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_224_coattention(pretrained=False, depth=12, **kwargs):
    model = VisionTransformer_Coattention(
        patch_size=16, embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
    return model


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

name2checkpoints = {
    # 'vit_base_patch16_224': '/datastore/users/you.zhou/pretrained_model/jx_vit_base_p16_224-80ecf9dd.pth'
    'vit_base_patch16_224': '/datastore/users/you.zhou@corp.vistel.cn/_userstore/data/pretrained_model/jx_vit_base_p16_224-80ecf9dd.pth'
}


class ModifyPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # print(B, N, C, H, W)
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.reshape(B * N, C, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        embed_dim = x.shape[2]
        x = x.reshape(B, -1, embed_dim)
        return x



def repeat_pos_embed_weight(pos_embed, repeat_num=12):
    b, c, f = pos_embed.shape
    cls_token_pos = pos_embed[:, 0, :].reshape(b, 1, f)
    need_repeat = pos_embed[:, 1:, :]
    repeated = need_repeat.repeat(1, repeat_num, 1)
    repeated = torch.cat((repeated, cls_token_pos), dim=1)
    return repeated



def load_vit_to_co_attn(pretrained_dict, model_dict):
    for k, v in model_dict.items():
        # Havenot been tested
        # print(k)
        if 'blocks' in k and 'Co' not in k:
            new_query = 'blocks' + k[k.find('.'):]
            # print("key: ", k, " MATCH --> new_query: ", new_query)
        elif 'cls_token' in k:
            new_query = 'cls_token'
        elif 'patch_embed' in k:
            new_query = 'patch_embed' + k[k.find('.'):]
        elif 'pos_embed' in k:
            # TODO: 不知道这里需不需要调整为这个k.kfind
            new_query = 'pos_embed'
        else:
            new_query = k

        # print(new_query)

        if new_query in pretrained_dict and v.size() == pretrained_dict[new_query].size():
            # print(f"Updating the new query.{new_query}")
            model_dict.update({k: pretrained_dict[new_query]})
            # if 'SA_blocks' in k:
            #     print(f"针对{k} 完成 {new_query}的更新")
            # print(f"针对{k} 完成 {new_query}的更新")
    return model_dict

def modify_transformer_coattention_all(
    name='vit_base_patch16_224',
    # for initialization
    image_num=6, 
    pretrained=True,                
    all_oct=False,
    
    # for model
    num_classes=2,     
    patch_size=16, 
    embed_dim=768, 
    depth=3, 
    num_heads=12, 
    mlp_ratio=4, 
    qkv_bias=True,    
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
    
):
    """
    modify the basic mmraf
    
    """
    # TODO: 注意这里的image num是12还是24需要考虑好
    if name == 'vit_base_patch16_224':
        model = vit_base_patch16_224_coattention(
            num_classes=2,     
            depth=3, 
        )
    else:
        assert "ERROR for not specify the model name"
        return

    default_img_considering_num = 32

    model.pos_embed1 = nn.Parameter(torch.zeros(1, 196 * 1 + 1, 768))
    model.pos_embed2 = nn.Parameter(torch.zeros(1, 196 * (image_num - 1) + 1, 768))

    model.patch_embed2 = ModifyPatchEmbed(in_chans=default_img_considering_num // (image_num - 1))


    if pretrained:
        pretrained_dict = torch.load(name2checkpoints[name])

        pretrained_dict['pos_embed1'] = repeat_pos_embed_weight(pretrained_dict['pos_embed'], repeat_num=1)
        pretrained_dict['pos_embed2'] = repeat_pos_embed_weight(pretrained_dict['pos_embed'], repeat_num=image_num - 1)

        model_dict = model.state_dict()

        model_dict = load_vit_to_co_attn(pretrained_dict, model_dict)

        model.load_state_dict(model_dict)

        print("ViT pretrained weight LOAD SUCCESSFULLY!")

    return model

