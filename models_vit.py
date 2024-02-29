# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from util.multiple_instance_learning import MIL, MIL2
from util.initial import init_weight
from util.cross_attention import CoAttention_Block, Block

import copy
import pdb

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config=None):
        super(LocalFeatureTransformer, self).__init__()

#         self.config = config
        self.d_model = 768
        self.nhead = 8
        self.layer_names = ['self', 'cross'] * 4
        encoder_layer = LoFTREncoderLayer(self.d_model, self.nhead, 'full')
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
    

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, modality='fundus', num_block=24, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.num_block = num_block
        #self.layer_mil = MIL2(kwargs['embed_dim'])
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]   #x = (16,3,224,224)
        x = self.patch_embed(x) #output = (16,196(num_patches),1024(embed_dim)),是一个2D卷积
        
        #input = (1,1,1024), output = (16,1,1024)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #output = (16,197,1024)
        x = torch.cat((cls_tokens, x), dim=1)
        #self.pos_embed = (1,197,1024), output = (16,197,1024)
        x = x + self.pos_embed
        #output = (16,197,1024)
        x = self.pos_drop(x)
        
        #output = (16*12,197,1024)
        for i in range(self.num_block):
            x = self.blocks[i](x)
        '''
        xx = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i+1 in [12,16,20,24]:
                xx.append(x)  #不用cls_token
        #xx = (16,4,197,1024)
        x = torch.stack(xx, dim = 1)
        '''
        #output = (16,197,1024)
        #for blk in self.blocks:
        #    x = blk(x)
        
        if self.global_pool:
            #output = (16,1024)
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            #outcome = (16,1024)
            outcome = self.fc_norm(x)
        else:  #原代码是输出cls_token，为了复用zy代码，改成没有任何操作直接返回
            #x = self.norm(x)
            #outcome = x[:, 0]
            return x
            

        return outcome

    def forward(self, x):
        #output = (B,4,embed_dim)
        x = self.forward_features(x)
        #output = (B,embed_dim)
        #x = self.layer_mil(x)
        #output = (B,num_class=7)
        x = self.head(x)
        
        return x
    
class MIL_VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, wo_class=True, modality='OCT_mil', num_block=24, **kwargs):
        super(MIL_VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.num_block = num_block
        if wo_class:   #直接输出分类结果
            self.mil = MIL(kwargs['embed_dim'], kwargs['num_classes'])
        else:          #输出特征(B,embed_dim)
            self.mil = MIL2(kwargs['embed_dim'])
        self.layer_mil = MIL2(kwargs['embed_dim'])
        
        init_weight(self.mil)
        init_weight(self.layer_mil)
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        
        B = x.shape[0]   #x = (16*12=192,3,224,224)
        x = self.patch_embed(x) #output = (16*12,196(num_patches),1024(embed_dim))
        
        #input = (1,1,1024), output = (16*12,1,1024)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #output = (16*12,197,1024) 
        x = torch.cat((cls_tokens, x), dim=1)
        #self.pos_embed = (1,197,1024), output = (16*12,197,1024), 197=(224/16)*(224/16)+1
        x = x + self.pos_embed
        #output = (16*12,197,1024)
        x = self.pos_drop(x)
        
        #output = (16*12,197,1024)
        #for i in range(self.num_block):
        #    x = self.blocks[i](x)
        
        xx = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i+1 in [12,16,20,24]:
                xx.append(x)  #不用cls_token
        #xx = (16*12,4,197,1024)
        x = torch.stack(xx, dim = 1)
        
        #for blk in self.blocks:
        #    x = blk(x)
        
        if self.global_pool:
            #output = (16*12,1024)
            x = x[:, :, 1:, :].mean(dim=2)  # global pool without cls token
            #outcome = (16*12,1024)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def forward(self, x):
        B = x.size(0)
        N = x.size(1)
        
        #pdb.set_trace()
        
        #输入应该为(B,N=12,C,H,W),output = (B*N,C,H,W)
        x = x.reshape(-1, x.size()[2], x.size()[3], x.size()[4])
        #output = (B*N,4,embed_dim) or (B*N,embed_dim)
        x = self.forward_features(x)
        #output = (B*N,embed_dim)
        x = self.layer_mil(x)
        #output = (B,N,embed_dim)
        x = x.view(B, N, -1)
        #output = (B,num_class=7) or = (B,embed_dim)
        x = self.mil(x)
        
        return x
    
class MM_VisionTransformer(nn.Module):
    def __init__(self, img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), modality='multi_add', **kwargs):
        super(MM_VisionTransformer, self).__init__()
        
        self.num_classes = kwargs['num_classes']
        
        self.fuse = modality[-3:]
        self.embed_dim = embed_dim
        
        if self.fuse == 'cat':   #拼接是特征拼接
            self.head = nn.Linear(self.embed_dim*2, self.num_classes) 
            kwargs['num_classes'] = embed_dim
            #self.fc_norm = norm_layer(embed_dim*2)
            self.bn = nn.BatchNorm1d(1, affine=False)
        elif self.fuse == 'add': #相加是结果相加
            #self.fc_norm = norm_layer(self.embed_dim)
            #self.head = nn.Linear(self.embed_dim, self.num_classes) 
            self.bn1 = nn.BatchNorm1d(1, affine=False)
            self.bn2 = nn.BatchNorm1d(1, affine=False)
            self.weights = torch.ones(self.num_classes) * 0.5
        
        self.fundus_ViT = VisionTransformer(
            img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        self.OCT_ViT = VisionTransformer(
            img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
    def forward(self, x):
        fundus_img, OCT_img = x[0],x[1]
        
        OCT_feat = self.OCT_ViT.forward_features(OCT_img)
        fundus_feat = self.fundus_ViT.forward_features(fundus_img)
        
        if self.fuse == 'cat':
            feat = torch.cat((OCT_feat, fundus_feat), dim = -1)
            #feat = self.fc_norm(feat)
            feat = self.head(feat)
            feat = self.bn(feat.unsqueeze(1)).squeeze(1)
            
        elif self.fuse == 'add':   
            '''
            add相当于直接把两个分类结果相加，不需要再经过一层线性层
            '''
            OCT_feat = self.OCT_ViT.head(OCT_feat)
            fundus_feat = self.fundus_ViT.head(fundus_feat)
            
            OCT_feat = self.bn1(OCT_feat.unsqueeze(1)).squeeze(1)
            fundus_feat = self.bn2(fundus_feat.unsqueeze(1)).squeeze(1)
            
            weights = self.weights.expand(OCT_feat.size()).cuda()
            
            feat = OCT_feat*weights + fundus_feat*weights
        
        return feat
    

class MM_MIL_VisionTransformer(nn.Module):
    def __init__(self, img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), modality='multi_mil_add', num_block=24, wo_cls_token=False, **kwargs):
        super(MM_MIL_VisionTransformer, self).__init__()
        
        self.num_classes = kwargs['num_classes']
        
        self.modality = modality
        self.embed_dim = embed_dim
        self.wo_cls_token =wo_cls_token
        
        if self.modality == 'multi_mil_cat':   #拼接是特征拼接
            self.head = nn.Linear(self.embed_dim*2, self.num_classes) 
            self.bn = nn.BatchNorm1d(1, affine=False)
            
            self.OCT_ViT = MIL_VisionTransformer(
                img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), wo_class=False, **kwargs)
        elif 'multi_mil_add' in self.modality: #相加是结果相加
            self.bn1 = nn.BatchNorm1d(1, affine=False)
            self.bn2 = nn.BatchNorm1d(1, affine=False)
            self.weights = torch.ones(self.num_classes) * 0.5
            
            if 'base' in self.modality:
                self.embed_dim = 768
                self.OCT_ViT = MIL_VisionTransformer(
                    img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            else:
                self.OCT_ViT = MIL_VisionTransformer(
                    img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
                
            
        elif self.modality[:14] == 'multi_mil_head': #拼接OCT和彩照特征进入mm-mil
            if 'base' in self.modality:
                self.embed_dim = 768
                self.OCT_ViT = VisionTransformer(
                    img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            else:
                self.OCT_ViT = VisionTransformer(
                    img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            self.fundus_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
            self.oct_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

            self.num_mil = int(self.modality[14])
            self.mils = nn.ModuleList([MIL(self.embed_dim, self.num_classes) for _ in range(self.num_mil)])
            
            init_weight(self.fundus_proj)
            init_weight(self.oct_proj)
            for mil in self.mils:
                init_weight(mil)
        elif 'multi_base_SA_CA' == self.modality:
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.
            Self_attention_depth=3
            Co_attention_depth=3
            self.embed_dim = 768
            
            if self.wo_cls_token:
                self.head = nn.Linear(self.embed_dim*2, self.num_classes) 
            else:
                self.head = nn.Linear(self.embed_dim*3, self.num_classes) 
            
            self.OCT_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            haf_sa_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Self_attention_depth)]  # SA stochastic depth decay rule
            haf_ca_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Co_attention_depth)]  # CA stochastic depth decay rule
        
            self.Self_attn_blocks = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
                for i in range(Self_attention_depth)])
            
            self.Co_attn_blocks = nn.ModuleList([
                CoAttention_Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_ca_dpr[i], norm_layer=norm_layer)
                for i in range(Co_attention_depth)])
        
        elif 'multi_base_3SA_3CA' == self.modality:
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.
            Self_attention_depth=3
            Co_attention_depth=3
            self.embed_dim = 768
            
            self.head = nn.Linear(self.embed_dim*2, self.num_classes) 
            
            self.OCT_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            haf_sa_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Self_attention_depth)]  # SA stochastic depth decay rule
            haf_ca_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Co_attention_depth)]  # CA stochastic depth decay rule
        
            self.Self_attn_blocks_OCT = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
                for i in range(Self_attention_depth)])
            
            self.Self_attn_blocks_fundus = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
                for i in range(Self_attention_depth)])
            
            self.Co_attn_blocks = nn.ModuleList([
                CoAttention_Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_ca_dpr[i], norm_layer=norm_layer)
                for i in range(Co_attention_depth)])
            
            self.ljz_model = LocalFeatureTransformer()
        
        elif 'multi_base_CA' == self.modality:
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.
            Co_attention_depth=3
            self.embed_dim = 768
            
            if self.wo_cls_token:
                self.head = nn.Linear(self.embed_dim*2, self.num_classes) 
            else:
                self.head = nn.Linear(self.embed_dim*3, self.num_classes) 
            
            self.OCT_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            haf_ca_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Co_attention_depth)]  # CA stochastic depth decay rule
        
            self.Co_attn_blocks = nn.ModuleList([
                CoAttention_Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_ca_dpr[i], norm_layer=norm_layer)
                for i in range(Co_attention_depth)])     
            
        if 'base' in self.modality:
            self.fundus_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
        else:
            self.fundus_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        
        
    def forward(self, x):
        #输入分别为(B,C,H,W)和(B,N,C,H,W)
        fundus_img, OCT_img = x[0],x[1]
        
        B = OCT_img.size(0)
        N = OCT_img.size(1)
        
        if self.modality == 'multi_mil_cat':
            #fundus_feat = (B,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)
        
            #OCT_feat(cat) = (B,embed_dim), OCT_feat(add) = (B,num_class=7)
            OCT_feat = self.OCT_ViT(OCT_img)
        
            #feat = (B,embed_dim*2)
            feat = torch.cat((OCT_feat, fundus_feat), dim = -1)
            #feat = (B,num_class)
            feat = self.head(feat)
            feat = self.bn(feat.unsqueeze(1)).squeeze(1)
            
        elif 'multi_mil_add' in self.modality:   
            '''
            add相当于直接把两个分类结果相加，不需要再经过一层线性层
            '''
            #fundus_feat = (B,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)

            #OCT_feat(cat) = (B,embed_dim), OCT_feat(add) = (B,num_class=7)
            OCT_feat = self.OCT_ViT(OCT_img)
        
            #fundus_feat = (B,num_class)
            fundus_feat = self.fundus_ViT.head(fundus_feat)
            
            OCT_feat = self.bn1(OCT_feat.unsqueeze(1)).squeeze(1)
            fundus_feat = self.bn2(fundus_feat.unsqueeze(1)).squeeze(1)
            weights = self.weights.expand(OCT_feat.size()).cuda()
            
            #feat = (B,num_class)
            feat = OCT_feat*weights + fundus_feat*weights
        
        elif self.modality[:14] == 'multi_mil_head': #拼接OCT和彩照特征进入mm-mil
            #pdb.set_trace()
            
            N1 = 1
            if fundus_img.dim() > 4: #使用oversample,此时输入应该为(B,N=6or12,C,H,W)
                N1 = fundus_img.shape[1]
                fundus_img = fundus_img.reshape(-1, fundus_img.size()[2], fundus_img.size()[3], fundus_img.size()[4])
            #fundus_feat = (B,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            
            #输入应该为(B,N=12,C,H,W),output = (B*N,C,H,W)
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
            #输出为(B*N, embed_dim)
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            OCT_feat = self.oct_proj(OCT_feat)
            fundus_feat = self.fundus_proj(fundus_feat)
            
            OCT_feat = OCT_feat.view(B, N, -1)
            fundus_feat = fundus_feat.view(B, N1, -1)
            mil_input = torch.cat((OCT_feat, fundus_feat), dim = 1)
            
            feat = []
            for mil in self.mils:
                feat.append(mil(mil_input))

            feat = torch.stack(feat, dim = 1)
            feat = torch.mean(feat, dim = 1)
        
        elif self.modality == 'multi_base_SA_CA':
            #输入应该为(B,N=12,C,H,W),output = (B*N,C,H,W)
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
        
            #fundus_feat = (B,197,embed_dim)
            #OCT_feat = (B*N,197,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            N1 = fundus_feat.shape[1]
            N2 = fundus_feat.shape[2]
            
            #OCT_feat = (B,N,197,embed_dim)
            OCT_feat = OCT_feat.reshape(B,N,fundus_feat.shape[1],fundus_feat.shape[2])
            #OCT_cls_token = (B,embed_dim)
            OCT_cls_token = torch.mean(OCT_feat[:,:,0,:], dim=1)
            #OCT_feat = (B,N*196,embed_dim)
            OCT_feat = OCT_feat[:,:,1:,:].reshape(B, N*(N1-1), N2)
            
            #concat后进入SA的cls_token是两个模态cls_token的均值
            #fundus_feat = [batch_size, 196, embed_dim]
            #OCT_feat = [batch_size, 12*196, embed_dim]
            #cls_token = [batch_size, embed_dim]
            cls_token = (OCT_cls_token+fundus_feat[:,0,:])/2
            #feat = [B,1+13*196,embed_dim]
            feat = torch.cat((cls_token.unsqueeze(1), fundus_feat[:, 1:, :], OCT_feat), dim=1)
            
            for blk in self.Self_attn_blocks:
                feat = blk(feat, register_hook=False)
            
            fundus_feat, OCT_feat = feat[:, 1:N1, :], feat[:, N1:, :]
            
            for blk in self.Co_attn_blocks:
                fundus_feat, OCT_feat = blk(fundus_feat, OCT_feat, register_hook=False)
            
            #mean后fundus和OCT feat都只剩下二维了, [batch_size, embed_dim]
            fundus_feat = torch.mean(fundus_feat, dim=1)
            OCT_feat = torch.mean(OCT_feat, dim=1)
            
            #feat = [batch_size, embed_dim*3]
            if self.wo_cls_token:
                feat = torch.cat((OCT_feat, fundus_feat), dim=1)
            else:
                feat = torch.cat((feat[:, 0, :], OCT_feat, fundus_feat), dim=1)
            feat = self.head(feat)
        
        elif self.modality == 'multi_base_3SA_3CA':
            #pdb.set_trace()
            #输入应该为(B,N=12,C,H,W),output = (B*N,C,H,W)
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
        
            #fundus_feat = (B,197,embed_dim)
            #OCT_feat = (B*N,197,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            N1 = fundus_feat.shape[1]
            N2 = fundus_feat.shape[2]
            
            #OCT_feat = (B,N*197,embed_dim)
            OCT_feat = OCT_feat[:, 1:, :].reshape(B, N*(N1-1), N2)
            
            #fundus_feat = [batch_size, 196, embed_dim]
            #OCT_feat = [batch_size, 12*196, embed_dim]
            fundus_feat = fundus_feat[:, 1:, :]
            #OCT_feat = OCT_feat[:, 1:, :]
            #pdb.set_trace()
            
            fundus_feat, OCT_feat = self.ljz_model(fundus_feat/15, OCT_feat/15)
#             for i in range(len(self.Co_attn_blocks)):
#                 OCT_feat = self.Self_attn_blocks_OCT[i](OCT_feat, register_hook=False)
#                 fundus_feat = self.Self_attn_blocks_fundus[i](fundus_feat, register_hook=False)
                
#                 #OCT_feat = self.Self_attn_blocks_OCT[2*i+1](OCT_feat, register_hook=False)
#                 #fundus_feat = self.Self_attn_blocks_fundus[2*i+1](fundus_feat, register_hook=False)
            
#                 fundus_feat, OCT_feat = self.Co_attn_blocks[i](fundus_feat, OCT_feat, register_hook=False)
            
            #mean后fundus和OCT feat都只剩下二维了, [batch_size, embed_dim]，去除cls_token
        
#             fundus_feat = torch.mean(fundus_feat[:, 1:, :], dim=1)
#             OCT_feat = torch.mean(OCT_feat[:, 1:, :], dim=1)
            fundus_feat = torch.mean(fundus_feat, dim=1)
            OCT_feat = torch.mean(OCT_feat, dim=1)
            
            #feat = [batch_size, embed_dim*2]
            feat = torch.cat((OCT_feat, fundus_feat), dim=1)
            feat = self.head(feat)
            
        
        elif self.modality == 'multi_base_CA':
            #输入应该为(B,N=12,C,H,W),output = (B*N,C,H,W)
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
        
            #fundus_feat = (B,197,embed_dim)
            #OCT_feat = (B*N,197,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            N1 = fundus_feat.shape[1]
            N2 = fundus_feat.shape[2]
            
            #OCT_feat = (B,N,197,embed_dim)
            OCT_feat = OCT_feat.reshape(B,N,fundus_feat.shape[1],fundus_feat.shape[2])
            #OCT_cls_token = (B,embed_dim)
            OCT_cls_token = torch.mean(OCT_feat[:,:,0,:], dim=1)
            
            #最后分类concat的cls_token是两个模态cls_token的均值
            #cls_token = (B,embed_dim)
            cls_token = (OCT_cls_token+fundus_feat[:,0,:])/2
            #fundus_feat = [batch_size, 196, embed_dim]
            fundus_feat = fundus_feat[:, 1:, :]
            #OCT_feat = (B,N*196,embed_dim)
            OCT_feat = OCT_feat[:,:,1:,:].reshape(B, N*(N1-1), N2)
            
            for blk in self.Co_attn_blocks:
                fundus_feat, OCT_feat = blk(fundus_feat, OCT_feat, register_hook=False)
            
            #mean后fundus和OCT feat都只剩下二维了, [batch_size, embed_dim]
            fundus_feat = torch.mean(fundus_feat, dim=1)
            OCT_feat = torch.mean(OCT_feat, dim=1)
            
            #feat = [batch_size, embed_dim*3]
            feat = torch.cat((cls_token, OCT_feat, fundus_feat), dim=1)
            feat = self.head(feat)
            
        return feat
    
    
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #model = VisionTransformer(
    #    img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MIL_vit_large_patch16(num_block=24, **kwargs):
    model = MIL_VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
    #model = MIL_VisionTransformer(
    #    img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MM_vit_large_patch16(modality='multi_add', **kwargs):
    model = MM_VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), modality=modality, **kwargs)
    return model

def MM_MIL_vit_large_patch16(modality='multi_mil_add', num_block=24, wo_cls_token=False, **kwargs):
    model = MM_MIL_VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), modality=modality, num_block=num_block, wo_cls_token=wo_cls_token, **kwargs)
    return model
