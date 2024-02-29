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
        #pdb.set_trace()
        
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
    

class MM_MIL_VisionTransformer(nn.Module):
    def __init__(self, img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), modality='multi_mil_add', num_block=24, wo_cls_token=False, **kwargs):
        super(MM_MIL_VisionTransformer, self).__init__()
        
        self.num_classes = kwargs['num_classes']
        
        self.modality = modality
        self.embed_dim = embed_dim
        self.wo_cls_token =wo_cls_token
        self.loss_func = ""
        
        if self.modality[:14] == 'multi_mil_head': #拼接OCT和彩照特征进入mm-mil
            if 'base' in self.modality:
                self.embed_dim = 768
                self.OCT_ViT = VisionTransformer(
                    img_size=img_size,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            else:
                self.OCT_ViT = VisionTransformer(
                    img_size=img_size,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
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
        
        elif 'multi_base_multi_head_baseline' == self.modality:
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.
            Self_attention_depth=3
            Co_attention_depth=3
            self.embed_dim = 768
            self.loss_func = 'weighted_loss_mm_output'
            #self.loss_func = 'mean'
            
            self.CFP_head = nn.Linear(self.embed_dim, self.num_classes)
            self.OCT_head = nn.Linear(self.embed_dim, self.num_classes)
            self.MM_head = nn.Linear(self.embed_dim, self.num_classes)
            
            self.OCT_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            haf_sa_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, Self_attention_depth)]  # SA stochastic depth decay rule
        
            self.Self_attn_blocks_fundus = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
                for i in range(Self_attention_depth)])
            
            self.Self_attn_blocks_OCT = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=haf_sa_dpr[i], norm_layer=norm_layer)
                for i in range(Self_attention_depth)])
            
            self.fundus_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
            self.oct_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

            self.num_mil = 4
            self.mils = nn.ModuleList([MIL2(self.embed_dim) for _ in range(self.num_mil)])
            
            init_weight(self.fundus_proj)
            init_weight(self.oct_proj)
            for mil in self.mils:
                init_weight(mil)
        
        elif 'multi_base_multi_head_wo_SA' == self.modality:
            drop_rate=0.
            attn_drop_rate=0.
            drop_path_rate=0.
            Self_attention_depth=3
            Co_attention_depth=3
            self.embed_dim = 768
            self.loss_func = 'weighted_loss_mm_output'
            #self.loss_func = 'mean'
            
            self.CFP_head = nn.Linear(self.embed_dim, self.num_classes)
            self.OCT_head = nn.Linear(self.embed_dim, self.num_classes)
            self.MM_head = nn.Linear(self.embed_dim, self.num_classes)
            
            self.OCT_ViT = VisionTransformer(
                img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
            
            self.fundus_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
            self.oct_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

            self.num_mil = 4
            self.mils = nn.ModuleList([MIL2(self.embed_dim) for _ in range(self.num_mil)])
            
            init_weight(self.fundus_proj)
            init_weight(self.oct_proj)
            for mil in self.mils:
                init_weight(mil)        
        
        if 'base' in self.modality:
            self.embed_dim = 768
            self.fundus_ViT = VisionTransformer(
                img_size=img_size,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
        else:
            self.fundus_ViT = VisionTransformer(
                img_size=img_size,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
        
        #self.fundus_ViT = VisionTransformer(
        #    img_size=img_size,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        #    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_block=num_block, **kwargs)
       
        
        
    def forward(self, x):
        #输入分别为(B,C,H,W)和(B,N,C,H,W)
        fundus_img, OCT_img = x[0],x[1]
        
        B = OCT_img.size(0)
        N = OCT_img.size(1)
        
        if self.modality[:14] == 'multi_mil_head': #拼接OCT和彩照特征进入mm-mil
            #pdb.set_trace()
            
            N1 = 1
            if fundus_img.dim() > 4: #使用oversample,此时输入应该为(B,N=6or12,C,H,W)
                N1 = fundus_img.shape[1]
                fundus_img = fundus_img.reshape(-1, fundus_img.size()[2], fundus_img.size()[3], fundus_img.size()[4])
            #fundus_feat = (B or B*N,embed_dim)
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
            
            return feat
        
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
        
        elif self.modality == 'multi_base_multi_head_baseline':
            N1 = fundus_img.shape[1]
            N2 = OCT_img.shape[1]
            
            #输入应该为(B,N=12or6,C,H,W),output = (B*N,C,H,W)
            fundus_img = fundus_img.reshape(-1, fundus_img.size()[2], fundus_img.size()[3], fundus_img.size()[4])
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
            
            #pdb.set_trace()
            
            #fundus_feat = (B*N1,embed_dim)
            #OCT_feat = (B*N2,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            #fundus_feat = (B,N1,embed_dim)
            fundus_feat = fundus_feat.reshape(B,N1,fundus_feat.shape[1])
            #OCT_feat = (B,N2,embed_dim)
            OCT_feat = OCT_feat.reshape(B,N2,OCT_feat.shape[1])
            
            #做case级别的SA，之前是图像级别的
            for blk in self.Self_attn_blocks_fundus:
                fundus_feat = blk(fundus_feat, register_hook=False)
            for blk in self.Self_attn_blocks_OCT:
                OCT_feat = blk(OCT_feat, register_hook=False)
            
            #模态映射，映射到同一空间
            #fundus_feat = (B,N1,embed_dim)
            #OCT_feat = (B,N2,embed_dim)
            fundus_feat = self.fundus_proj(fundus_feat)
            OCT_feat = self.oct_proj(OCT_feat)
            
            #mil_input = (B, N1+N2, embed_dim)
            mil_input = torch.cat((fundus_feat, OCT_feat), dim = 1)
            
            feat = []
            fundus_feat = []
            OCT_feat = []
            for mil in self.mils:
                #(B, N1+N2, embed_dim)
                output = mil(mil_input)
                feat.append(torch.sum(output, dim=1))
                fundus_feat.append(torch.sum(output[:,:N1,:], dim=1))
                OCT_feat.append(torch.sum(output[:,N1:,:], dim=1))
            
            #feat = (B,4,embed_dim)
            #fundus_feat = (B,4,embed_dim)
            #OCT_feat = (B,4,embed_dim)
            feat = torch.stack(feat, dim = 1)
            fundus_feat = torch.stack(fundus_feat, dim = 1)
            OCT_feat = torch.stack(OCT_feat, dim = 1)
            
            #mean后fundus、OCT和MM feat都只剩下二维了, (B,embed_dim)
            feat = torch.mean(feat, dim=1)
            fundus_feat = torch.mean(fundus_feat, dim=1)
            OCT_feat = torch.mean(OCT_feat, dim=1)
            
            #feat都变成(B,7)
            feat = self.MM_head(feat)
            fundus_feat = self.CFP_head(fundus_feat)
            OCT_feat = self.OCT_head(OCT_feat)
            
            return feat,fundus_feat,OCT_feat
        
        elif self.modality == 'multi_base_multi_head_wo_SA':
            N1 = fundus_img.shape[1]
            N2 = OCT_img.shape[1]
            
            #输入应该为(B,N=12or6,C,H,W),output = (B*N,C,H,W)
            fundus_img = fundus_img.reshape(-1, fundus_img.size()[2], fundus_img.size()[3], fundus_img.size()[4])
            OCT_img = OCT_img.reshape(-1, OCT_img.size()[2], OCT_img.size()[3], OCT_img.size()[4])
            
            #pdb.set_trace()
            
            #fundus_feat = (B*N1,embed_dim)
            #OCT_feat = (B*N2,embed_dim)
            fundus_feat = self.fundus_ViT.forward_features(fundus_img)            
            OCT_feat = self.OCT_ViT.forward_features(OCT_img)
            
            #fundus_feat = (B,N1,embed_dim)
            fundus_feat = fundus_feat.reshape(B,N1,fundus_feat.shape[1])
            #OCT_feat = (B,N2,embed_dim)
            OCT_feat = OCT_feat.reshape(B,N2,OCT_feat.shape[1])
            
            #模态映射，映射到同一空间
            #fundus_feat = (B,N1,embed_dim)
            #OCT_feat = (B,N2,embed_dim)
            fundus_feat = self.fundus_proj(fundus_feat)
            OCT_feat = self.oct_proj(OCT_feat)
            
            #mil_input = (B, N1+N2, embed_dim)
            mil_input = torch.cat((fundus_feat, OCT_feat), dim = 1)
            
            feat = []
            fundus_feat = []
            OCT_feat = []
            for mil in self.mils:
                #(B, N1+N2, embed_dim)
                output = mil(mil_input)
                feat.append(torch.sum(output, dim=1))
                fundus_feat.append(torch.sum(output[:,:N1,:], dim=1))
                OCT_feat.append(torch.sum(output[:,N1:,:], dim=1))
            
            #feat = (B,4,embed_dim)
            #fundus_feat = (B,4,embed_dim)
            #OCT_feat = (B,4,embed_dim)
            feat = torch.stack(feat, dim = 1)
            fundus_feat = torch.stack(fundus_feat, dim = 1)
            OCT_feat = torch.stack(OCT_feat, dim = 1)
            
            #mean后fundus、OCT和MM feat都只剩下二维了, (B,embed_dim)
            feat = torch.mean(feat, dim=1)
            fundus_feat = torch.mean(fundus_feat, dim=1)
            OCT_feat = torch.mean(OCT_feat, dim=1)
            
            #feat都变成(B,7)
            feat = self.MM_head(feat)
            fundus_feat = self.CFP_head(fundus_feat)
            OCT_feat = self.OCT_head(OCT_feat)
            
            return feat,fundus_feat,OCT_feat
        
        return feat
    
    
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #model = VisionTransformer(
    #    img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MM_MIL_vit_large_patch16(modality='multi_mil_add', num_block=24, wo_cls_token=False, **kwargs):
    model = MM_MIL_VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), modality=modality, num_block=num_block, wo_cls_token=wo_cls_token, **kwargs)
    #model = MM_MIL_VisionTransformer(
    #    img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    #    norm_layer=partial(nn.LayerNorm, eps=1e-6), modality=modality, num_block=num_block, wo_cls_token=wo_cls_token, **kwargs)
    
    return model
