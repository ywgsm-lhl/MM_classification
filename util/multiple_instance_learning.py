import torch
import torch.nn as nn


class InstanceAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1, bias=False),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x

class InstanceAttention2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttention2, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1, bias=False),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        x = x*att   # (B,N,C)
        return x

class InstanceAttentionBias(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttentionBias, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class InstanceAttentionSigmoid(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttentionSigmoid, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class InstanceGatedAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceGatedAttention, self).__init__()
        self.attn_u = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
        )
        self.attn_v = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Sigmoid(),
        )
        self.attn = nn.Linear(channel // reduction, 1)
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att_u = self.attn_u(x.view(-1, C))  # (B*N, C/r)
        att_v = self.attn_v(x.view(-1, C))  # (B*N, C/r)
        att = self.attn(att_u*att_v)  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class MIL(nn.Module):

    def __init__(self, channel, num_classes=7, att_type='normal'):
        super(MIL, self).__init__()
        if att_type == 'normal':
            self.ins_attn = InstanceAttention(channel)
        if att_type == 'sigmoid':
            self.ins_attn = InstanceAttentionSigmoid(channel)
        if att_type == 'normnal_with_bias':
            self.ins_attn = InstanceAttentionBias(channel)
        elif att_type == 'gated':
            self.ins_attn = InstanceGatedAttention(channel)
        self.ins_atts = None
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        x = self.ins_attn(x, visualize)
        if visualize:
            self.ins_atts = x
        x = self.fc(x)
        return x


class MIL2(nn.Module):

    def __init__(self, channel, att_type='normal'):
        super(MIL2, self).__init__()
        if att_type == 'normal':
            self.ins_attn = InstanceAttention2(channel)
        if att_type == 'sigmoid':
            self.ins_attn = InstanceAttentionSigmoid(channel)
        if att_type == 'normnal_with_bias':
            self.ins_attn = InstanceAttentionBias(channel)
        elif att_type == 'gated':
            self.ins_attn = InstanceGatedAttention(channel)

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        x = self.ins_attn(x, visualize)
        return x
