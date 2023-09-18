import math
import torch
import numpy as np
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
from caps_vit import PreNorm, ConvGRU
from Utils import AverageMeter, accuracy
from einops.layers.torch import Rearrange
from caps_vit import ConvGRU, conv3x3_bn_relu


# constants
TOKEN_ATTEND_SELF_VALUE = -5e-4


class eca_layer(nn.Module):
    """
    Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map, 输入特征图的通道数
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        # x: input features with shape [b, c, d, h, w]
        b, c, h, w, t = x.size()
        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y= self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
    
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, 
                 use_1X1conv=False, stride=1, start_block=False, end_block=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.ReLU())
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, stride=stride)
        if use_1X1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)
        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)
        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)
        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block
    def forward(self, X):
        identity = X
        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.start_block:
            out = self.bn2(out)  
        out = self.ecalayer(out)
        out += identity
        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out) 
        return out
    
    
class S3KAIResNet(nn.Module):
    def __init__(self, band, num_classes, reduction=2, PARAM_KERNEL_SIZE=24):
        super(S3KAIResNet, self).__init__()
        self.conv1X1 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, kernel_size=(1, 1, 7), 
                                 stride=(1, 1, 2), padding=0)
        self.conv3X3 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, kernel_size=(3, 3, 7), 
                                 stride=(1, 1, 2), padding=(1, 1, 0))
        self.conv5X5 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, kernel_size=(5, 5, 7), 
                                 stride=(1, 1, 2), padding=(2, 2, 0))
        
        self.batch_norm1X1 = nn.Sequential(
            nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.batch_norm3X3 = nn.Sequential(
            nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        self.batch_norm5X5 = nn.Sequential(
            nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(band // reduction, PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
        self.res_net1 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0), end_block=True)
        
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        
        self.conv2 = nn.Conv3d(in_channels=PARAM_KERNEL_SIZE, out_channels=128, padding=(0, 0, 0), 
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True), 
                                         nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=PARAM_KERNEL_SIZE, padding=(0, 0, 0), 
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(nn.BatchNorm3d(PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1, 
                                                        affine=True), 
                                         nn.ReLU(inplace=True))
        
        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(nn.Linear(PARAM_KERNEL_SIZE, num_classes))
        
    def forward(self, X):
        x_1x1 = self.conv1X1(X)
        x_1x1 = self.batch_norm1X1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3X3(X)
        x_3x3 = self.batch_norm3X3(x_3x3).unsqueeze(dim=1)
        x_5x5 = self.conv5X5(X)
        x_5x5 = self.batch_norm5X5(x_5x5).unsqueeze(dim=1)
        
        x1 = torch.cat([x_5x5, x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ], 
            dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)
        
        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))
        
        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        return self.full_connection(x4)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class GroupedFeedForward(nn.Module):
    def __init__(self, *, dim, groups, mult=4):  # groups: levels(top-down:4, bottom-up:5)
        super().__init__()
        total_dim = dim * groups  # groups * dim
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups=groups),
            nn.GELU(), 
            nn.Conv1d(total_dim * mult, total_dim, 1, groups=groups),
            nn.GELU(),
            Rearrange('b (l d) n -> b n l d', l=groups)
        )
    
    def forward(self, levels):
        return self.net(levels)


class ConsensusAttention(nn.Module):
    def __init__(self, num_patches_side, attend_self=True, local_consensus_radius=0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius
        
        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side), 
                torch.arange(num_patches_side)
            )).float()
            
            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)
            
    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim=-1), levels
        
        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)
        
        if not self.attend_self:
            self_mask = torch.eye(n, device=device, dtype=torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)
            
        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)
        
        attn = sim.softmax(dim=-1)
        out = einsum('b l i j, b j l d -> b i l d', attn, v)
        return out


class GLOM(nn.Module):
    def __init__(self, in_channels, *, dim=32, levels=5, patch_size=9, sub_patch_size=1, 
                 consensus_self=False, local_consensus_radius=0):
        super(GLOM, self).__init__()
        # bottom level - incoming image, tokenize and add position
        num_patches_side = (patch_size // sub_patch_size)  # 9
        num_patches = num_patches_side ** 2  # 81
        self.levels = levels
        
        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=sub_patch_size, p2=sub_patch_size), 
            nn.Linear(sub_patch_size ** 2 * in_channels, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.image_to_levels = nn.Sequential(
            nn.Conv2d(in_channels, levels*dim, 1), nn.GELU(),   # (batch, 5*32, 9, 9)
            Rearrange('b (l d) h w -> b (h w) l d', d=dim), nn.LayerNorm(dim)
        )
        
        self.pos_emb = nn.Embedding(num_patches, dim)  # (81, 32) 
        
        # bottom-up and top-down
        self.bottom_up = GroupedFeedForward(dim=dim, groups=levels)
        self.top_down = GroupedFeedForward(dim=dim, groups=levels-1)
        
        # consensus attention
        self.attention = ConsensusAttention(num_patches_side, attend_self=consensus_self, 
                                            local_consensus_radius=local_consensus_radius)
        
    def forward(self, img, iters=3, levels=None, return_all=False):
        # img: (batch, 48, 9, 9)
        b, device = img.shape[0], img.device
        # need to have twice the number of levels of iterations in order for information 
        # to propagate up and back down. can be overridden
        iters = default(iters, self.levels*2)
        
        tokens = self.image_to_tokens(img)  # (batch, 81, 32)
        n = tokens.shape[1]  # 81
        
        pos_embs = self.pos_emb(torch.arange(n, device=device))
        pos_embs = rearrange(pos_embs, 'n d -> () n () d')  # (1, 81, 1, 32)
        
        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')  # (batch, 81, 1, 32)
        
        if not exists(levels):
            levels = self.image_to_levels(img)
            
        hiddens = [levels]
        
        num_contributions = torch.empty(self.levels, device=device).fill_(4)
        # top level does not get a top-down contribution, so have to account for this 
        # when doing the weighted mean
        num_contributions[-1] = 3
        
        for _ in range(iters):
            # each iteration, attach original input at the most bottom level, to be bottomed-up
            levels_with_input = torch.cat((bottom_level, levels), dim=-2)  # (batch, 81, 6, 32)
            
            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])  # (batch, 81, 5, 32)
            
            # positional embeddings given to top-down networks
            top_down_out = self.top_down(levels_with_input[..., 2:, :] + pos_embs)   # (batch, 81, 4, 32)
            top_down_out = F.pad(top_down_out, (0, 0, 0, 1), value=0.)  # (batch, 81, 5, 32)
            
            consensus = self.attention(levels)  # (batch, 81, 5, 32)
            
            # hinton said to use the weighted mean of (1) bottom up (2) top down 
            # (3) previous level value {t - 1} (4) consensus value  # (batch, 81, 5, 32)
            levels_sum = torch.stack((levels, bottom_up_out, top_down_out, consensus)).sum(dim=0)  
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')
            
            levels = levels_mean  # set for next iteration
            hiddens.append(levels)
            
        if return_all:
            return torch.stack(hiddens)  # return (time step, batch, num columns, levels, dimension)
        
        return levels  # (batch, 81, 5, 32)

    
class CapsGLOM(nn.Module):
    def __init__(self, band, patch_size=9, preliminary_layer_channels=96, num_group_norm=8, 
                 steps=8, sub_patch_size=1, dim=32, kernel_size=3, num_classes=13):
        super(CapsGLOM, self).__init__()
        
        self.preliminary_layer = conv3x3_bn_relu(band, preliminary_layer_channels)
        
        cl_channel = preliminary_layer_channels / steps
        cl_channel = int(cl_channel)  # 12
        cl2_channel = int(cl_channel / 2)  # 6
        self.convgru = ConvGRU(cl_channel, hidden_channels=[cl_channel, cl2_channel], kernel_size=kernel_size)
        
        output_channel = cl2_channel * steps  # 48
        self.group_normalization = nn.GroupNorm(num_group_norm, output_channel)
        
        self.glom = GLOM(output_channel, dim=dim, patch_size=patch_size, sub_patch_size=sub_patch_size)
        
        num_patches_side = (patch_size // sub_patch_size)  # 9
        num_patches = num_patches_side ** 2  # 81
        
        self.to_latent = nn.Identity()
        self.conv1d = nn.Sequential(
            nn.Conv1d(num_patches, 1, 1), 
            nn.GELU()
        )
        self.mlp_head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x: (64, 176, 9, 9)
        x = self.preliminary_layer(x)  # (64, 96, 9, 9)
        x = self.convgru(x)  # (64, 48, 9, 9)
        x = self.group_normalization(x)  # (64, 48, 9, 9)
        levels = self.glom(x)  # (batch, 81, 5, 32)
        top_level = self.to_latent(levels[:, :, -1])  # (batch, 81, 32)
        x = self.conv1d(top_level)
        output = self.mlp_head(x.squeeze())
        return output
    

def auxiliary_test_epoch(model, test_loader):
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
            
            batch_pred = model(batch_data)  # (batch, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            pp = pred.squeeze()  # (100, )
            
            tar = np.append(tar, batch_target.data.cpu().numpy())
            pre = np.append(pre, pp.data.cpu().numpy())
    return tar, pre


def auxiliary_valid_epoch(model, true_loader):
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(true_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
        
            batch_pred = model(batch_data)  # (100, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            pp = pred.squeeze()  # (100, )
            pre = np.append(pre, pp.data.cpu().numpy())
    return pre  # (100, )


def auxiliary_train_epoch(model, train_loader, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   
        
        optimizer.zero_grad()
        batch_pred = model(batch_data)  # (batch, 13)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)  # 计算所有训练样本的平均损失
        top1.update(prec1[0].data, n)  # 计算所有训练样本的平均准确率
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
