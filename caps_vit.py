import torch
import numpy as np
import torch.nn as nn
from torch import optim
from copy import deepcopy
from torch import autograd
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat


def conv3x3_bn_relu(band, out_channel=96):
    return nn.Sequential(
        nn.Conv2d(band, out_channel, 3, 1, 1), 
        nn.BatchNorm2d(out_channel), 
        nn.ReLU(inplace=True)
    )


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        
        assert hidden_channels % 2 == 0, 'The number of output channels should be divisible by 2'
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxz = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.Whz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxr = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.Whr = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxg = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding)
        self.Whg = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
    
    def forward(self, x, h):
        z = torch.sigmoid(self.Wxz(x) + self.Whz(h))
        r = torch.sigmoid(self.Wxr(x) + self.Whr(h))
        g = torch.tanh(self.Wxg(x) + self.Whg(r * h))
        h = z * h + (1 - z) * g
        return h
    
    def init_hidden(self, batch_size, shape):
        return Variable(torch.zeros(batch_size, self.hidden_channels, shape[0], shape[1])).cuda()
    
    
class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, steps=8):
        super(ConvGRU, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.steps = steps
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input_tensor):
        internal_state = []
        outputs = []
        total_channels = input_tensor.size(1)  # 96
        channels_per_step = int(total_channels / self.steps)  # 12
        for step in range(self.steps):
            x = input_tensor[:, step * channels_per_step:(step + 1) * channels_per_step, :, :]
            bsize, _, height, width = x.size()
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    h = getattr(self, name).init_hidden(batch_size=bsize, shape=(height, width))
                    internal_state.append(h)
                # do forward
                h = internal_state[i]
                x = getattr(self, name)(x, h)
                internal_state[i] = x
            # only record effective steps
            outputs.append(x)
        result = outputs[0]
        for i in range(self.steps - 1):
            result = torch.cat([result, outputs[i + 1]], dim=1)
        return result  # (batch_size, 48, 9, 9)


def squash(s, axis=-1, epsilon=1e-7):
    s_square_norm = torch.sum(torch.square(s), axis=axis, keepdims=True)  # (64, 784, 1)
    V = s_square_norm / (1. + s_square_norm) / torch.sqrt(s_square_norm + epsilon)
    return V * s


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, caps1_num, caps1_dim, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.caps1_dim = caps1_dim
        self.conv2d = nn.Conv2d(in_channels, caps1_num * caps1_dim, kernel_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        assert x.dim() == 4, 'The input should be a 4-dimensional tensor'  # (batch, in_channels, h, w)
        x = self.conv2d(x)  # (64, 16*8, 7, 7)
        x = self.relu(x)
        x = rearrange(x, 'b (n d) h w -> b (n h w) d', d=self.caps1_dim)
        x = squash(x)
        return x
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    
class PreNorm(nn.Module):
    def __init__(self, caps_dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(caps_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    
class FeedForward(nn.Module):
    def __init__(self, caps_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(caps_dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, caps_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    
class Attention(nn.Module):
    """
    caps_dim: The embedding feature (capsule feature) has a dimension of 8.
    dim_head: The dimension of attention. 16
    heads: The number of heads in the multi head attention mechanism. 4
    """
    def __init__(self, caps_dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads  # 64 The dimension after the multi head attention cascade, where there are 4 heads of attention.
        self.heads = heads  # 4
        self.scale = dim_head ** (-0.5)  # 16 ** (-0.5) = 0.25
        
        self.to_qkv = nn.Linear(caps_dim, inner_dim * 3, bias=False)  # (batch, 784, 64 * 3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, caps_dim), 
            nn.Dropout(dropout)  # Dropout also exists in attention.
        )
    
    def forward(self, x):
        # x: (batch, 784, 8)
        b, n, _, h = *x.shape, self.heads  # batch, 784, 8, 4
        
        # get qkv tuple: ((batch, 784, 64), (...), (...))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v: (batch, 4, 784, 16)
        
        # transpose(k) * q / sqrt(dim_head) -> (batch, heads, 784, 784)
        dots = torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale  # batch matrix multiplication: (batch, 4, 784, 784)
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)  # (batch, 4, 784, 784)
        # attn * attention matrix -> output
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)  # (batch, 4, 784, 16)
        # cat all output -> (batch, 400, dim_head * heads)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (batch, 784, 64)
        out = self.to_out(out)  # Restore to the dimension of the original embedded features for residual learning: (batch, 784, 8)
        return out
    
    
class Transformer(nn.Module):
    """
    caps_dim: The dimension of embedding feature (capsule feature). 8
    depth: Number of encoder layer in Transformer. 5
    heads: The number of heads in the multi head attention mechanism. 4
    mlp_dim: The dimension of the middle hidden layer of the MLP layer in the Transformer encoder. 8
    dim_head: The dimension of QKV. 16
    """
    def __init__(self, caps_dim, depth, heads, dim_head, mlp_dim, dropout, caps_num, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(caps_dim, Attention(caps_dim, heads=heads, dim_head=dim_head, dropout=dropout))), 
                Residual(PreNorm(caps_dim, FeedForward(caps_dim, mlp_dim, dropout=dropout)))
            ]))
        
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(caps_num, caps_num, [1, 2], 1, 0))
            
    def forward(self, x):
        # x: (batch, 784, 8)
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x)
                x = ff(x)
        elif self.mode == 'CAF':
            # x: (batch, 784, 8)
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)  # (x0, x1, x2, x3, x4)
                if nl > 1:
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x)
                x = ff(x)
                nl += 1  # 1, 2, 3, 4, 5
        return x
    
    
class ViT(nn.Module):
    """
    caps_dim: The dimension of embedding feature (capsule feature). 8
    depth: Number of encoder layer in Transformer. 5
    heads: The number of heads in the multi head attention mechanism. 4
    mlp_dim: The dimension of the middle hidden layer of the MLP layer in the Transformer encoder. 8
    dim_head: The dimension of QKV. 16
    """
    def __init__(self, caps_num, caps_dim, depth, heads, mlp_dim, dim_head=16, dropout=0.1, emb_dropout=0.1, 
                 mode='CAF'):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, caps_num, caps_dim))  # (1, 784, 8) Attention, must be 1 here!!!
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(caps_dim, depth, heads, dim_head, mlp_dim, dropout, caps_num, mode)
        
    def forward(self, x):
        # x: (batch, caps_num, caps_dim) (64, 784, 8)
        x += self.pos_embedding  # (64, 784, 8)
        x = self.dropout(x)
        # transformer: x: (batch, caps_num, caps_dim) -> (batch, caps_num, caps_dim)
        x = self.transformer(x)
        x = squash(x)  # Capsules processed by Transformer. (64, 784, 8)
        return x
    
    
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dim=False):
    squared_norm = torch.sum(torch.square(s), axis=axis, keepdim=keep_dim)
    return torch.sqrt(squared_norm + epsilon)


class DigitalCapsule(nn.Module):
    def __init__(self, in_caps_num, in_caps_dim, out_caps_num, out_caps_dim, routings=3):
        super(DigitalCapsule, self).__init__()
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.routings = routings
        
        # self.weight: (1, 784, 13, 16, 8)
        self.weight = nn.Parameter(0.01 * torch.randn(1, in_caps_num, out_caps_num, out_caps_dim, in_caps_dim))
        
    def forward(self, x):
        # x: (64, 784, 8)
        # x[:, :, None, :, None]: (64, 784, 1, 8, 1)
        x_hat = torch.matmul(self.weight, x[:, :, None, :, None])  # (batch, 784, 13, 16, 1)
        x_hat_detached = x_hat.detach()
        
        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps] (64, 13, 784)
        b = torch.zeros(x.size(0), self.in_caps_num, self.out_caps_num, 1, 1).cuda()  # (batch, 784, 13, 1, 1)
        
        assert self.routings > 0, 'The routings should be > 0'
        for i in range(self.routings):
            c = F.softmax(b, dim=2)  # (batch, 784, 13, 1, 1)
            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                vj = squash(torch.sum(c.mul(x_hat), dim=1, keepdim=True), axis=-2)  # (batch, 1, 13, 16, 1)
            # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path
            else:
                vj = squash(torch.sum(c.mul(x_hat_detached), dim=1, keepdim=True), axis=-2)  # (batch, 1, 13, 16, 1)
                vj_tiled = repeat(vj, 'b () i j k -> b n i j k', n=self.in_caps_num)  # (batch, 784, 13, 16, 1)
                agreement = torch.matmul(x_hat_detached.transpose(-1, -2), vj_tiled)  # (batch, 784, 13, 1, 1)
                b += agreement        
        return torch.squeeze(torch.squeeze(vj, 1), -1)  # (batch, 1, 13, 16, 1) -> (batch, 13, 16)


class Decoder(nn.Module):
    def __init__(self, caps2_dim, band, patch_size=9):
        super(Decoder, self).__init__()
        self.in_channels = caps2_dim
        self.band = band  # 176
        self.bn = nn.BatchNorm1d(caps2_dim)
        
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 3, stride=1), nn.BatchNorm2d(256), nn.ReLU(), # (64, 256, 3, 3)
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(), # (64, 256, 5, 5)
            nn.ConvTranspose2d(256, self.band, 3, stride=2, padding=1)  # (batch_size, 176, 9, 9) when spatial_size=7, stride=1, padding=0
        )  
        
    def forward(self, encode):  # x: (batch, 16)
        encode = self.bn(encode)
        return self.model(encode[:, :, None, None])
   
    
class CapsViT(nn.Module):
    def __init__(self, band, patch_size=9, preliminary_layer_channels=96, num_group_norm=8, steps=8, 
                 caps1_num=16, caps1_dim=8, kernel_size=3, mode='CAF', caps2_caps=13, caps2_dim=16):
        super(CapsViT, self).__init__()
        
        self.preliminary_layer = conv3x3_bn_relu(band, preliminary_layer_channels)

        cl_channel = preliminary_layer_channels / steps
        cl_channel = int(cl_channel)  # 12
        cl2_channel = int(cl_channel / 2)  # 6
        self.convgru = ConvGRU(cl_channel, hidden_channels=[cl_channel, cl2_channel], kernel_size=kernel_size)
        
        output_channel = cl2_channel * steps  # 48
        self.group_normalization = nn.GroupNorm(num_group_norm, output_channel)
        
        self.primary_caps = PrimaryCapsule(output_channel, caps1_num=caps1_num, caps1_dim=caps1_dim, 
                                           kernel_size=3)
        
        padding = int((kernel_size - 1) / 2)  # 1
        spatial_size = patch_size - kernel_size + 1  # 5
        caps1_caps = caps1_num * spatial_size ** 2  # 784
        self.vit = ViT(caps_num=caps1_caps, caps_dim=caps1_dim, depth=5, heads=4, mlp_dim=8, dropout=0.1, 
                       emb_dropout = 0.1, mode=mode)
        
        self.digital_cap = DigitalCapsule(caps1_caps, caps1_dim, caps2_caps, caps2_dim)
        self.decoder = Decoder(caps2_dim, band, patch_size)
        
    def forward(self, x, y=None):
        # x: (64, 176, 9, 9)
        x = self.preliminary_layer(x)  # (64, 96, 9, 9)
        x = self.convgru(x)  # (64, 48, 9, 9)
        x = self.group_normalization(x)  # (64, 48, 9, 9)
        x = self.primary_caps(x)  # (64, 784, 8)
        x = self.vit(x)  # (batch, caps1_caps, caps1_dim) -> (64, 784, 8)
        x = self.digital_cap(x)  # (batch, 13, 16)
        if self.training:
            encode = torch.stack([x[ind, y[ind], :] for ind in range(x.size(0))])  # (batch, 16)
            x_recon = self.decoder(encode)
            return x, x_recon
        else:
            y_proba = safe_norm(x, axis=-1)  # (batch, 13)
            y_pred = torch.argmax(y_proba, axis=1)  # (batch, )
            encode = torch.stack([x[ind, y_pred[ind], :] for ind in range(x.size(0))])  # (batch, 16)
            return x, encode


class Margin_Recon_Loss(nn.Module):
    def __init__(self):
        super(Margin_Recon_Loss, self).__init__()
        
    def forward(self, digital_caps_output, y, x, x_recon, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        num_classes = digital_caps_output.size(1)
        caps2_output_norm = safe_norm(digital_caps_output, axis=-1)  # (batch, 13)
        T = F.one_hot(y, num_classes=num_classes)  # (batch, 13)
        present_error = torch.square(torch.maximum(torch.tensor(0.).cuda(), m_plus - caps2_output_norm))  # (batch, 13)
        absent_error = torch.square(torch.maximum(torch.tensor(0.).cuda(), caps2_output_norm - m_minus))  # (batch, 13)
        L = T * present_error + lambda_ * (1.0 - T) * absent_error  # (batch, 13)
        margin_loss = torch.mean(torch.sum(L, axis=1))
        recon_loss = nn.MSELoss()(x_recon, x)
        return margin_loss + 1e-2 * recon_loss