import torch
from SpykeTorch import functional as sf
from torch import nn, einsum
from einops import rearrange
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda = True
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()

        self.conv1 = snn.Convolution(30, 30, 5, 0.8, 0.05)
        self.conv1_t = 10
        self.k1 = 1
        self.r1 = 2

        self.conv2 = snn.Convolution(30, 30, 2, 0.8, 0.05)
        self.conv2_t = 1
        self.k2 = 1
        self.r2 = 1

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0

    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            
            self.spk_cnt1 += 1
            if self.spk_cnt1 >= 500:

                self.spk_cnt1 = 0
                ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                ap = torch.min(ap, self.max_ap)
                an = ap * -0.75
                self.stdp1.update_all_learning_rate(ap.item(), an.item())
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
            self.save_data(input, pot, spk, winners)
                
            spk_in = sf.pad(sf.pooling(spk, 2, 2, 1), (10,10,10,10))
            spk_in = sf.pointwise_inhibition(spk_in)
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
            self.save_data(spk_in, pot, spk, winners)

            spk_out = sf.pooling(spk, 2, 2, 1)
            return spk_out
        else:

            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2, 1), (10,10,10,10)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            spk = sf.pooling(spk, 2, 2, 1)
            return spk
    
    def stdp(self):
        # print("how are you")

        self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

        self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.5,
                 last_stage=True):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)    
        q = self.to_q(x)
        self.to_q.stdp()  
        q=torch.stack(list(q), dim=0)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        v = self.to_v(x)
        self.to_v.stdp()        
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        self.to_k.stdp()        
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        # print("helloooooooooooooooooooooooooooooooo")
        return out



class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.5, last_stage=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class ViT(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes,s2_kernel_size,
               threshold, stdp_lr, anti_stdp_lr,dropout=0.5, image_size=28, dim=30, kernels=[7], strides=[4],
                 heads=[1] , depth = [1], pool='cls', emb_dropout=0.5, scale_dim=4):
        super(ViT, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.kernel_size = s2_kernel_size
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.anti_stdp_lr = anti_stdp_lr
        self.dropout = torch.ones(self.number_of_features) * dropout
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

        self.conv1 = snn.Convolution(input_channels, self.number_of_features, self.kernel_size, 0.8, 0.05)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        ##### Stage 1 #######
        self.R1= Rearrange('b c h w -> b (h w) c', h = image_size//2, w = image_size//2)
        self.norm1=nn.LayerNorm(dim)
        self.conv2=Transformer(dim=dim, img_size=image_size//2,depth=depth[0], heads=heads[0],dim_head=self.dim,
        mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True)


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, number_of_classes)
        )
        self.stdp = snn.STDP(self.conv1, stdp_lr)
        self.anti_stdp = snn.STDP(self.conv1, anti_stdp_lr)

        self.decision_map = []
        for i in range(self.number_of_classes):
            self.decision_map.extend([i]*self.features_per_class)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}


    def forward(self, input):
        input = input.float()
        pot = self.conv1(input)       
        if self.training and self.dropout[0] > 0:
            sf.feature_inhibition_(pot, self.to_be_dropped)

        spk, pot = sf.fire(pot, self.threshold, True)
        winners = sf.get_k_winners(pot, 1, 0, spk)
        output = -1
        pot= self.R1(pot)
        pot = self.norm1(pot)
        b, n, _ = pot.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        pot = torch.cat((cls_tokens, pot), dim=1)        
        pot = self.conv2(pot)            
        pot= pot.mean(dim=1) if self.pool == 'mean' else pot[:, 0]
        pot=self.mlp_head(pot)          
        # print("HELLO")               

        if len(winners) != 0:
            # print("yes11iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii1")
            output = self.decision_map[winners[0][0]]

        if self.training:
            self.ctx["input_spikes"] = input
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
        else:
            self.ctx["input_spikes"] = None
            self.ctx["potentials"] = None
            self.ctx["output_spikes"] = None
            self.ctx["winners"] = None
        # if output != -1:   
        #     print(output,"pppppppppppppppppppppppppppppppppppppp")
        return output
        


    def update_dropout(self):
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])    
   