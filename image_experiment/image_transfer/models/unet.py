from __future__ import annotations
import torch
from torch import nn
from .blocks import ResBlock, AttentionBlock, timestep_embedding

class ImageUNet(nn.Module):
    def __init__(self, image_size=32, in_channels=3, base_channels=64, channel_mults=(1,2,2,4), num_classes: int | None=None, time_dim: int | None=None):
        super().__init__(); self.num_classes=num_classes; time_dim=time_dim or base_channels*4
        self.time_mlp=nn.Sequential(nn.Linear(base_channels,time_dim),nn.SiLU(),nn.Linear(time_dim,time_dim))
        chs=[base_channels*m for m in channel_mults]
        self.in_conv=nn.Conv2d(in_channels,chs[0],3,padding=1)
        self.downs=nn.ModuleList(); in_ch=chs[0]; res=image_size; skips=[]
        for ch in chs:
            self.downs.append(ResBlock(in_ch,ch,time_dim)); skips.append(ch); in_ch=ch
            if res==16: self.downs.append(AttentionBlock(ch))
            if ch != chs[-1]: self.downs.append(nn.Conv2d(ch,ch,4,2,1)); res//=2
        self.mid=nn.Sequential(ResBlock(in_ch,in_ch,time_dim), AttentionBlock(in_ch), ResBlock(in_ch,in_ch,time_dim))
        self.ups=nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(ResBlock(in_ch+ch,ch,time_dim)); in_ch=ch
            if res==16: self.ups.append(AttentionBlock(ch))
            if ch != chs[0]: self.ups.append(nn.ConvTranspose2d(ch,ch,4,2,1)); res*=2
        self.out=nn.Sequential(nn.GroupNorm(min(8,chs[0]),chs[0]),nn.SiLU(),nn.Conv2d(chs[0],in_channels,3,padding=1))
        # Keep the class-only parameter last.  ``nn.Module`` constructors consume
        # the global torch RNG while initializing their parameters.  Constructing
        # the embedding before the backbone therefore used to shift every shared
        # parameter whenever conditioning was enabled (or ``num_classes``
        # changed).  With the embedding last, resetting the same model seed before
        # constructing a conditional and an unconditional U-Net gives bitwise
        # identical values for every common state-dict key.
        self.class_emb=nn.Embedding(num_classes,time_dim) if num_classes is not None else None
        self.base_channels=base_channels
    def forward(self,x,t,y=None):
        emb=self.time_mlp(timestep_embedding(t,self.base_channels))
        if self.class_emb is not None:
            if y is None: raise ValueError('Conditional ImageUNet requires labels')
            emb=emb+self.class_emb(y)
        h=self.in_conv(x); stack=[]
        for m in self.downs:
            if isinstance(m,ResBlock): h=m(h,emb); stack.append(h)
            else: h=m(h)
        for m in self.mid: h=m(h,emb) if isinstance(m,ResBlock) else m(h)
        for m in self.ups:
            if isinstance(m,ResBlock):
                s=stack.pop();
                if s.shape[-2:]!=h.shape[-2:]: s=torch.nn.functional.interpolate(s,size=h.shape[-2:],mode='nearest')
                h=m(torch.cat([h,s],dim=1),emb)
            else: h=m(h)
        return self.out(h)
