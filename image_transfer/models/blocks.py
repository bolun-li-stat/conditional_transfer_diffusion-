import math, torch
from torch import nn

def timestep_embedding(t, dim):
    half=dim//2; freqs=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/(half-1 if half>1 else 1))
    args=t.float()[:,None]*freqs[None]
    emb=torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return torch.nn.functional.pad(emb,(0,dim-emb.shape[-1]))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__(); g1=min(8,in_ch); g2=min(8,out_ch)
        self.n1=nn.GroupNorm(g1,in_ch); self.c1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.e=nn.Linear(emb_dim,out_ch); self.n2=nn.GroupNorm(g2,out_ch); self.c2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.skip=nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity(); self.act=nn.SiLU()
    def forward(self,x,emb):
        h=self.c1(self.act(self.n1(x))); h=h+self.e(self.act(emb))[:,:,None,None]
        h=self.c2(self.act(self.n2(h))); return h+self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.norm=nn.GroupNorm(min(8,ch),ch); self.qkv=nn.Conv1d(ch,ch*3,1); self.proj=nn.Conv1d(ch,ch,1)
    def forward(self,x):
        b,c,h,w=x.shape; z=self.norm(x).view(b,c,h*w); q,k,v=self.qkv(z).chunk(3,dim=1)
        attn=torch.softmax(torch.bmm(q.transpose(1,2),k)/(c**0.5),dim=-1); out=torch.bmm(v,attn.transpose(1,2))
        return x+self.proj(out).view(b,c,h,w)
