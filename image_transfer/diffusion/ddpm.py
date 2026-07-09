from __future__ import annotations
import torch
from .schedules import make_beta_schedule

class ImageDDPM:
    def __init__(self, timesteps=1000, schedule='linear', device='cpu'):
        self.timesteps=timesteps; self.device=torch.device(device); b=make_beta_schedule(schedule,timesteps).to(self.device)
        self.betas=b; self.alphas=1-b; self.alpha_bars=torch.cumprod(self.alphas,0)
    def q_sample(self,x0,t,noise=None):
        if noise is None: noise=torch.randn_like(x0)
        ab=self.alpha_bars[t].view(-1,1,1,1); return ab.sqrt()*x0+(1-ab).sqrt()*noise, noise
    def loss(self,model,x0,y=None):
        t=torch.randint(0,self.timesteps,(x0.shape[0],),device=x0.device); xt,eps=self.q_sample(x0,t)
        return torch.nn.functional.mse_loss(model(xt,t,y),eps)
    @torch.no_grad()
    def sample(self,model,shape,y=None,steps=None):
        model.eval(); steps=steps or self.timesteps; x=torch.randn(shape,device=self.device)
        seq=torch.linspace(self.timesteps-1,0,steps,device=self.device).long()
        for tval in seq:
            t=torch.full((shape[0],),int(tval),device=self.device,dtype=torch.long); beta=self.betas[t].view(-1,1,1,1); alpha=self.alphas[t].view(-1,1,1,1); ab=self.alpha_bars[t].view(-1,1,1,1)
            eps=model(x,t,y); mean=(x-beta/(1-ab).sqrt()*eps)/alpha.sqrt()
            x=mean if int(tval)==0 else mean+beta.sqrt()*torch.randn_like(x)
        return x.clamp(-1,1)
