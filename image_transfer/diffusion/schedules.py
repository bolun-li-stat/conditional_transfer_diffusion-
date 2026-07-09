import math, torch

def make_beta_schedule(kind: str, timesteps: int, beta_start=1e-4, beta_end=0.02):
    if kind == 'linear': return torch.linspace(beta_start,beta_end,timesteps)
    if kind == 'cosine':
        s=0.008; steps=timesteps+1; x=torch.linspace(0,timesteps,steps)
        ac=torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2; ac=ac/ac[0]
        return torch.clamp(1-ac[1:]/ac[:-1], max=0.999)
    raise ValueError(f'Unknown beta schedule {kind}')
