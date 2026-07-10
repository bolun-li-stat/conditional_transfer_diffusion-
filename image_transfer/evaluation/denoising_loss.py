import torch
@torch.no_grad()
def evaluate_denoising_bins(model,diffusion,loader,device,label=None,max_batches=4):
    bins={'low':[],'mid':[],'high':[],'all':[]}; model.eval()
    for i,(x,_) in enumerate(loader):
        if i>=max_batches: break
        x=x.to(device); y=None if label is None else torch.full((x.shape[0],),label,device=device,dtype=torch.long)
        for name,lo,hi in [('low',0,0.2),('mid',0.2,0.7),('high',0.7,1.0)]:
            t=torch.randint(int(lo*diffusion.timesteps), max(int(hi*diffusion.timesteps),1), (x.shape[0],), device=device)
            xt,eps=diffusion.q_sample(x,t); mse=torch.nn.functional.mse_loss(model(xt,t,y),eps).item(); bins[name].append(mse); bins['all'].append(mse)
    return {k:(sum(v)/len(v) if v else float('nan')) for k,v in bins.items()}
