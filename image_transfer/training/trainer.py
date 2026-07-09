from __future__ import annotations
import time, torch
from torch.utils.data import DataLoader
from image_transfer.diffusion.ddpm import ImageDDPM
from image_transfer.models.unet import ImageUNet
from image_transfer.training.ema import EMA
from image_transfer.training.checkpointing import save_checkpoint
from image_transfer.evaluation.denoising_loss import evaluate_denoising_bins

def train_image_model(dataset, val_dataset, *, conditional, num_classes, image_size, base_channels, channel_mults, timesteps, schedule, steps, batch_size, lr, device, precision='fp32', ema_decay=0.999, checkpoint_path=None):
    device=torch.device(device); model=ImageUNet(image_size=image_size, base_channels=base_channels, channel_mults=tuple(channel_mults), num_classes=num_classes if conditional else None).to(device)
    diff=ImageDDPM(timesteps=timesteps, schedule=schedule, device=device); opt=torch.optim.AdamW(model.parameters(), lr=lr); ema=EMA(model, ema_decay)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False); it=iter(loader); final=float('nan'); start=time.time()
    for step in range(max(0,steps)):
        try: x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device); y=y.to(device) if conditional else None
        opt.zero_grad(set_to_none=True); loss=diff.loss(model,x,y); loss.backward(); opt.step(); ema.update(model); final=float(loss.item())
    train_seconds=time.time()-start
    valloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0) if val_dataset is not None else loader
    den=evaluate_denoising_bins(ema.shadow,diff,valloader,device,label=0 if conditional else None)
    if checkpoint_path: save_checkpoint(checkpoint_path, ema.shadow, opt, steps, {'conditional':conditional})
    return ema.shadow, diff, {'final_train_loss':final,'wallclock_train_seconds':train_seconds,'validation_epsilon_mse_target':den['all'],'validation_epsilon_mse_low_noise':den['low'],'validation_epsilon_mse_mid_noise':den['mid'],'validation_epsilon_mse_high_noise':den['high']}
