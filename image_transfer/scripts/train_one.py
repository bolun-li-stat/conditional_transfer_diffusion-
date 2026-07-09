from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from image_transfer.utils.io import load_yaml, ensure_dir, append_csv_row, resolve_env_path
from image_transfer.utils.seed import set_seed
from image_transfer.utils.device import get_device
from image_transfer.training.trainer import train_image_model
from image_transfer.scripts.make_job_grid import EXP_DIR
FIELDS=['dataset','experiment','experiment_name','target_synset','target_name','aux_set','aux_synsets','aux_composition','model_type','n0','m_per_aux','K_aux','total_train_images','seed','image_size','training_steps','checkpoint_path','final_train_loss','validation_epsilon_mse_target','validation_epsilon_mse_low_noise','validation_epsilon_mse_mid_noise','validation_epsilon_mse_high_noise','fid_target','kid_target_mean','kid_target_std','classifier_target_top1_acc','classifier_target_top5_acc','auxiliary_leakage_rate','top1_prediction_histogram_json','num_generated','num_real_eval','sampler','sampling_steps','wallclock_train_seconds','wallclock_eval_seconds','skipped_equal_total_baseline','skip_reason']

def fake_dataset(n, image_size, num_classes, seed):
    g=torch.Generator().manual_seed(seed); x=torch.rand(n,3,image_size,image_size,generator=g)*2-1; y=torch.arange(n)%num_classes; return TensorDataset(x,y.long())

def run(args, job=None):
    cfg=load_yaml(args.config); exp=args.experiment or (job and job['experiment']) or 'A'; set_seed(int(args.seed or (job and job.get('seed',0)) or 0))
    image_size=int(args.image_size or cfg.get('image_size',32)); n0=int(args.n0 or (job and job.get('n0',8)) or 8); m=int(args.m_per_aux or (job and job.get('m_per_aux',n0)) or n0); k=int(args.K_aux or (job and job.get('K_aux',cfg.get('K_aux',1))) or 1)
    steps=int(args.max_steps if args.max_steps is not None else cfg.get('training',{}).get('steps',2)); num_classes=1+k; model_type=(job and job.get('model_type')) or 'unconditional_n0'; conditional=not model_type.startswith('unconditional')
    total=n0+(k*m if conditional else 0)
    if model_type=='unconditional_equal_total': total=n0+k*m
    dataset=fake_dataset(max(total,1),image_size,num_classes,int(args.seed or 0)); val=fake_dataset(max(n0,1),image_size,num_classes,int(args.seed or 0)+1)
    outdir=Path((job and job.get('output_dir')) or Path(resolve_env_path(cfg.get('output_root'),'image_transfer_results'))/EXP_DIR[exp]); ensure_dir(outdir)
    ckpt=outdir/'checkpoints'/f'{model_type}_n0{n0}_seed{args.seed or 0}.pt'
    t0=time.time(); model,diff,metrics=train_image_model(dataset,val,conditional=conditional,num_classes=num_classes,image_size=image_size,base_channels=int(cfg.get('model',{}).get('base_channels',16)),channel_mults=cfg.get('model',{}).get('channel_mults',[1]),timesteps=int(cfg.get('diffusion',{}).get('timesteps',10)),schedule=cfg.get('diffusion',{}).get('schedule','linear'),steps=steps,batch_size=int(cfg.get('training',{}).get('batch_size',4)),lr=float(cfg.get('optimizer',{}).get('lr',1e-3)),device=get_device(args.device or cfg.get('device','cpu')),checkpoint_path=ckpt)
    eval_start=time.time(); samples=diff.sample(model,(int(args.num_generated or cfg.get('num_generated',4)),3,image_size,image_size), y=torch.zeros(int(args.num_generated or 4),dtype=torch.long,device=diff.device) if conditional else None, steps=int(cfg.get('sampling_steps',2))); wall_eval=time.time()-eval_start
    ensure_dir(outdir/'samples'); torch.save(samples.cpu(), outdir/'samples'/f'{model_type}_samples.pt')
    row={f:'' for f in FIELDS}; row.update(metrics); row.update({'dataset':cfg.get('dataset','fake'),'experiment':exp,'experiment_name':{'A':'equal_target','B':'equal_total','C':'similarity_sweep'}[exp],'target_synset':(job or {}).get('target_synset','dog'),'target_name':(job or {}).get('target_name','dog'),'aux_set':(job or {}).get('aux_set','none'),'aux_synsets':(job or {}).get('aux_composition','[]'),'aux_composition':(job or {}).get('aux_set','none'),'model_type':model_type,'n0':n0,'m_per_aux':m,'K_aux':k,'total_train_images':total,'seed':int(args.seed or (job and job.get('seed',0)) or 0),'image_size':image_size,'training_steps':steps,'checkpoint_path':str(ckpt),'fid_target':float('nan'),'kid_target_mean':float('nan'),'kid_target_std':float('nan'),'classifier_target_top1_acc':float('nan'),'classifier_target_top5_acc':float('nan'),'auxiliary_leakage_rate':float('nan'),'top1_prediction_histogram_json':'{}','num_generated':int(args.num_generated or cfg.get('num_generated',4)),'num_real_eval':n0,'sampler':'ddpm','sampling_steps':int(cfg.get('sampling_steps',2)),'wallclock_eval_seconds':wall_eval,'skipped_equal_total_baseline':False,'skip_reason':''})
    append_csv_row(outdir/'metrics.csv', row, FIELDS); append_csv_row(Path(resolve_env_path(cfg.get('output_root'),'image_transfer_results'))/'all_metrics.csv', row, FIELDS)
    print(outdir/'metrics.csv')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--experiment',choices=['A','B','C']); p.add_argument('--max-steps',type=int); p.add_argument('--n0',type=int); p.add_argument('--m-per-aux',type=int); p.add_argument('--num-generated',type=int); p.add_argument('--device'); p.add_argument('--seed',type=int,default=0); p.add_argument('--image-size',type=int); p.add_argument('--K-aux',type=int,dest='K_aux')
    run(p.parse_args())
