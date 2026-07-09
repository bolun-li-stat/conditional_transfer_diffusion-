from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from image_transfer.utils.io import load_yaml, ensure_dir, resolve_env_path
from image_transfer.data.class_sets import select_aux_synsets, class_name
EXP_DIR={'A':'A_equal_target','B':'B_equal_total','C':'C_similarity_sweep'}
EXP_NAME={'A':'equal_target','B':'equal_total','C':'similarity_sweep'}
REQ=['experiment','experiment_name','dataset','target_synset','target_name','aux_set','aux_composition','n0','m_per_aux','K_aux','seed','model_type','config_path','output_dir']

def rows_for(exp,cfg,config_path):
    dataset=cfg.get('dataset','cifar10'); outroot=resolve_env_path(cfg.get('output_root'),'image_transfer_results'); k=int(cfg.get('K_aux',5)); seeds=cfg.get('seeds',[0]); targets=cfg.get('targets',[{'synset':'dog','name':'dog'}])
    ecfg=cfg.get('experiments',{}).get(exp,{}); n0s=ecfg.get('n0_values',cfg.get('n0_values',[100])); aux_sets=ecfg.get('aux_sets',['close','medium','far','mix'])
    if exp=='C': aux_sets=ecfg.get('compositions',['close_only','mostly_close','balanced_mix','mostly_far','far_only'])
    rows=[]
    for target in targets:
      ts=target.get('synset') or target.get('name'); tn=target.get('name') or class_name(ts); aux_cfg=target.get('auxiliary_sets') or cfg.get('auxiliary_sets',{})
      for n0 in n0s:
       m=int(n0 if cfg.get('m_per_aux_rule','equal_n0')=='equal_n0' else cfg.get('m_per_aux',n0))
       for aux in aux_sets:
        model_types=['unconditional_n0'] if exp=='A' else []
        if exp=='B': model_types.append('unconditional_equal_total')
        model_types += [f'conditional_{aux}' if exp!='C' else f'similarity_{aux}' ]
        for mt in model_types:
         for seed in seeds:
          comp = 'none' if mt.startswith('unconditional') else aux
          aux_syn=[] if comp=='none' else select_aux_synsets(aux_cfg, 'mix' if aux=='mix' else aux, k)
          rows.append({'experiment':exp,'experiment_name':EXP_NAME[exp],'dataset':dataset,'target_synset':ts,'target_name':tn,'aux_set':aux if comp!='none' else 'none','aux_composition':json.dumps(aux_syn),'n0':n0,'m_per_aux':m,'K_aux':k,'seed':seed,'model_type':mt,'config_path':str(config_path),'output_dir':str(Path(outroot)/EXP_DIR[exp])})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['A','B','C','all'],required=True); p.add_argument('--config',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    cfg=load_yaml(a.config); exps=['A','B','C'] if a.experiment=='all' else [a.experiment]
    allrows=[]
    for e in exps: allrows += rows_for(e,cfg,a.config)
    ensure_dir(Path(a.out).parent)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=REQ); w.writeheader(); w.writerows(allrows)
    print(f'wrote {len(allrows)} jobs to {a.out}')
if __name__=='__main__': main()
