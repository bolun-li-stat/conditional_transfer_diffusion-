from pathlib import Path
import torch

def save_checkpoint(path, model, optimizer=None, step=0, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True); torch.save({'model':model.state_dict(),'optimizer': optimizer.state_dict() if optimizer else None,'step':step,'extra':extra or {}}, path)
