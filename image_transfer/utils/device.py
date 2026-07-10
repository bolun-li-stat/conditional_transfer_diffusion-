import torch

def get_device(name: str | None = None) -> torch.device:
    if name and name != 'auto': return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
