import copy, torch
class EMA:
    def __init__(self, model, decay=0.999): self.decay=decay; self.shadow=copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in self.shadow.parameters()]
    @torch.no_grad()
    def update(self, model):
        for a,b in zip(self.shadow.parameters(), model.parameters()): a.mul_(self.decay).add_(b, alpha=1-self.decay)
