"""Paired evaluation and mismatch diagnostics."""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def epsilon_mse(model, diffusion, x: np.ndarray, seed: int, batch_size: int = 512) -> float:
    gen = torch.Generator(device=diffusion.device).manual_seed(seed)
    xt = torch.as_tensor(x, device=diffusion.device)
    values = []
    for part in xt.split(batch_size):
        t = torch.randint(diffusion.T, (len(part),), device=diffusion.device, generator=gen)
        eps = torch.randn(part.shape, device=diffusion.device, generator=gen)
        noisy = diffusion.q_sample(part, t, eps)
        labels = torch.zeros(len(part), dtype=torch.long, device=diffusion.device)
        values.append(((model(noisy, t, labels)-eps)**2).sum(1).cpu())
    return float(torch.cat(values).mean())


@torch.no_grad()
def score_risk(model, diffusion, covariance: np.ndarray, n: int, seed: int,
               timestep_range: tuple[int, int] | None = None, batch_size: int = 512) -> float:
    rng = np.random.default_rng(seed)
    x0 = rng.multivariate_normal(np.zeros(len(covariance)), covariance, size=n).astype(np.float32)
    lo, hi = timestep_range or (0, diffusion.T)
    t_np = rng.integers(lo, hi, size=n)
    eps_np = rng.standard_normal(x0.shape).astype(np.float32)
    vals = []
    cov_t = torch.as_tensor(covariance, dtype=torch.float32, device=diffusion.device)
    eye = torch.eye(len(covariance), device=diffusion.device)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start+batch_size, n))
        x = torch.as_tensor(x0[sl], device=diffusion.device)
        t = torch.as_tensor(t_np[sl], device=diffusion.device)
        eps = torch.as_tensor(eps_np[sl], device=diffusion.device)
        noisy = diffusion.q_sample(x, t, eps)
        ab = diffusion.alpha_bars[t]
        learned = -model(noisy, t, torch.zeros(len(x), device=diffusion.device, dtype=torch.long)) / torch.sqrt(1-ab)[:, None]
        matrices = ab[:, None, None] * cov_t + (1-ab)[:, None, None] * eye
        truth = -torch.linalg.solve(matrices, noisy.unsqueeze(-1)).squeeze(-1)
        vals.append(((learned-truth)**2).sum(1).cpu())
    return float(torch.cat(vals).mean())


def generated_metrics(samples: np.ndarray, true_cov: np.ndarray) -> dict[str, float]:
    mean = samples.mean(0)
    cov = np.cov(samples, rowvar=False)
    vals, vecs = np.linalg.eigh((cov + cov.T)/2)
    sqrt_cov = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    middle = sqrt_cov @ true_cov @ sqrt_cov
    middle_vals = np.linalg.eigvalsh((middle + middle.T)/2)
    w2 = mean @ mean + np.trace(cov + true_cov) - 2*np.sqrt(np.clip(middle_vals, 0, None)).sum()
    return {"gaussian_w2_squared": float(max(w2, 0)), "mean_error": float(np.linalg.norm(mean)),
            "covariance_error": float(np.linalg.norm(cov-true_cov, "fro")/np.linalg.norm(true_cov, "fro"))}


def mismatch_diagnostics(target: np.ndarray, auxiliaries: list[np.ndarray], alpha_bars: np.ndarray) -> tuple[float, float]:
    relative = np.mean([np.linalg.norm(a-target, "fro")/np.linalg.norm(target, "fro") for a in auxiliaries])
    eye = np.eye(len(target))
    distances = []
    for ab in alpha_bars:
        target_inv = np.linalg.inv(ab*target + (1-ab)*eye)
        distances += [np.linalg.norm(np.linalg.inv(ab*a+(1-ab)*eye)-target_inv, "fro")**2/len(target) for a in auxiliaries]
    return float(relative), float(np.mean(distances))


def gradient_alignment(model, diffusion, class_arrays: list[np.ndarray], seed: int,
                       batch_size: int) -> tuple[float, float, float]:
    gradients = []
    for label, array in enumerate(class_arrays):
        gen = torch.Generator(device=diffusion.device).manual_seed(seed + label)
        x = torch.as_tensor(array[:batch_size], device=diffusion.device)
        labels = torch.full((len(x),), label, device=diffusion.device, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        loss = diffusion.loss(model, x, labels, gen)
        grads = torch.autograd.grad(loss, model.shared_parameters())
        gradients.append(torch.cat([g.detach().flatten() for g in grads]))
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0).cpu())
    c1, c2 = cos(gradients[0], gradients[1]), cos(gradients[0], gradients[2])
    return c1, c2, (c1+c2)/2
