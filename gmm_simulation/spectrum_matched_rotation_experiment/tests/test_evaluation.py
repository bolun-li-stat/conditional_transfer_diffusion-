import numpy as np
import torch

from conditional_model import ConditionalDenoiser
from diffusion import DDPM
from eval import epsilon_mse, gradient_alignment


class FixedPrediction(torch.nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, x, t, labels):
        return self.prediction[:len(x)]


def test_epsilon_mse_is_per_coordinate(monkeypatch):
    device = torch.device("cpu")
    diffusion = DDPM(4, 1e-4, 2e-2, device)
    noise = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -2.0]])
    prediction = torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, -1.0]])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())
    observed = epsilon_mse(FixedPrediction(prediction), diffusion,
                           np.zeros((2, 3), dtype=np.float32), seed=7, batch_size=2)
    expected = float(((prediction - noise) ** 2).mean(dim=1).mean())
    assert observed == expected
    assert observed != expected * 3


def _alignment(seed: int, recorder=None):
    torch.manual_seed(4)
    model = ConditionalDenoiser(4, 3, 8, 3, 12, 2)
    diffusion = DDPM(10, 1e-4, 2e-2, torch.device("cpu"))
    if recorder is not None:
        original = diffusion.q_sample
        def recording_q_sample(x, t, noise):
            recorder.append((t.clone(), noise.clone()))
            return original(x, t, noise)
        diffusion.q_sample = recording_q_sample
    arrays = [np.random.default_rng(10 + label).normal(size=(8, 4)).astype(np.float32)
              for label in range(3)]
    return gradient_alignment(model, diffusion, arrays, seed, 8)


def test_gradient_alignment_is_reproducible_and_seeded():
    first = _alignment(123)
    second = _alignment(123)
    changed = _alignment(124)
    assert first == second
    assert not np.allclose(first, changed)


def test_gradient_alignment_reuses_timestep_and_noise():
    records = []
    _alignment(123, records)
    assert len(records) == 3
    for timesteps, noise in records[1:]:
        assert torch.equal(timesteps, records[0][0])
        assert torch.equal(noise, records[0][1])
