import copy
from pathlib import Path

import numpy as np
import torch

from config import smoke_config
from data import build_paired_split
from diffusion import DDPM
from train import _model, train_one


def test_completed_checkpoint_resume_preserves_loss_and_weights(tmp_path: Path):
    cfg = smoke_config(tmp_path)
    cfg.training_steps = 1
    split = build_paired_split(cfg.d, cfg.seed, cfg.rotation_deg, cfg.n_target_train,
                               cfg.n_aux_train, cfg.n_validation, cfg.n_test)
    diffusion = DDPM(cfg.T, cfg.beta_start, cfg.beta_end, torch.device("cpu"))
    torch.manual_seed(77)
    initial = copy.deepcopy(_model(cfg).state_dict())
    trained, loss, checkpoint = train_one(cfg, "target_only", split, diffusion,
                                          initial, resume=False)
    expected = {key: value.detach().clone() for key, value in trained.state_dict().items()}
    assert checkpoint.exists() and np.isfinite(loss)

    def fail_if_retrained(*args, **kwargs):
        raise AssertionError("completed checkpoint was trained again")
    diffusion.loss = fail_if_retrained
    resumed, resumed_loss, _ = train_one(cfg, "target_only", split, diffusion,
                                         initial, resume=True)
    assert resumed_loss == loss and np.isfinite(resumed_loss)
    assert all(torch.equal(expected[key], resumed.state_dict()[key]) for key in expected)
