# Gaussian simulation experiments

Two independent Gaussian DDPM studies live here:

- [`original_ar1_experiment/`](original_ar1_experiment/README.md): the AR(1)-Toeplitz experiment with frozen legacy paths and an additive target-only conditional control.
- [`spectrum_matched_rotation_experiment/`](spectrum_matched_rotation_experiment/README.md): a spectrum-controlled covariance-orientation experiment with architecture-matched baselines.

Run commands from the corresponding subdirectory. Neither experiment imports Python modules from the other.

## Three-estimator controls

Both studies now support the same scientific estimators while retaining their
historical on-disk names:

| Estimator | Meaning | Original AR(1) `model_type` | Rotation `model_type` |
|---|---|---|---|
| `U_T` | target-only unconditional | `unconditional` | `unconditional` |
| `C_T` | target-only conditional control | `target_only_conditional` | `target_only` |
| `C_J` | joint conditional, with auxiliary classes | `conditional` | `joint_conditional` |

For every lower-is-better metric, the primary paper comparison is `C_J-U_T`.
The architecture-matched auxiliary-training comparison is `C_J-C_T`, and
`C_T-U_T` is the architecture/parameterization gap (not a causal architecture
effect). Negative gaps favor the first model; positive gaps favor the second.
The identity `C_J-U_T=(C_J-C_T)+(C_T-U_T)` is checked on complete paired rows.

This is an additive extension. Omitting `--model-types` preserves each module's
legacy model set, order, identities, checkpoints, and results. Existing
checkpoints need not be retrained; either new model can be run alone into an
existing results directory. Analyses retain computable comparisons and mark
comparisons with a missing estimator incomplete.

Both modules have independent hosted GMM test jobs. Exact tiny-CPU legacy
non-regression can be reproduced from the repository root with:

```bash
git worktree add --detach /tmp/gmm-base <BASE_COMMIT_SHA>
python gmm_simulation/tests/golden_regression.py \
  --base-worktree /tmp/gmm-base \
  --modified-worktree .
git worktree remove /tmp/gmm-base
```

The runner covers both Original legacy experiments and the Rotation legacy
pair, compares loaded model/optimizer/generator states, logs, metrics, IDs,
configs, and saved arrays, prints a JSON summary, and removes its tiny results.
