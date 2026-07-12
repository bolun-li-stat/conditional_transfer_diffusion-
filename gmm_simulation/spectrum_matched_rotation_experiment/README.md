# Spectrum-matched covariance rotation experiment

This equal-target-sample-size study isolates covariance orientation mismatch. In the older AR(1) design, changing `rho` changes both eigenvalues and principal directions, so mismatch also changes marginal task difficulty. Here every class covariance has eigenvalues `1.8` and `0.2` with equal multiplicity (trace `d`, condition number `9`); paired rotations change only principal directions. All means are zero so transfer cannot be attributed to mean separation.

## Why the spectrum is fixed at 1.8 / 0.2

The eigenvalues are written as `1 ± delta` with `delta = 0.8`. Because each
eigenvalue occurs `d/2` times, the mean eigenvalue is one and `trace(Sigma)=d`,
keeping the overall data scale aligned with a unit-variance reference. The
condition number is 9: anisotropy is strong enough for rotations to have a
visible effect, without using an eigenvalue close to zero and making the problem
unnecessarily ill-conditioned. This is a preregistered design choice, not a
theoretically unique spectrum, and it will not be tuned after viewing results.

For the paired block rotation,

```text
||Sigma_theta - Sigma_0||_F / ||Sigma_0||_F
  = sqrt(2) |lambda_high-lambda_low| |sin(theta)|
    / sqrt(lambda_high^2+lambda_low^2).
```

This relative distance is 0 at 0 degrees, approximately 0.883 at 45 degrees,
and approximately 1.207 at 75 degrees. We use 75 rather than 90 degrees because
75 is already near maximal orientation mismatch, while the `+theta` and
`-theta` auxiliary covariances remain distinct. At 90 degrees they coincide as
the same complete eigenspace swap.

At `theta=0`, all distributions coincide, providing a positive-transfer control. With limited shared capacity, large rotations can create competing shared-gradient demands and may cause negative transfer. This is a finite-model/optimization hypothesis, not an inevitability: sufficiently expressive models can represent class-specific score maps without harmful interference.

Both `target_only` and `joint_conditional` use the same `ConditionalDenoiser`, three-class embedding table, initialization seed, and target samples. The target-only model always receives label 0. It is trained once per seed and capacity because it does not depend on rotation; analysis pairs that common baseline with each joint run before taking averages.

The primary metric is integrated target score risk. Noise-bin risks draw timesteps uniformly within `low=0..99`, `mid=100..499`, and `high=500..999`; truncated smoke configurations report only nonempty bins. Other metrics are target validation epsilon-MSE and Gaussian W2 squared, with mean/covariance errors as diagnostics. No RBF-MMD is computed.

## Commands

```bash
python train.py --experiment smoke

python train.py --experiment full --seeds 0 1 2 3 4 \
  --results-dir results_rotation --resume
python train.py --experiment full --seeds 5 6 7 8 9 \
  --results-dir results_rotation --resume
# Continue with seeds 10..19 in additional shards.

python analyze_results.py --results-dir results_rotation
```

For a complete run, shard all 20 seeds without overlap, for example `0..4`, `5..9`,
`10..14`, and `15..19`, then analyze the shared results directory after all shards finish.

CLI overrides include `--training-steps`, `--device`, `--n-generated`, `--score-risk-mc-samples`, `--skip-generation`, `--force`, and `--resume`. `--skip-generation` leaves W2 and generation diagnostics missing while still evaluating score risk and validation epsilon-MSE. Resume/checkpoint IDs include seed, capacity, model type, training steps, dimension, and spectrum; joint IDs additionally include rotation. Results are upserted by this key and existing rows are not rerun without `--force`.

Training writes only to `results_rotation/metrics/seed_NNN.csv`. Each file is
protected by a file lock and atomically replaced, so non-overlapping seed shards
can safely share one results directory; an accidental same-seed overlap cannot
corrupt the CSV. `analyze_results.py` reads all per-seed files and produces the
read-only consolidated artifact `results_rotation/metrics.csv`. Pairing uses a
full auditable configuration hash, not merely seed and capacity. A partial seed
set is always labeled `incomplete`; only the preregistered 20-seed set can be
classified as positive, negative, or inconclusive.

The full prespecified grid uses seeds 0 through 19, angles 0/45/75, and standard/limited capacities: 120 joint models plus 40 common target-only models.
