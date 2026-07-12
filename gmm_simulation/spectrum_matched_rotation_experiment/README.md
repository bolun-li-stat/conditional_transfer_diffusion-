# Spectrum-matched covariance rotation experiment

This equal-target-sample-size study isolates covariance orientation mismatch. In the older AR(1) design, changing `rho` changes both eigenvalues and principal directions, so mismatch also changes marginal task difficulty. Here every class covariance has eigenvalues `1.8` and `0.2` with equal multiplicity (trace `d`, condition number `9`); paired rotations change only principal directions. All means are zero so transfer cannot be attributed to mean separation.

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

The full prespecified grid uses seeds 0 through 19, angles 0/45/75, and standard/limited capacities: 120 joint models plus 40 common target-only models.
