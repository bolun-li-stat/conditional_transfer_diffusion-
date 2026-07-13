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

## Capacities and transfer endpoints

The **standard** capacity (time embedding 64, class embedding 32, width 256,
four hidden layers) is the primary setting. The **limited** capacity (64, 8,
128, two layers) is a preregistered shared-capacity stress test, not an addition
made after seeing standard-capacity results. Both belong to the formal design.
If only limited capacity exhibits negative transfer, the conclusion must be
reported as capacity-dependent negative transfer. If neither does, that null
finding is reported without tuning the grid.

Integrated target score risk is the primary endpoint. For each paired seed,
`gap_score = score_risk_joint - score_risk_target_only`; negative values mean
positive score transfer and positive values mean negative score transfer.
Gaussian W2 squared is the main final-sample secondary endpoint, with
`gap_w2 = W2_joint - W2_target_only` and the analogous sign convention. These
produce separate `score_transfer_status` and `sample_transfer_status` values.
If their directions disagree, transfer is metric-dependent; there is no single
overall transfer status. Validation epsilon-MSE, mean error, and relative
covariance error are diagnostics rather than primary transfer definitions.

Noise-bin score risks draw timesteps uniformly within `low=0..99`,
`mid=100..499`, and `high=500..999`; truncated smoke configurations report only
nonempty bins. No RBF-MMD, FID, or additional sample metric is computed.

## Commands

```bash
python train.py --experiment smoke --skip-generation
python train.py --experiment smoke --generation-only --resume

python train.py --experiment full --capacity standard --seeds 0 1 2 3 4 \
  --results-dir results_rotation --skip-generation --resume
python train.py --experiment full --capacity standard --seeds 0 1 2 3 4 \
  --results-dir results_rotation --generation-only --resume

python analyze_results.py --results-dir results_rotation
```

For Colab, run eight training shards: four standard-capacity shards and four
limited-capacity shards, each covering `0..4`, `5..9`, `10..14`, or `15..19`.
Each shard trains 5 reusable target-only and 15 joint models. A session therefore
handles 20 models, not all 160. Replace `standard` with `limited` for the stress
test. Checkpoints and per-seed metrics permit interruption and resume across
sessions and Google Drive-backed result directories.

`--capacity` accepts `standard`, `limited`, or `all` (the full-run default).
`--skip-generation` trains and evaluates score quantities while leaving generation
metrics missing. A later mutually exclusive `--generation-only` run loads the
completed checkpoint, performs no optimizer step, computes only W2/mean/covariance
metrics, and updates the same row without erasing score fields. Changing
evaluation sample counts does not change checkpoint identity or retrain a model;
it does define a distinct evaluation design for scientific aggregation.

Training writes only to `results_rotation/metrics/seed_NNN.csv`. Each file is
protected by a file lock and atomically replaced, so non-overlapping seed shards
can safely share one results directory; an accidental same-seed overlap cannot
corrupt the CSV. `analyze_results.py` reads all per-seed files and produces the
read-only consolidated artifact `results_rotation/metrics.csv`. The identity
hierarchy separates training design, checkpoint, complete evaluation design,
seed pair, and model setting. `design_manifest.csv` exposes all unhashed fields.
Analysis groups by `design_id`, capacity, and rotation, so mixed pilot/formal or
evaluation configurations cannot form a false 20-seed result. Score and sample
completeness are assessed independently.

The full prespecified grid remains 120 joint models plus 40 common target-only
models (160 total). Generation evaluation may be deferred, but formal conclusions
require all 20 seeds for both capacities. Generated samples are not saved by
default; only aggregate metrics are written.
