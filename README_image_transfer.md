# Image-transfer diffusion experiments

This module is a research-oriented framework for studying positive and negative transfer in class-conditional image diffusion models. It is isolated from, and does not change, the Gaussian-mixture simulation pipeline in the repository.

No real ImageNet experiment results are included or claimed here. The supplied ImageNet files are configuration templates; running a full grid requires separately provided data, reviewed class groups, GPU compute, and an analysis of the resulting per-run records.

## Model architecture

New image experiments use a pixel-space, epsilon-predicting `adm_unet`. A convolutional U-Net is a natural fit for dense 32×32/64×64 noise prediction, follows the DDPM/ADM literature, and is substantially more practical than a DiT for repeated low-data grids. The reverse variance remains fixed: this change is architectural and does not alter the diffusion objective.

The ADM-style model has two residual blocks per resolution in the main profile, GroupNorm/SiLU, sinusoidal time embeddings, class embeddings, adaptive GroupNorm scale/shift conditioning, multi-head self-attention, residual up/downsampling, dropout, and zero-initialized residual and output projections. Upsampling uses nearest-neighbor resize followed by convolution; downsampling uses residual stride convolutions. Decoder skip shapes are exact by construction—interpolation is never used to repair a mismatch. `legacy_simple_unet` remains available for historical YAML/checkpoint reproduction, with a deprecation warning when an old YAML omits `model.architecture`.

| Profile | Base channels | ResBlocks/level | Attention | Dropout | Parameters (unconditional) | Intended use |
|---|---:|---:|---|---:|---:|---|
| `smoke_tiny` | 16 | 1 | none at encoder/decoder resolutions | 0.0 | 219,219 | CPU tests/FakeData plumbing only |
| `pilot_small` | 40 | 2 | 16×16 | 0.1 | 10,549,043 | ImageNet64 pipeline and hyperparameter pilot |
| `main_default` | 64 | 2 | 16×16, 8×8 | 0.1 | 28,289,923 | all primary study conditions |
| `capacity_large` | 96 | 2 | 16×16, 8×8 | 0.1 | 63,604,035 | limited capacity sensitivity only |

`main_default` is fixed across every `n0`, target, auxiliary composition, seed, and target-only/conditional comparison. Model capacity is never selected from the sample size or semantic distance. The small `imagenet64_capacity_sensitivity.yaml` design treats profile as an explicit factor for only `n0 ∈ {50,100}`, target-only/close/far, and three seeds.

Conditional and unconditional models construct all shared modules before initializing the class embedding under a separate RNG stream. Thus every common name-and-shape backbone tensor is exactly equal at a shared initialization seed. The one-label conditional model is the primary architecture-matched control; the unconditional model remains a secondary scientific comparison.

No pretrained diffusion model, VAE, encoder, latent diffusion, DiT, classifier guidance, or classifier-free guidance is used. Class dropout is fixed at zero and sampling guidance scale is implicitly one. This ensures that any cross-class sharing is learned only from the experiment's auxiliary data. Pretrained networks used strictly for post-training metrics do not initialize or guide the diffusion model.

With only 50–500 independent target images, excess capacity primarily raises overfitting and memorization risk rather than classical underfitting risk. Diagnose this using the train/validation denoising gap together with generated-to-target-train versus generated-to-holdout nearest-neighbor summaries; do not use a parameters-per-image ratio to select architecture automatically.

The checked-in ImageNet main design currently has French bulldog as its sole target. The model supports arbitrary targets and class counts, but expanding the target set is a separate, researcher-reviewed experimental-design decision. Appropriate reporting language is: “Results are observed under a standard ADM-style U-Net and are tested for limited capacity sensitivity.” The architecture does not directly identify the theoretical shared-specific decomposition.

Inspect any resolved model without training:

```bash
python -m image_transfer.scripts.inspect_model \
  --config image_transfer/configs/imagenet64_main_grid.yaml \
  --out model_audit.json
```

The implementation is original repository code informed by the architectural design in *Diffusion Models Beat GANs on Image Synthesis* (Dhariwal and Nichol, 2021) and *Improved Denoising Diffusion Probabilistic Models* (Nichol and Dhariwal, 2021); it does not copy the OpenAI `guided-diffusion` source.

## Research question

Fix a target class (c_0). The central comparison is between:

1. a target-only diffusion model trained on target images; and
2. a class-conditional diffusion model trained on the same target images plus images from auxiliary classes, sampled and evaluated only at target label (c_0).

The experiments ask when auxiliary classes improve target generation, when they cause negative transfer, and how transfer changes with target sample size, auxiliary sample size, number of auxiliary classes, and independently measured auxiliary similarity.

The primary baselines are one-label conditional U-Nets. They match the parameterization of the multi-class conditional U-Net and isolate transfer from the architectural difference between conditional and unconditional models. Unconditional models remain as legacy comparisons.

## Experimental designs and baselines

Let `n0` be the number of target training images, `K_aux` the number of auxiliary classes, and `m_per_aux` the number of images from each auxiliary class. Thus

```text
N_aux   = K_aux * m_per_aux
N_total = n0 + N_aux
```

### Experiment A: equal target sample size

All paired models use the same `n0` target images.

- `conditional_close`, `conditional_medium`, `conditional_far`, and `conditional_mix` use `n0` target images plus `m_per_aux` images from each of `K_aux` auxiliary classes.
- Primary baseline: `conditional_target_only_n0`, a one-label conditional `ImageUNet` trained on only the same `n0` target images. Every input label is zero.
- Legacy baseline: `unconditional_n0`, trained on only the same `n0` target images.

This is the main target-scarce transfer comparison. A positive transfer gap means that adding auxiliary classes improves the target metric relative to the relevant target-only baseline.

### Experiment B: equal total sample size

Each auxiliary conditional model uses `n0` target images and `K_aux * m_per_aux` auxiliary images. Its target-only baseline instead uses `N_total` target images.

- Primary baseline: `conditional_target_only_equal_total`, the same conditional architecture with one label and `N_total` target images.
- Legacy baseline: `unconditional_equal_total`, an unconditional model with `N_total` target images.

The target evaluation set remains fixed across all models. Feasibility is checked only after the manifest has reserved target validation and evaluation images. If fewer than `N_total` target training candidates remain, strict runs fail or are explicitly recorded as skipped according to `data_split.insufficient_data_action`; the code does not shrink or replace the holdout set.

### Experiment C: auxiliary similarity and composition

Experiment C varies a predeclared auxiliary composition, such as `close_only`, `mostly_close`, `balanced_mix`, `mostly_far`, or `far_only`, while holding the intended target and auxiliary budgets fixed. The primary and legacy baselines are the same as in Experiment A.

The grid also supports two complementary sensitivity analyses:

- fix `K_aux` and vary `m_per_aux` or `auxiliary_ratio_values`; and
- fix a divisible `total_auxiliary_budget` and vary `K_aux`.

Auxiliary classes are drawn reproducibly from frozen candidate groups using `aux_draw_seed` and `aux_draw_id`. The grid records the exact synsets and the number of unique combinations available. If only one unique combination exists, it creates one draw rather than treating duplicate class sets as independent repetitions.

Target-only controls are also deduplicated by the quantity that actually changes their training data. Experiments A/C emit one control per target, `n0`, seed tuple, and protocol; Experiment B emits one per target exposure `n0 + m_per_aux * K_aux`. Aggregation broadcasts that single predeclared control to every compatible auxiliary composition or `m`/`K` factorization. Repeatedly training an identical control is never counted as extra replication.

Experiments B and C are not launched automatically after A. Generate and inspect their job grids separately.

## Training protocols

Every job records its protocol and actual exposure counts, including optimizer steps, target examples seen, auxiliary examples seen, total examples seen, effective target fraction, batch sizes, loss weight, and wall-clock time.

### `natural_compute_matched`

- All models use the same optimizer-step count and nominal total batch size.
- Conditional models sample naturally from the pooled target-plus-auxiliary dataset.
- The expected target fraction is determined by the pooled class composition.
- This controls nominal training compute, but a conditional model may see fewer target examples than a target-only model.

### `target_exposure_matched`

- A target-only model sees `B_target` target images per optimizer step.
- Its paired conditional model also sees `B_target` target images per step and additionally sees `B_aux` auxiliary images, balanced approximately across auxiliary classes.
- The loss is explicit:

```text
loss = target_loss + auxiliary_loss_weight * auxiliary_loss
```

This protocol controls target exposure but gives the conditional model additional examples and therefore additional compute. It must not be described as compute-matched. Report the two protocols separately.

## Five independent random seeds and paired randomness

Jobs split randomness into five recorded streams:

- `data_split_seed`: manifest partitions and nested subset ordering;
- `model_initialization_seed`: model weights;
- `training_seed`: data order, training timesteps, and training noise;
- `sampling_seed`: initial and reverse-process sampling noise;
- `evaluation_seed`: fixed corruption banks.

Within a paired comparison, models use the same target manifest, sampling seed, number of generated images, sampling batch size, sampler, and sampling steps. This common-random-number design reduces paired Monte Carlo noise. Sampling uses its own explicit `torch.Generator`, so its output does not depend on how much randomness training consumed.

Conditional and unconditional construction also preserves matched initialization for all common, same-shaped backbone parameters at a shared model-initialization seed. The class embedding is initialized separately.

The real CIFAR pilot has three paired repetitions. Main experiments should use at least five paired seeds; the ImageNet main template provides ten as the recommended final design.

## Fixed data manifests and leakage prevention

Before any model-specific training subset is built, the data layer creates a schema-versioned JSON manifest identified by the dataset, target, experiment family, and `data_split_seed`. It records:

- a dataset fingerprint and manifest SHA256;
- target evaluation and validation references;
- an ordered target training candidate pool;
- train/evaluation candidate pools for every frozen auxiliary class;
- requested and actual split sizes and explicit feasibility information.

For `eval_source: train_holdout`, target evaluation images are reserved first, target validation images second, and only the remainder becomes eligible for training. For an official test source, evaluation and validation are disjoint subsets of that source and training remains in the official training split. Target train, validation, and evaluation references never overlap.

The shuffled target candidate order is fixed by the split seed. Consequently, when `nested_training_subsets: true`, the `n0=50` subset is a prefix of `n0=100`, which is a prefix of `n0=250`, and so on. For a fixed target, experiment family, and split seed, changing `model_type`, `aux_set`, training protocol, or `n0` does not change the target validation or evaluation set. Experiment A models use exactly the same `n0` target references.

In strict mode, an undersized requested holdout raises a clear error or produces an explicit skip record. It never silently reduces the holdout or chooses replacement reference images. Real-feature caches include the manifest hash, feature extractor identity, preprocessing, image size, and metric schema, so incompatible reference features cannot be reused.

Training images used for nearest-neighbor checks are read through deterministic evaluation transforms while retaining the exact manifest training indices; stochastic training augmentation is not used for memorization comparisons.

## Fixed denoising evaluation

Denoising evaluation uses persisted corruption banks rather than drawing fresh timesteps and noise for each model. A bank is keyed by the manifest hash, evaluation seed, diffusion horizon, corruptions per image, noise-bin definition, and schema version. Paired models therefore see the same images, timesteps, and epsilon noise.

- The validation bank is used for checkpoint selection.
- The final test/evaluation bank is used only for final evaluation and never selects a checkpoint. Main transfer summaries and denoising plots use these `test_epsilon_mse_*` fields; validation fields remain checkpoint-selection diagnostics.
- Overall epsilon MSE follows the declared uniform-timestep distribution; it is not an unweighted average of unequal-width low/mid/high bins.
- MSE is averaged over image dimensions, then aggregated by image to obtain an image-clustered standard error.
- Low-, mid-, and high-noise results are reported separately.

Key fields include `validation_epsilon_mse_target`, `test_epsilon_mse_target`, their three noise-bin MSEs and clustered standard errors, image/corruption counts, and separate validation/test corruption-bank hashes.

## Generation metrics

No single metric is treated as sufficient. Main tables should report complementary quality, diversity, semantic, and memorization diagnostics.

| Metric | Interpretation and implementation rule |
|---|---|
| Target KID | `kid_target_mean` and `kid_target_std`; subset size and number of subsets are configured. It is explicitly unavailable when the sample count is too small. |
| Target FID | `fid_target`; retained as a standard compatibility metric, not the only primary outcome. |
| PRDC | `precision_target`, `recall_target`, `density_target`, and `coverage_target`, using the same reusable features as FID/KID and batched distance computation. |
| Semantic fidelity | `classifier_target_top1_acc`, `classifier_target_top5_acc`, and a top-1 histogram. ImageNet uses an exact reviewed synset-to-index mapping, not fuzzy class-name matching. |
| Auxiliary leakage | `auxiliary_leakage_rate`, the fraction of generated target samples classified as one of the auxiliary classes. |
| Denoising | Final test-bank overall and low/mid/high target epsilon MSE with image-clustered uncertainty; validation-bank values are checkpoint-selection diagnostics. |
| Memorization | Generated-to-target-train, target-holdout, auxiliary-train, and auxiliary-holdout nearest-neighbor summaries, near-duplicate counts/rates, and deterministic grids. |

The metric metadata records backend and package versions, feature extractor and dimensionality, input range, resize/interpolation/antialias behavior, manifest/cache keys, and real/generated sample counts.

### Strict mode and debug mode

Set `evaluation.mode` explicitly:

- `strict`: a missing or failed FID/KID/feature backend raises an error. There is no mathematical fallback under the FID or KID field names.
- `debug`: a fast pooled-pixel diagnostic may be emitted as `debug_pooled_pixel_distance`. It is never written to `fid_target` or `kid_target_mean`.

Before training begins, strict mode initializes the configured Inception, exact-mapping classifier, and diagnostic feature backends. Missing weights or mappings therefore fail before an expensive model fit. A custom complete ImageNet mapping can be supplied with `evaluation.classifier_synset_mapping_path`; unknown target or auxiliary synsets are errors in strict strict mode.

The real CIFAR pilot deliberately disables classifier fidelity because the repository does not bundle a version-pinned CIFAR-10 classifier. It never substitutes an ImageNet classifier. ImageNet classifier metadata includes architecture, weights, preprocessing, and exact mapping status.

### Why Inception Score is off by default

`compute_inception_score: false` is the default. Inception Score is poorly aligned with a single-target experiment: label diversity is intentionally absent, and the score does not compare generated samples with the fixed target distribution. It is therefore excluded from the default main table. Enable it only for an explicitly designed, label-balanced multi-class generation evaluation.

CMMD is an optional future sensitivity analysis (`compute_cmmd: false`), not a required metric in this implementation. The project does not add a large unstable dependency solely to report CMMD.

## Samplers and common random numbers

Sampling configuration is explicit:

```yaml
sampling:
  sampler: ddim
  steps: 50
  batch_size: 64
  ddim_eta: 0.0
```

- `ddpm` implements adjacent ancestral transitions and requires `sampling.steps == diffusion.timesteps`. A skipped-timestep DDPM request fails with an explanatory error.
- `ddim` implements the actual respaced DDIM update, supports `2 <= steps <= diffusion.timesteps`, and records `ddim_eta`.

Every result records sampler, step count, eta, sampling seed, and sampling batch size. Paired models use the same initial-noise stream.

## Checkpoints and exact resume

Each run writes canonical `<run_id>_last.pt` and `<run_id>_best.pt` checkpoints. Best means lowest fixed target-validation denoising MSE; test FID, KID, or other final metrics never select a checkpoint.

Checkpoints separately contain raw model state, EMA state, optimizer state, GradScaler state, current step, best validation metric, Python/NumPy/torch CPU/torch CUDA RNG states, config hash, manifest hash, git SHA, training protocol metadata, and resumable data-loader state. Resume restores raw and EMA models independently together with optimizer, scaler, RNG, and data progress; it does not load EMA weights into the raw model while retaining stale raw-model optimizer moments.

The test suite verifies bitwise-equivalent resume in deterministic CPU mode with `num_workers: 0`. The checked-in reproducible pilot/main templates use `num_workers: 0` for this reason. CUDA kernels can still be nondeterministic even though all required checkpoint state is restored; record such settings and do not claim bitwise identity unless it has been verified on the exact platform.

Use `--resume` on a job worker:

```bash
python -m image_transfer.scripts.run_job \
  --jobs-csv "$JOBS" \
  --job-index 0 \
  --device cuda \
  --resume
```

A checkpoint alone does not mark a run complete. A worker skips only when its schema-valid per-run result JSON exists. Use `--force` to intentionally rerun a completed job.

## Concurrent-safe outputs

SLURM workers never append to a shared metrics CSV. Each worker atomically writes one result or failure record:

```text
$RESULTS_ROOT/
  manifests/
    <dataset>/*.manifest.json
  corruption_banks/
    *.json
  cache/
    real_features/
  A_equal_target/
    jobs/
    checkpoints/
    logs/
    samples/
    configs/
    cache/
    figures/
  B_equal_total/
    jobs/
    checkpoints/
    logs/
    samples/
    configs/
    cache/
    figures/
  C_similarity_sweep/
    jobs/
    checkpoints/
    logs/
    samples/
    configs/
    cache/
    figures/
  run_results/
    <run_id>.json
  failures/
    <run_id>.json
  all_metrics.csv
  summary_metrics.csv
  paired_transfer_gaps.csv
  job_completeness.csv
  failed_jobs.csv
  similarity_correlations.csv
  figures/
```

Failure JSON includes the exception type, message, traceback, config, job, git SHA, and timestamp. Run identifiers include the experiment, target, model type, auxiliary composition/draw, sample sizes, split/model/training seeds, protocol, sampler, sampling seed, evaluation seed, and config hash.

Do not commit datasets, checkpoints, generated samples, feature caches, cluster logs, private paths, or credentials.

## Aggregation, paired gaps, and plots

Aggregate only after workers have produced independent JSON files:

```bash
python -m image_transfer.scripts.aggregate_results \
  --results-root "$RESULTS_ROOT" \
  --expected-jobs "$JOBS"

python -m image_transfer.scripts.plot_results \
  --results-root "$RESULTS_ROOT" \
  --experiment A
```

`--expected-jobs` may be repeated for several grids. Aggregation validates result schemas, deduplicates by `run_id`, records invalid/missing/failed jobs, and writes the six CSV files shown above.

Primary pairing is:

| Experiment | Primary baseline | Legacy baseline |
|---|---|---|
| A | `conditional_target_only_n0` | `unconditional_n0` |
| B | `conditional_target_only_equal_total` | `unconditional_equal_total` |
| C | `conditional_target_only_n0` | `unconditional_n0` |

All reported transfer gaps use an improvement-positive convention:

```text
lower-is-better: improvement = baseline metric - auxiliary-model metric
higher-is-better: improvement = auxiliary-model metric - baseline metric
```

Thus positive always means the auxiliary model improved. Candidate rows retain their own `m_per_aux` and `K_aux`, while baseline matching uses experiment, target, `n0`, the effective target-only training count, split/model/training/sampling/evaluation seeds, protocol, sampler, sampling steps, effective run-spec hash, and config hash. This is what permits an identical baseline to be reused across compatible auxiliary factorizations without changing the estimand. The aggregator computes differences within seeds, averages auxiliary-set draws within the seed cluster, and only then summarizes across independent seed clusters using the mean, sample standard deviation, standard error, 95% t confidence interval, completed pair count, and missing/failed pair count.

Plots show paired improvement means with 95% intervals and a zero reference line. Raw paired points may be shown faintly, but raw values from different seeds are never sorted and connected into a false trajectory. Natural-compute-matched and target-exposure-matched runs remain distinct. Default figures cover transfer versus `n0`, auxiliary sample size, `K_aux`, and measured auxiliary similarity; final test-bank denoising by noise bin; semantic fidelity/leakage; density versus coverage; and fixed sample grids. Similarity analysis reports Spearman association and, where possible, a clustered bootstrap interval.

## Installation and tests

Install an appropriate PyTorch build first, then the image dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-image.txt
```

The default test suite does not download pretrained weights or datasets:

```bash
python -m pytest -q
```

## Offline CPU smoke test

`cifar10_fake_smoke.yaml` uses `FakeData`, two training steps, tiny images, debug metrics, and no network downloads. Its outputs verify plumbing only and must never be reported as CIFAR-10 or scientific results.

```bash
export RESULTS_ROOT=$(mktemp -d /tmp/conditional_transfer_fake_smoke.XXXXXX)
mkdir -p "$RESULTS_ROOT/A_equal_target/jobs"
JOBS="$RESULTS_ROOT/A_equal_target/jobs/cifar10_fake_A.csv"

python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/cifar10_fake_smoke.yaml \
  --out "$JOBS"

N_JOBS=$(($(wc -l < "$JOBS") - 1))
for JOB_INDEX in $(seq 0 $((N_JOBS - 1))); do
  python -m image_transfer.scripts.run_job \
    --jobs-csv "$JOBS" \
    --job-index "$JOB_INDEX" \
    --device cpu \
    --force
done

python -m image_transfer.scripts.aggregate_results \
  --results-root "$RESULTS_ROOT" \
  --expected-jobs "$JOBS"
python -m image_transfer.scripts.plot_results \
  --results-root "$RESULTS_ROOT" \
  --experiment A
```

## Real CIFAR-10 pilot

`cifar10_real_pilot.yaml` uses real CIFAR-10 (`use_fake_data: false`), dog and automobile targets, the official test source for fixed holdouts, three pilot seeds, and a deliberately small training template. It defaults to debug mode, disables FID/KID/classifier metrics, and labels any debug nearest-neighbor feature fallback explicitly; it is a pipeline and trend pilot, not a main-scale result.

```bash
export DATA_ROOT=/path/to/cifar10
export RESULTS_ROOT=/path/to/cifar10_pilot_results
mkdir -p "$RESULTS_ROOT/A_equal_target/jobs"
export JOBS="$RESULTS_ROOT/A_equal_target/jobs/cifar10_real_A.csv"

python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/cifar10_real_pilot.yaml \
  --out "$JOBS"

# Inspect the grid before launching any jobs.
python - <<'PY'
import csv, os
path = os.environ["JOBS"]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print("jobs:", len(rows))
print("model types:", sorted({row["model_type"] for row in rows}))
print("protocols:", sorted({row["training_protocol"] for row in rows}))
PY
```

## ImageNet data and main-grid template

ImageNet is never downloaded automatically. Place or symlink ILSVRC2012 data as:

```text
$DATA_ROOT/
  train/
    n02108915/
    ...
  val/
    n02108915/
    ...
```

Missing required synset directories fail explicitly. Symlinked subset directories are supported and avoid unnecessary copies.

The current pilot lists French bulldog and pre-existing close/medium/far groups only as a code template. A final study must include multiple target classes. Researchers must audit, pre-register, and freeze every target and auxiliary candidate list before examining transfer results; do not choose targets, groups, or draws after seeing FID/KID.

Generate—but do not launch—the main template:

```bash
export DATA_ROOT=/path/to/ILSVRC2012
export RESULTS_ROOT=/path/to/image_transfer_results
mkdir -p "$RESULTS_ROOT/A_equal_target/jobs"
JOBS="$RESULTS_ROOT/A_equal_target/jobs/imagenet64_main_A.csv"

python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/imagenet64_main_template.yaml \
  --out "$JOBS"

echo "job rows: $(($(wc -l < "$JOBS") - 1))"
```

Generate B and C only after confirming manifest split sizes, equal-total feasibility, frozen class groups, storage requirements, and expected GPU cost. Grid creation does not train models or claim that an experiment has been completed.

## Ginsburg workflow

Scratch storage is not backed up. Keep repository, data, results, logs, and model caches separate.

```bash
ssh "$USER@cluster.example.edu"
mkdir -p "/shared/projects/$USER/conditional_transfer_diffusion"/{repo,data,results,logs,cache}
cd "/shared/projects/$USER/conditional_transfer_diffusion/repo"
git clone https://github.com/bolun-li-stat/conditional_transfer_diffusion-.git .

module load anaconda/3-2023.09
conda create -n ctdiff_img python=3.11 -y
conda activate ctdiff_img
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-image.txt
```

Verify an interactive GPU environment before submitting an array:

```bash
srun --pty -t 0-01:00 --gres=gpu:1 -A stats /bin/bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Generate and inspect the desired grid, then submit exactly its zero-based rows:

```bash
export PROJECT_ROOT="/shared/projects/$USER/conditional_transfer_diffusion/repo"
export DATA_ROOT="/shared/projects/$USER/conditional_transfer_diffusion/data"
export RESULTS_ROOT="/shared/projects/$USER/conditional_transfer_diffusion/results"
export JOBS="$RESULTS_ROOT/A_equal_target/jobs/imagenet64_A_jobs.csv"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$JOBS")"
python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/imagenet64_main_template.yaml \
  --out "$JOBS"

LAST_INDEX=$(($(wc -l < "$JOBS") - 2))
test "$LAST_INDEX" -ge 0
sbatch --array=0-"$LAST_INDEX" \
  scripts/cluster/run_image_transfer_array.sh \
  "$JOBS"
```

Monitor with:

```bash
squeue -u "$USER"
tail -f "/shared/projects/$USER/conditional_transfer_diffusion/logs/ctdiff_img_<job>_<task>.out"
```

The main template is intentionally large. Do not submit it without estimating row count, GPU-hours, disk usage, and metric-weight downloads. A smaller reviewed pilot grid should be completed first.

## Configuration migration

Use the new explicit files for new runs:

- `cifar10_fake_smoke.yaml`: offline FakeData plumbing test; replaces the ambiguous fake `cifar10_sanity.yaml` name.
- `cifar10_real_pilot.yaml`: real CIFAR-10 pilot with three seeds and two targets; replaces `cifar10_real_sanity.yaml` for new work.
- `imagenet64_main_template.yaml`: strict-mode template with explicit manifests, two protocols, five RNG streams, and ten recommended seeds.

Older configuration keys remain useful for reading earlier runs, but migrate main jobs as follows:

| Legacy setting | Current setting |
|---|---|
| one overloaded `seed`/`seeds` | `data_split_seed`, `model_initialization_seeds`, `training_seeds`, `sampling_seeds`, `evaluation_seeds` |
| implicit pooled training | `training_protocols` and `training.protocol` |
| `evaluation.eval_split` | `data_split.eval_source` |
| implicit/model-specific holdouts | `data_split.manifest_root`, fixed sizes, nested subsets, and insufficiency action |
| `evaluation.compute_fid_kid` | separate `compute_fid` and `compute_kid` |
| `evaluation.compute_classifier` | `compute_classifier_fidelity` |
| top-level `sampling_steps` only | `sampling.sampler`, `sampling.steps`, `sampling.batch_size`, and `sampling.ddim_eta` |
| shared metrics CSV | atomic `run_results/<run_id>.json` followed by aggregation |

Always set `evaluation.mode` explicitly. Do not combine outputs from old and new split/metric definitions unless their manifests, preprocessing, and schemas are demonstrably compatible.

## Relationship to the shared-specific theory

The image experiments are phenomenon-first: they are designed to establish and characterize positive and negative transfer under controlled target scarcity, auxiliary quantity, auxiliary class count, and similarity mismatch.

The shared-specific decomposition supplies an explanatory framework in terms of a tradeoff between variance reduction from reusable structure and residual transfer bias from mismatch. A standard conditional U-Net does not explicitly identify the theoretical shared and class-specific components. Empirical patterns may therefore be described as **consistent with** the theory's predictions, not as directly validating or recovering the decomposition.

## Non-goals of this implementation

This implementation does not:

- add explicit shared-specific adapters, LoRA modules, or correction-capacity sweeps;
- claim that a standard conditional U-Net identifies shared and specific score components;
- implement a complete adaptive auxiliary-class selector;
- require CMMD or make Inception Score a single-target primary metric;
- run or claim completion of the full ImageNet grid;
- provide ImageNet data, generated samples, checkpoints, caches, or experimental results;
- invent additional ImageNet semantic groups without researcher review.

Before a main run, freeze multiple reviewed target classes, their auxiliary candidate pools, split sizes, protocols, metrics, seed plan, and planned exclusions. Preserve the generated job grids and manifests as part of the experimental provenance.
