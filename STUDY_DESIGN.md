# Image-transfer study design

## Question and estimand

The study asks when predeclared auxiliary classes improve or harm generation for a target class under target-data scarcity. Its primary comparison is a target-plus-auxiliary conditional ADM U-Net against an architecture-matched one-label conditional target-only ADM U-Net. The unconditional model is a secondary historical comparison and is disabled in the sparse checked-in ImageNet designs.

This is a phenomenon-first design. The same pixel-space, epsilon-predicting, fixed-reverse-variance model family is used across main conditions. It does not identify theoretical shared or class-specific components, use pretrained generative weights, choose capacity from sample size, or use classifier guidance. Findings may be consistent with a proposed mechanism; they do not by themselves recover or validate that mechanism.

`main_default` is fixed for the main phenomenon grid. `smoke_tiny` is plumbing-only, `pilot_small` supports limited pilots, and `capacity_large` appears only in the disabled capacity check. The one-label conditional baseline preserves class-conditioning parameters and is therefore the primary control for transfer.

## Target decisions

Target and auxiliary groups are researcher inputs, not outcomes selected by the code. A main-stage target set must be reviewed, frozen, contain at least four targets, and span more than one supercategory. The checked-in main target file is intentionally empty and unreviewed. The French-bulldog file supports engineering and single-target case-study checks, but it is also marked unreviewed/unfrozen and cannot satisfy the main gate.

The target-set content, analysis plan, model configuration, resolved run configuration, and environment definition are hashed into provenance and pairing keys. Changing any of them changes run identity.

## Six random streams and manifest v2

Each job records six separate random streams:

1. `holdout_seed` fixes validation/evaluation partitions.
2. `training_subset_seed` fixes nested target and auxiliary training orders.
3. `model_initialization_seed` fixes weights.
4. `training_seed` fixes loader order, diffusion timesteps, and training noise.
5. `sampling_seed` fixes initial and reverse-process sampling noise.
6. `evaluation_seed` fixes persisted corruption banks.

Auxiliary-set draw randomness is a recorded sensitivity factor, not an independent target repetition.

Manifest schema v2 separates a split manifest from a subset manifest. The split identity covers the dataset fingerprint, target and auxiliary evaluation pools, holdout seed, sizes, and evaluation source. The subset identity covers the split hash, training-subset seed, and nested candidate orders. Changing experiment family, model type, protocol, or training-subset seed cannot change a fixed holdout. Within one subset order, target `n0=50` is a prefix of 100, then 250, then 500. Target train/validation/evaluation references are disjoint, as are auxiliary train/evaluation references.

Paired models share split/subset identities, model initialization, sampling noise, evaluation corruptions, sampler settings, target-set version, and environment lock. Aggregation rejects ambiguous or provenance-incompatible baseline matches.

## Training protocols

`natural_compute_matched` fixes optimizer steps and total nominal batch size. The target-plus-auxiliary model samples from the pooled dataset, so it may receive less target exposure than the target-only model. Actual pooled, target, and auxiliary losses, batch sizes, example counts, effective rates, runtime, and peak memory are recorded.

`target_exposure_matched` gives the target-only and target-plus-auxiliary models the same target batch per step, then adds a balanced auxiliary batch and the declared auxiliary loss weight. It uses more examples and compute for the auxiliary model and must not be described as compute-matched. It is a smaller mechanism control, not part of the natural-protocol main grid.

Validation, test, and train-diagnostic corruption banks are distinct. Validation selects the best checkpoint; the test bank is used only for final outcomes; the train bank supports generalization-gap diagnostics. `checkpoint_interval` controls last-checkpoint writes independently of validation frequency. Exact CPU resume is tested with `num_workers: 0`; CUDA bitwise identity is not claimed without platform-specific verification.

## Sparse configurations

| Configuration | Design | Expected rows |
|---|---|---:|
| GPU micro-smoke | one target, `n0=100`, target-only/close/far, one natural repeat | 3 |
| Release pilot | same three models, natural and target-exposure protocols, two training subsets | 12 |
| Main template | four `n0` values, target-only/close/far/mix, natural protocol, five training subsets | 80 per target; 320 for the minimum four targets |
| Target-exposure control | `n0={50,100}`, target-only/close/far, three training subsets | 18 |
| Auxiliary-size check | one target, `n0=100`, `K_aux=3`, ratios 0.5/1/2, close/far | 21 |
| Fixed-budget K check | one target, `n0=100`, total auxiliary budget 300, `K_aux={1,3,5}`, close/far | 21 |
| Capacity check | `n0={50,100}`, target-only/close/far, three profiles, three training subsets | 54 |

The main and sensitivity experiments are disabled by default. The job-grid tool prints model/target/protocol/size/profile breakdowns, checkpoint and sample-storage estimates, and refuses oversized grids without an explicit override. The old main-grid filename is a compatibility template with the same target-freeze and readiness gates; it cannot launch an inline single-target substitute.

## Outcomes and diagnostics

Primary endpoints are final target test-bank epsilon MSE and target KID. KID is the primary generated-distribution metric. FID is retained for comparability, but about 500 real references is below the configured 1,000-sample reliability threshold, so counts and warnings must accompany it. Inception Score is disabled because label diversity is intentionally absent in target-conditioned generation.

Secondary outcomes include FID, precision/recall/density/coverage, exact-mapping classifier fidelity, auxiliary leakage, noise-bin MSE, train/validation/test gaps, and memorization diagnostics. Nearest-neighbor thresholds are calibrated from validation references only, then applied to test/generated comparisons. Generated-to-target-train, target-holdout, auxiliary-train, and auxiliary-holdout summaries remain separate.

All gaps are improvement-positive: baseline minus model for lower-is-better metrics and model minus baseline for higher-is-better metrics. Auxiliary draws are averaged within a seed cluster; optimization repeats are summarized within a training subset; subsets are summarized within target; targets form the highest level. When only one target exists, across-target hierarchical inference is unavailable.

Required aggregate artifacts include `all_metrics.csv`, `summary_metrics.csv`, `paired_transfer_gaps.csv`, `subset_level_summaries.csv`, `target_level_summaries.csv`, `hierarchical_summaries.csv`, `job_completeness.csv`, `failed_jobs.csv`, `environment_summary.csv`, and `readiness_summary.json`.

## Readiness boundary

Real strict runs require the exact locked environment and offline-verified metric assets. The 12-job release pilot must complete and pass the schema-v2 validator before a main-stage grid is generated. A passed status certifies engineering health and exact grid/result evidence only. The status must match its release-pilot config, model, environment, target-set and git provenance; the main target set is separately validated rather than equated with the pilot target.

The current checked-in status is `not_run`. No real ImageNet metric improvement, completed main grid, or general transfer conclusion is claimed.
