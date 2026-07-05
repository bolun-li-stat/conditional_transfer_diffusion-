# Conditional vs Unconditional DDPM Transfer on Gaussian Mixtures

This repository implements a clean simulation project for studying **positive and negative transfer in class-conditional diffusion models**.  The proposal PDF is stored at `docs/Conditioal_vs_Unconditional_Diffusion_Model__Proposal.pdf` and is used only as background context; the implementation follows the experiment specification in this repository.

## Research question

For a fixed target class/component 1, when does a multi-class class-conditional DDPM outperform a target-only unconditional DDPM, and when can auxiliary classes hurt target-class generation?

The comparison is between:

1. **Target-only unconditional DDPM**: trained only on Gaussian component/class 1 and modeled by `epsilon_theta(x_t, t)`.
2. **Multi-class conditional DDPM**: trained on all components/classes and sampled/evaluated only at target label `c = 1`, modeled by `epsilon_theta(x_t, t, c)`.

Both models use the same original DDPM epsilon-prediction objective, forward process, linear beta schedule, and reverse sampling equations. They differ only in training data and whether the MLP receives a learned class embedding.

## Repository structure

```text
README.md
requirements.txt
config.py
data.py
diffusion.py
unconditional_model.py
conditional_model.py
train.py
eval.py
plot_results.py
utils.py
notebooks/
  00_smoke_test.ipynb
  01_experiment_low_target_data.ipynb
  02_experiment_same_total_budget.ipynb
  03_plot_and_analyze_results.ipynb
results/
  .gitkeep
docs/
  Conditioal_vs_Unconditional_Diffusion_Model__Proposal.pdf
```

## Data-generating process

The synthetic data are `d = 100` dimensional Gaussian mixtures with `K = 3` components by default. Component 1 is the target component. The target mean is zero and auxiliary means are random unit directions scaled to Euclidean separation `Delta = 20` with a minimum pairwise distance check.

Covariances are AR(1)-Toeplitz matrices:

```text
Sigma(rho)_{ij} = rho^{|i-j|}
```

Two covariance scenarios are implemented:

* **Scenario A / shared covariance**: all components share `rho = 0.2`, `rho = 0.0`, or `rho = -0.2`.
* **Scenario B / covariance mismatch**: target has `rho = 0.1`; level `0` is a homogeneous `rho = 0.1` sanity check, and auxiliaries for `mild`, `medium`, or `strong` are sampled in balanced lower/upper intervals around 0.1.

## Experiments

### Experiment 1: low-target-data transfer

The unconditional model is trained only on the target component. The conditional model uses the **same target samples** plus auxiliary data. Defaults:

* `n_target_train in {200, 500, 800}`
* `n_aux_train = n_target_train` for each auxiliary class
* seeds `{0, ..., 19}`

This tests whether auxiliary classes help when target data are scarce.

### Experiment 2: same-total-training-budget

The conditional model has `K*n` total samples with only `n` true target samples. The unconditional model also has `K*n` total samples, all from the target component. Defaults:

* `n in {200, 500, 800}`
* conditional target subset is drawn from the same larger target pool used for the unconditional target-only training set whenever possible
* seeds `{0, ..., 19}`

This stronger benchmark tests whether auxiliary samples can replace true target samples.

## Metrics

The **primary metric** is integrated target-class score-estimation risk:

```text
E_{t, x_t | c=1} ||s_hat(x_t, t) - s_true(x_t, t)||^2
```

Because the target distribution is Gaussian, `s_true` is computed analytically for the noisy target marginal.

Auxiliary metrics are:

* target validation epsilon MSE
* generated-sample mean error
* relative covariance error
* Gaussian `W2^2`
* RBF-kernel MMD

All metrics are saved to CSV under `results_T1000_K3/`.

## Installation

```bash
python -m pip install -r requirements.txt
```

For GPU runs, install a PyTorch build compatible with your CUDA environment if the default package resolver does not select one.

## Quick smoke test

The smoke test is intentionally small and verifies the end-to-end pipeline.

```bash
python train.py --experiment smoke
python plot_results.py --results-dir results_T1000_K3 --make-pca
```

For an even faster local check:

```bash
python train.py --experiment smoke --training-steps 5 --n-generated 64 --score-risk-mc-samples 64 --force
```

## Running Experiment 1

```bash
python train.py --experiment low_target_data --training-steps 20000 --resume
```

Useful options:

```bash
python train.py --experiment low_target_data \
  --seeds 0 1 2 \
  --sampling-mode balanced \
  --results-dir results_T1000_K3 \
  --resume
```

## Running Experiment 2

```bash
python train.py --experiment same_total_budget --training-steps 20000 --resume
```

## Running all experiments

```bash
python train.py --experiment all --training-steps 20000 --resume
```

An optional extended run can use `--training-steps 50000` if results are unstable or inconclusive.

## Generating plots

```bash
python plot_results.py --results-dir results_T1000_K3
```

If generated samples were saved with `--save-samples`, include PCA plots:

```bash
python plot_results.py --results-dir results_T1000_K3 --make-pca
```

Figures are written to `results_T1000_K3/figures/`. Aggregated metrics are written to `results_T1000_K3/aggregated_metrics.csv`; additional summaries are written by sample size/model and covariance setting/model.

## Checkpoints and resume

By default, the code saves last checkpoints and training logs under:

```text
results_T1000_K3/checkpoints/
results_T1000_K3/logs/
results_T1000_K3/configs/
```

Use `--resume` to continue from saved checkpoints and skip settings that already have final metrics. Partial CSV rows are appended after each trained model/setting so that completed work is not lost if a Colab runtime disconnects.

Use `--force` to rerun settings even if matching metrics already exist.

## Google Colab instructions

The notebooks in `notebooks/` are designed to be opened and run in your own Colab Pro session. Do **not** share passwords, Google credentials, or tokens with the code.

Recommended Colab workflow:

1. Open the desired notebook in Colab.
2. Manually select a GPU runtime: **Runtime -> Change runtime type -> GPU or Premium GPU**.
3. Optionally mount Google Drive if you want persistent outputs.
4. Run the device-check cell. The code prints `torch.cuda.is_available()` and, when available, `torch.cuda.get_device_name(0)`.
5. The training code automatically uses CUDA when available and otherwise falls back to CPU.
6. Save outputs frequently under Drive-backed `results_T1000_K3/` if you are worried about disconnects.
7. Use checkpoint resume and partial CSV skipping to continue later.

Codex can generate and test this project in its own coding environment, but running the full GPU experiment in Google Colab requires you to open the notebook or repo in your own Colab session and manually choose an available GPU runtime. The code automatically uses the best available CUDA device in that session, but the exact GPU assigned by Colab cannot be guaranteed.

## Interpreting results

The main CSV has the requested columns:

```text
experiment_type,covariance_scenario,rho,mismatch_level,target_rho,
auxiliary_rhos,sqrt_alpha_bar_T,K,d,Delta,n,n_target_train,
n_aux_train,seed,model_type,sampling_mode,training_steps,
score_risk,validation_epsilon_mse,mean_error,covariance_error,
gaussian_w2_squared,mmd_rbf,final_train_loss,checkpoint_path,figure_dir
```

For transfer gaps, compute:

```text
score_risk_conditional - score_risk_unconditional
```

Values below zero indicate that the conditional model is better on target score risk. Values above zero indicate negative transfer or that the target-only unconditional model is better.

If negative transfer is not observed, try stronger mismatch, smaller target sample sizes, smaller model capacity, natural class sampling, larger `K`, or extended training.


### Colab sharding

The full default grid has 2 experiments, 20 seeds, 3 sample sizes, and 7 covariance settings (840 settings before the conditional/unconditional model pair). Split full runs into seed shards on Colab and write to a Google Drive-backed `results_T1000_K3` directory with `--resume`, for example `python train.py --experiment all --seeds 0 1 2 3 4 --training-steps 20000 --results-dir /content/drive/MyDrive/conditional_transfer_diffusion_results_T1000_K3 --resume`.
