# Conditional Transfer Diffusion

This repository contains two independent experiment modules that study transfer in diffusion models from complementary angles.

## GMM simulation

[`gmm_simulation/`](gmm_simulation/) links to two self-contained studies: the relocated, scientifically unchanged AR(1)-Toeplitz experiment and a spectrum-matched covariance-rotation experiment. The latter isolates principal-direction mismatch while holding covariance eigenvalues fixed.

## Image experiment

[`image_experiment/`](image_experiment/) contains the ImageNet/CIFAR transfer pipeline, staged experiment configurations, runtime evidence probes, release validator, tests, and cluster workflow.

```bash
cd image_experiment
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r environment/requirements-image-lock.txt
python -m pytest -q
```

The release readiness record remains `not_run` until the exact cluster environment, GPU load and resume probes, and real-data pilot all pass validation.

## Repository layout

```text
.
├── .github/              # module-specific continuous integration
├── gmm_simulation/       # original AR(1) and spectrum-rotation studies
├── image_experiment/     # real-image transfer pipeline
└── README.md             # repository navigation
```

The modules are operationally separate while supporting the same research question. Run commands from the module directory shown above so imports and relative configuration paths remain isolated.
