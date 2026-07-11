# Image-transfer environments

The CPU lock was exercised with Python 3.12.13, torch 2.7.1+cpu and torchvision 0.22.1+cpu. It pins the direct dependencies and the critical transitive dependencies selected by that tested environment. The CUDA file pins the intended torch/CUDA build but must be validated on the target GPU cluster before a real-data run. Its presence is not evidence that a CUDA run completed.

For CPU setup, create a clean Python 3.12.13 environment and run:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r environment/requirements-image-lock.txt
python -m image_transfer.scripts.freeze_environment \
  --source-spec environment/requirements-image-lock.txt \
  --out readiness/environment_exact_lock.json
python -m image_transfer.scripts.inspect_environment \
  --lock readiness/environment_exact_lock.json \
  --source-spec environment/requirements-image-lock.txt \
  --out readiness/environment_runtime_report.json
```

Alternatively, from the `image_experiment` module root run `conda env create -f environment/image-transfer-cpu.yml`; the YAML supplies the CPU wheel index and the module-relative tested requirements path.

For GPU setup, treat `environment/image-transfer-cuda.yml` as a source specification only. Activate the final target-cluster environment, generate an exact lock with `freeze_environment`, and pass that lock plus the source specification to `inspect_environment --require-cuda-exact`. Do not substitute the CPU lock when generating GPU grids. Every preflight and run records the source, exact-lock, and runtime hashes; changing any layer prevents unmarked cross-environment pairing.
