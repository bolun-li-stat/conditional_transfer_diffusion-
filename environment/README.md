# Image-transfer environments

The CPU lock was exercised with Python 3.12.13, torch 2.7.1+cpu and torchvision 0.22.1+cpu. It pins the direct dependencies and the critical transitive dependencies selected by that tested environment. The CUDA file pins the intended torch/CUDA build but must be validated on the target GPU cluster before a real-data run. Its presence is not evidence that a CUDA run completed.

For CPU setup, create a clean Python 3.12.13 environment and run:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r environment/requirements-image-lock.txt
python -m image_transfer.scripts.inspect_environment \
  --lock environment/requirements-image-lock.txt
```

Alternatively, from the repository root run `conda env create -f environment/image-transfer-cpu.yml`;
the YAML supplies the CPU wheel index and the repository-relative exact lock path.

For the intended GPU setup, create `environment/image-transfer-cuda.yml` on the target cluster, then audit that YAML directly with `inspect_environment`. Do not substitute the CPU lock when generating GPU grids. Every preflight and run records the selected definition's SHA256. Changing any version therefore changes the environment identity and prevents unmarked cross-environment pairing.
