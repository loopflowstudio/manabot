# gpu-ops: stage 1 — GPU Docker Image

First stage of the gpu-ops wave. Get the Docker image right before building
provisioning on top of it.

## What to build

Update `ops/Dockerfile` to produce a GPU-capable training image. The image runs on
DLAMI hosts with NVIDIA Container Toolkit. Host provides drivers, container provides
CUDA runtime + PyTorch + manabot + managym.

## Data structures

```yaml
# ops/specs/base.yaml — shared machine config referenced by sandbox/job specs
ami: ami-0xxxxxxxxxxxxxxxxx   # Ubuntu 22.04 DLAMI, us-east-1
docker_image: manabot:latest   # local build for now, ECR later
cuda_version: "12.4"
```

## Key changes

1. **`ops/Dockerfile`** — rewrite with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base
   - Multi-stage: build stage (Rust + managym) → runtime stage (slim)
   - PyTorch with CUDA from `--extra-index-url https://download.pytorch.org/whl/cu124`
   - CMD defaults to train.py but allows override for interactive use

2. **`ops/specs/base.yaml`** — pin DLAMI AMI and CUDA version

3. **`ops/machine.sh`** — update or replace: NVIDIA Container Toolkit setup for DLAMI

## Constraints

- Image must work for both sandbox (interactive, override CMD) and job (default CMD)
- Must not bake in any AWS account references
- Rust build is slow — use multi-stage to cache the build layer
- PyTorch CUDA version must match the DLAMI driver version

## Done when

- `docker build -f ops/Dockerfile .` succeeds locally (CPU, no GPU test)
- On a DLAMI g5.xlarge: `docker run --gpus all <image> python -c "import torch; print(torch.cuda.is_available())"` → `True`
- On a DLAMI g5.xlarge: smoke test training run completes
