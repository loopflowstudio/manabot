# 01: GPU Docker Image

Update the ops Dockerfile to work on GPU instances with NVIDIA Container Toolkit.

## What to build

A Dockerfile that builds a manabot training image capable of running on GPU. The
image assumes the host has NVIDIA drivers installed (DLAMI provides this). The
container uses `--gpus all` at runtime.

## Approach

- Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (or similar — check what
  matches current DLAMI driver versions)
- Install Rust toolchain, uv, Python 3.12
- Install managym (Rust extension) and manabot
- Include PyTorch with CUDA support
- Pin the DLAMI AMI ID in a config file so it's reproducible
- Test locally with `docker build` (CPU path) and on a GPU instance

## Key decisions

- We're building our own image rather than using DLAMI's pre-installed PyTorch
  because we need the Rust toolchain for managym. DLAMI just provides the drivers
  on the host.
- The image should work for both sandbox (interactive use) and job (train.py
  entrypoint) — use CMD for the default but allow override.

## Done when

- `docker build -f ops/Dockerfile .` succeeds
- On a g5.xlarge with DLAMI: `docker run --gpus all <image> python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- On a g5.xlarge: `docker run --gpus all <image> python manabot/model/train.py --config-name simple experiment.wandb=false train.total_timesteps=128` completes without error
