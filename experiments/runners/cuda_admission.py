"""Retain device admission evidence before allowing a scientific GPU workload.

Runs in a separate process, so diagnostic tensor operations cannot consume the
learner's random stream. An unknown CUDA error remains unknown; never fall back.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import stat

import torch


def diagnose(out: Path) -> None:
    devices = {
        str(path): stat.filemode(path.stat().st_mode)
        for path in sorted(Path("/dev").glob("nvidia*"))
    }
    version = Path("/proc/driver/nvidia/version")
    evidence: dict[str, object] = {
        "status": "starting",
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "driver": version.read_text() if version.exists() else None,
        "devices": devices,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    try:
        torch.cuda.init()
        evidence["device_count"] = torch.cuda.device_count()
        evidence["device_name"] = torch.cuda.get_device_name()
        if (
            torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090"
        ):
            raise ValueError("One RTX 5090 is required; no fallback")
        x = torch.ones((16, 16), device="cuda", requires_grad=True)
        loss = (x @ x).sum()
        loss.backward()
        torch.cuda.synchronize()
        if x.grad is None or not bool(torch.isfinite(x.grad).all()):
            raise ValueError("Nonfinite CUDA admission gradient")
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        out.write_text(json.dumps(evidence, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    diagnose(parser.parse_args().out)


if __name__ == "__main__":
    main()
