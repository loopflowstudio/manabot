# gpu-ops

GPU provisioning for manabot. Two scripts: sandbox (interactive SSH box) and job
(fire-and-forget training). AWS first, thin provider interface for future backends.

## What to build

### Provider layer (`ops/provider.py`, `ops/aws.py`)

```python
@dataclass
class MachineSpec:
    instance_type: str       # "g5.xlarge"
    region: str              # "us-east-1"
    disk_gb: int             # EBS volume size
    ami: str                 # DLAMI AMI ID
    spot: bool
    max_spot_price: str | None = None

@dataclass
class Machine:
    id: str                  # provider-specific instance ID
    public_ip: str | None
    status: str              # "running", "stopped", "terminated"
    tags: dict[str, str]

class Provider(Protocol):
    def create(self, spec: MachineSpec, tags: dict[str, str]) -> Machine: ...
    def wait_until_ready(self, machine: Machine, timeout: int = 300) -> Machine: ...
    def stop(self, machine: Machine) -> None: ...
    def terminate(self, machine: Machine) -> None: ...
    def list(self, tags: dict[str, str]) -> list[Machine]: ...
    def run_command(self, machine: Machine, command: str) -> str: ...
```

`AWSProvider` implements this with boto3. Creds from default chain (SSO works).
Tags every resource with `manabot:user` (from STS caller identity) and `manabot:role`
(sandbox/job). Security group created per-user on first use, allows SSH from caller IP.

### Sandbox (`ops/sandbox.py`)

```
python ops/sandbox.py                  # create or resume, SSH in
python ops/sandbox.py --stop           # stop instance (keep disk)
python ops/sandbox.py --start          # restart stopped instance
python ops/sandbox.py --terminate      # destroy everything
python ops/sandbox.py --status         # show state
```

Create flow: check for existing sandbox by tag → if running, SSH in → if stopped,
start and SSH in → if none, create from spec, run setup, SSH in.

Setup (user-data): install NVIDIA Container Toolkit, pull Docker image, clone repo.

Sandboxes are on-demand (not spot) — you don't want interruption during interactive work.

### Job runner (`ops/job.py`)

```
python ops/job.py --config simple      # launch training, self-terminates
python ops/job.py --list               # show running jobs
python ops/job.py --logs <job-id>      # tail logs
python ops/job.py --cancel <job-id>    # terminate early
```

Launch flow: create spot instance → run Docker container with train.py → on
completion, `shutdown -h now`. W&B handles checkpoints. WANDB_API_KEY passed as env var.

Jobs are spot by default. Interruption recovery = restart from last W&B checkpoint.

### GPU Dockerfile (`ops/Dockerfile`)

Rewrite with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base. Multi-stage build:
build stage (Rust toolchain + managym) → runtime stage (slim, PyTorch CUDA, manabot).
CMD defaults to train.py, overridable for sandbox use.

### Config (`ops/specs/`)

```yaml
# ops/specs/sandbox.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 100
spot: false
ami: ami-0xxxxxxxxxxxxxxxxx  # Ubuntu 22.04 DLAMI

# ops/specs/job.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 50
spot: true
max_spot_price: "1.50"
ami: ami-0xxxxxxxxxxxxxxxxx
```

## Key decisions

- Provider is a Protocol, not an ABC. No framework. ~20 lines.
- AWS creds from default boto3 chain. No manabot-specific credential config.
- Every resource tagged with user identity for multi-user disambiguation.
- DLAMI provides host drivers. Container provides CUDA runtime + app.
- No ECR required initially — build Docker image on the instance. ECR is a future optimization.
- SSM for run_command (no SSH keys needed for remote execution). SSH for interactive sandbox (more natural).

## Constraints

- No AWS account references in committed code
- PyTorch CUDA version must match DLAMI driver version
- Image works for both sandbox (interactive) and job (train.py entrypoint)
- `boto3` added as an optional dependency in pyproject.toml (`pip install -e ".[ops]"`)

## Done when

- `python ops/sandbox.py` creates a g5.xlarge and opens SSH
- Running it again reconnects to existing instance
- `python ops/sandbox.py --terminate` cleans up
- `python ops/job.py --config simple` launches spot training, self-terminates on completion
- `python ops/job.py --list` shows running jobs
- W&B shows completed run with checkpoints
- All works with `aws sso login` credentials
