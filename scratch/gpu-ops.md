# gpu-ops

GPU provisioning for manabot. Two scripts: sandbox (interactive SSH box) and job
(fire-and-forget training). AWS first, thin provider interface for future backends.

## Review decisions locked

- Keep this as one coherent wave (provider + sandbox + job + Docker + specs).
- Expand provider contract now so CLI flows do not leak boto3 calls.
- Treat these as v1 non-negotiable: IAM/SSM preflight, CloudWatch logs, idempotent bootstrap,
  spot resume path.
- Do the full scope now, including startup-time optimization via prebuilt GPU image path
  (no on-instance Docker build in the hot path).
- Optimize for a single operator today; keep code public-safe and reusable, but do not add
  complexity purely for multi-contributor ergonomics.

## What to build

### Provider layer (`ops/provider.py`, `ops/aws.py`)

```python
@dataclass
class MachineSpec:
    instance_type: str       # "g5.xlarge"
    region: str              # "us-east-1"
    disk_gb: int             # EBS volume size
    ami: str                 # Ubuntu 22.04 DLAMI AMI ID
    spot: bool
    max_spot_price: str | None = None
    key_name: str | None = None

@dataclass
class Machine:
    id: str                  # provider-specific instance ID
    public_ip: str | None
    status: str              # "pending"|"running"|"stopped"|"terminated"
    tags: dict[str, str]

@dataclass
class CommandResult:
    command_id: str
    status: str              # "Pending"|"InProgress"|"Success"|"Failed"|...
    stdout: str
    stderr: str

class Provider(Protocol):
    def create(
        self,
        spec: MachineSpec,
        tags: dict[str, str],
        *,
        user_data: str | None = None,
        iam_instance_profile: str | None = None,
    ) -> Machine: ...
    def start(self, machine: Machine) -> Machine: ...
    def wait_until_ready(self, machine: Machine, timeout: int = 300) -> Machine: ...
    def wait_for_ssm(self, machine: Machine, timeout: int = 300) -> None: ...
    def stop(self, machine: Machine) -> None: ...
    def terminate(self, machine: Machine) -> None: ...
    def list(self, tags: dict[str, str]) -> list[Machine]: ...
    def run_command(self, machine: Machine, command: str, timeout: int = 3600) -> CommandResult: ...
    def logs(self, group: str, stream_prefix: str, limit: int = 200) -> list[str]: ...
```

`AWSProvider` implements this with boto3. Credentials from default chain (SSO works).
Resources are tagged with:

- `manabot:user` (from STS caller identity)
- `manabot:role` (`sandbox` or `job`)
- `manabot:managed=true`
- `manabot:job-id=<id>` for jobs

Provider-owned AWS setup (idempotent):

- security group (SSH only from caller public IP)
- keypair import from local `~/.ssh/id_ed25519.pub` when `key_name` is unset
- instance profile with SSM + CloudWatch Logs permissions
- CloudWatch log groups:
  - `/manabot/sandbox/<user>`
  - `/manabot/job/<user>`

If IAM creation is denied by account policy, fail with actionable instructions and allow
pointing to pre-provisioned role/profile names (private Terraform in `../studio` is optional,
not required for public-repo workflows).

### Sandbox (`ops/sandbox.py`)

```bash
python ops/sandbox.py                  # create or resume, SSH in
python ops/sandbox.py --status         # show state
python ops/sandbox.py --start          # restart stopped instance and SSH in
python ops/sandbox.py --stop           # stop instance (keep disk)
python ops/sandbox.py --terminate      # destroy sandbox
```

Flow:

1. Find machine by tags (`user + role=sandbox`).
2. If running: open SSH.
3. If stopped: `provider.start()` → wait ready → SSH.
4. If missing: create on-demand machine with sandbox user-data bootstrap, wait ready/SSM,
   verify bootstrap marker, SSH.

Bootstrap is idempotent via marker file (`/opt/manabot/bootstrap.v1.done`):

- install Docker + NVIDIA container toolkit
- pull prebuilt GPU runtime image
- clone/update repo

### Job runner (`ops/job.py`)

```bash
python ops/job.py --config simple      # launch spot training job
python ops/job.py --list               # show running/recent jobs
python ops/job.py --logs <job-id>      # tail CloudWatch logs
python ops/job.py --cancel <job-id>    # terminate early
python ops/job.py --resume <job-id>    # restart from last W&B checkpoint
```

Launch flow:

1. Create spot instance tagged with `job-id` and `wandb-run-id`.
2. User-data starts a systemd one-shot that runs training container.
3. Container logs stream to CloudWatch.
4. On success: terminate instance.
5. On spot interruption/failure: job status becomes resumable; `--resume` launches new spot
   machine with same W&B run metadata.

`WANDB_API_KEY` is passed from local env via SSM parameter injection at launch time.

### GPU Docker image (`ops/Dockerfile` + publish path)

- Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- Multi-stage: build Rust extension + runtime with PyTorch CUDA + manabot.
- Default CMD: `python manabot/model/train.py`.
- Add image publishing path so ops scripts pull a prebuilt image (`ghcr.io/...:<git-sha>`).
- Keep optional local build fallback for contributors without registry access.

### Config (`ops/specs/`)

```yaml
# ops/specs/sandbox.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 100
spot: false
ami: ami-0xxxxxxxxxxxxxxxxx
key_name: null

# ops/specs/job.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 50
spot: true
max_spot_price: "1.50"
ami: ami-0xxxxxxxxxxxxxxxxx
key_name: null

# ops/specs/runtime.yaml
image: ghcr.io/loopflowstudio/manabot-gpu:latest
fallback_build: true
log_group_prefix: /manabot
```

## Key decisions

- Provider remains Protocol-first, but now includes start/SSM/log operations needed by CLI.
- No account IDs or account-specific constants in committed code.
- Default AWS auth is boto3 default chain (`aws sso login` supported).
- Hot path uses prebuilt image pull; building on-instance is fallback only.
- SSM for remote commands; SSH for interactive sandbox.
- CloudWatch is the canonical log source for job observability.

## Constraints

- No AWS account references in committed code.
- v1 assumes your AWS principal can create/update IAM role/policy/instance profile and CloudWatch log groups for manabot-managed resources.
- Public repo remains runnable with `aws sso login` + AWS permissions; no private infra repo dependency.
- PyTorch CUDA must match DLAMI driver compatibility.
- Bootstrap must be idempotent and safe to rerun.
- `boto3` added as optional dependency in `pyproject.toml` (`pip install -e ".[ops]"`).

## Done when

- `python ops/sandbox.py` creates/resumes g5.xlarge and opens SSH.
- Re-running sandbox command reconnects without reprovisioning.
- `python ops/sandbox.py --terminate` cleans up managed sandbox resources.
- `python ops/job.py --config simple` launches spot training and instance self-terminates on success.
- `python ops/job.py --logs <job-id>` streams CloudWatch logs.
- `python ops/job.py --resume <job-id>` continues from latest W&B checkpoint after interruption.
- Measured startup targets are met on warm path:
  - sandbox command to SSH: <5 min
  - job command to training start: <7 min
- All flows work using `aws sso login` credentials and public-repo code only.
