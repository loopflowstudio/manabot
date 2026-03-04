# 02: Provider Abstraction + AWS Implementation

The primitives: create a machine, wait for it, connect to it, destroy it.

## What to build

```python
# ops/provider.py

@dataclass
class MachineSpec:
    instance_type: str       # e.g. "g5.xlarge"
    region: str              # e.g. "us-east-1"
    disk_gb: int             # EBS volume size
    ami: str                 # DLAMI AMI ID
    spot: bool               # spot vs on-demand
    max_spot_price: str | None

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

```python
# ops/aws.py — implements Provider using boto3

class AWSProvider:
    # Uses ec2 client from default boto3 session (AWS SSO works out of the box)
    # Tags every resource with {"manabot:user": <iam-identity>, "manabot:role": "sandbox"|"job"}
    # Uses SSM RunCommand for run_command()
    # Security group: allows SSH inbound (22) from caller's IP only
    # IAM instance profile for S3/ECR access
```

```python
# ops/config.py

# Machine spec configs loaded from ops/specs/*.yaml
# e.g. ops/specs/sandbox.yaml, ops/specs/job-simple.yaml
```

## Key decisions

- Provider is a Protocol, not an ABC. Duck typing. No registration machinery.
- AWS creds come from the default boto3 chain (SSO, env vars, instance profile).
  No manabot-specific credential config.
- Every resource tagged with user identity for multi-user disambiguation.
- Security group is created per-user on first use, cached by tag lookup.

## Done when

- `AWSProvider().create(spec)` launches an EC2 instance with correct tags
- `AWSProvider().wait_until_ready(machine)` blocks until SSH-able
- `AWSProvider().terminate(machine)` cleans up
- `AWSProvider().list({"manabot:role": "sandbox"})` returns running sandboxes
- All of the above work with `aws sso login` credentials
