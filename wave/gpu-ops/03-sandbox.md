# 03: Sandbox

`python ops/sandbox.py` — get a GPU box to play with.

## What to build

```
python ops/sandbox.py                  # create or resume sandbox, SSH in
python ops/sandbox.py --stop           # stop instance (keep disk)
python ops/sandbox.py --start          # restart stopped instance
python ops/sandbox.py --terminate      # destroy everything
python ops/sandbox.py --status         # show current sandbox state
```

## Behavior

**Create flow:**
1. Check for existing sandbox (tagged `manabot:user=<you>, manabot:role=sandbox`)
2. If running → SSH in directly
3. If stopped → start it, wait, SSH in
4. If none → create from spec, wait, run setup script, SSH in

**Setup script** (runs once on first create via user-data or SSM):
- Install NVIDIA Container Toolkit if not present
- Pull Docker image from ECR (or build locally if no ECR configured)
- Clone the repo (or sync from local)
- Print connection info

**SSH connection:**
- Uses `ssh -o StrictHostKeyChecking=no -i <key>` or SSM session
- Prints the SSH command so user can reconnect manually

## Spec file

```yaml
# ops/specs/sandbox.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 100
spot: false          # sandboxes are on-demand — you don't want interruption
ami: ami-0xxxxx      # Ubuntu 22.04 DLAMI
```

## Done when

- `python ops/sandbox.py` creates a g5.xlarge and opens an SSH session
- Running it again reconnects to the same instance
- `--stop` stops it, `--start` resumes, `--terminate` destroys
- Total time from command to SSH prompt: <5 minutes
