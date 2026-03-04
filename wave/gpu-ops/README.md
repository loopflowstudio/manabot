## Vision

First-class GPU ops for manabot: spin up a sandbox to play around in, or kick off a
remote training job and walk away. Public-repo friendly — no account references baked
in, any team with AWS credentials can use it.

Start with AWS. Keep a thin provider interface so adding SFCompute or others later
doesn't require rewriting the world.

"I want to just `python ops/sandbox.py` and be SSH'd into a GPU box in under 5
minutes. And `python ops/job.py --config simple` should start training remotely and
kill the machine when it's done."

### Not here

- Multi-node / distributed training (future wave)
- Job orchestration / pipelines / reactive workflows (future wave)
- Multi-cloud arbitrage (just AWS for now)
- Web dashboard or UI

## Goals

- Two working commands: sandbox (interactive) and job (fire-and-forget)
- Sub-5-minute wall time from command to usable GPU
- Jobs self-terminate on completion, push checkpoints to W&B
- Config-driven machine specs (instance type, region, disk size)
- User identity from AWS SSO — no extra credential files
- Clean provider abstraction that a second backend could implement
- Updated Dockerfile for GPU (CUDA, DLAMI-compatible)
- Works for any contributor who has an AWS account

## Risks

- DLAMI + Docker interaction: NVIDIA Container Toolkit setup can be finicky.
  Mitigate by testing a specific DLAMI version and pinning it.
- Spot instance interruption during long training. Mitigate with checkpointing
  (already in W&B flow) and optional on-demand fallback.
- AWS SSM vs SSH: SSM is cleaner (no key management, no open ports) but less
  familiar. May need both paths.
- ECR setup as a prerequisite adds friction for new contributors.

## Metrics

- Time from command to SSH session (sandbox): target <5 min
- Time from command to training start (job): target <7 min
- Lines of code in ops/: target <500 total
- Number of AWS prerequisites a new contributor must set up: target ≤3
