# 04: Job Runner

`python ops/job.py --config simple` — fire and forget.

## What to build

```
python ops/job.py --config simple              # launch training job
python ops/job.py --config attention           # different config
python ops/job.py --list                       # show running jobs
python ops/job.py --logs <job-id>              # tail logs
python ops/job.py --cancel <job-id>            # terminate early
```

## Behavior

**Launch flow:**
1. Create instance from job spec (spot by default)
2. Tag with `manabot:role=job, manabot:config=<name>, manabot:user=<you>`
3. Run via SSM or user-data:
   ```bash
   docker run --gpus all \
     -e WANDB_API_KEY=$WANDB_API_KEY \
     <ecr-image> \
     python manabot/model/train.py --config-name <config>
   ```
4. On completion (success or failure): push final checkpoint, terminate instance
5. On spot interruption: W&B checkpoint is the recovery mechanism

**Self-termination:**
- Wrapper script that runs train.py, then calls `shutdown -h now`
- Or: a CloudWatch alarm on CPU idle that terminates after 15 min idle
- Simplest: the user-data script ends with `shutdown -h now`

**Logs:**
- Training logs go to W&B (already built)
- Instance logs available via SSM or CloudWatch
- `--logs` tails CloudWatch or SSM output

## Spec file

```yaml
# ops/specs/job.yaml
instance_type: g5.xlarge
region: us-east-1
disk_gb: 50
spot: true
max_spot_price: "1.50"
ami: ami-0xxxxx
```

Configs can override instance type:
```yaml
# ops/specs/job-attention.yaml (optional)
instance_type: g5.2xlarge
```

## Done when

- `python ops/job.py --config simple` launches a spot instance that trains and self-terminates
- W&B shows the run with correct config and final checkpoint
- `--list` shows running jobs with their config and uptime
- `--cancel` terminates a running job
- Instance is gone within 20 minutes of training completion
