# Open questions / assumptions

- Assumed `--resume <job-id>` can rely on a local metadata store (`~/.manabot/jobs.json`) to recover `wandb-run-id` and config name. If cross-machine resume is required without local state, we should add a provider-side metadata source (tags/queryable DB).
- Used SSM Run Command to inject `WANDB_API_KEY` into `/opt/manabot/job.env` at launch/start time. This satisfies “via SSM at launch” but does **not** use AWS Systems Manager Parameter Store.
- `ops/specs/*.yaml` keep placeholder AMI values (`ami-REPLACE_WITH_DLAMI`) and expect operators to set real DLAMI IDs before first use.
- Job container log streaming is implemented via Docker `awslogs` driver (CloudWatch as canonical sink) rather than CloudWatch agent.
