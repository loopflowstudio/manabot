"""
bootstrap.py
User-data script generation for sandbox and job GPU machine bootstrapping.
"""

from __future__ import annotations

from ops.provider import RuntimeSpec

BOOTSTRAP_MARKER = "/opt/manabot/bootstrap.v1.done"
DEFAULT_REPO = "https://github.com/loopflowstudio/manabot.git"

_COMMON_BOOTSTRAP = """
  sudo mkdir -p /opt/manabot
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl git gnupg lsb-release

  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
"""

_SYNC_REPO = """
  if [ -d /opt/manabot/repo/.git ]; then
    sudo git -C /opt/manabot/repo fetch --all --prune
    sudo git -C /opt/manabot/repo reset --hard origin/main
  else
    sudo git clone "$REPO_URL" /opt/manabot/repo
  fi
"""


def sandbox_user_data(
    runtime: RuntimeSpec,
    *,
    repo_url: str = DEFAULT_REPO,
    marker_path: str = BOOTSTRAP_MARKER,
) -> str:
    """Build cloud-init user-data for idempotent sandbox setup."""

    return f"""#!/usr/bin/env bash
set -euo pipefail

MARKER={marker_path}
REPO_URL={repo_url}
IMAGE={runtime.image}
FALLBACK_BUILD={"1" if runtime.fallback_build else "0"}

if [ ! -f "$MARKER" ]; then
{_COMMON_BOOTSTRAP}
  sudo usermod -aG docker ubuntu || true

{_SYNC_REPO}

  if ! sudo docker pull "$IMAGE"; then
    if [ "$FALLBACK_BUILD" = "1" ]; then
      sudo docker build -t "$IMAGE" -f /opt/manabot/repo/ops/Dockerfile /opt/manabot/repo
    else
      echo "Failed to pull runtime image and fallback build is disabled." >&2
      exit 1
    fi
  fi

  echo "bootstrap_complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | sudo tee "$MARKER" >/dev/null
fi
"""


def job_user_data(
    runtime: RuntimeSpec,
    *,
    config_name: str,
    job_id: str,
    wandb_run_id: str,
    region: str,
    log_group: str,
    marker_path: str = BOOTSTRAP_MARKER,
) -> str:
    """Build cloud-init user-data for spot job setup + systemd service install."""

    return f"""#!/usr/bin/env bash
set -euo pipefail

MARKER={marker_path}
IMAGE={runtime.image}
FALLBACK_BUILD={"1" if runtime.fallback_build else "0"}
CONFIG_NAME={config_name}
JOB_ID={job_id}
WANDB_RUN_ID={wandb_run_id}
AWS_REGION={region}
LOG_GROUP={log_group}
REPO_URL={DEFAULT_REPO}

if [ ! -f "$MARKER" ]; then
{_COMMON_BOOTSTRAP}

  if ! sudo docker pull "$IMAGE"; then
    if [ "$FALLBACK_BUILD" = "1" ]; then
{_SYNC_REPO}
      sudo docker build -t "$IMAGE" -f /opt/manabot/repo/ops/Dockerfile /opt/manabot/repo
    else
      echo "Failed to pull runtime image and fallback build is disabled." >&2
      exit 1
    fi
  fi

  cat <<'SCRIPT' | sudo tee /opt/manabot/run-job.sh >/dev/null
#!/usr/bin/env bash
set -euo pipefail
source /opt/manabot/job.env

/usr/bin/docker run --rm --gpus all \
  --env-file /opt/manabot/job.env \
  --log-driver=awslogs \
  --log-opt awslogs-region=$AWS_REGION \
  --log-opt awslogs-group=$LOG_GROUP \
  --log-opt awslogs-stream=$JOB_ID/train \
  "$IMAGE" python manabot/model/train.py --config-name "$CONFIG_NAME"

# terminate (instance shutdown behavior is set by launcher)
/sbin/shutdown -h now
SCRIPT
  sudo chmod +x /opt/manabot/run-job.sh

  cat <<'UNIT' | sudo tee /etc/systemd/system/manabot-job.service >/dev/null
[Unit]
Description=manabot GPU training job
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=no
ExecStart=/opt/manabot/run-job.sh

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  echo "bootstrap_complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | sudo tee "$MARKER" >/dev/null
fi
"""
