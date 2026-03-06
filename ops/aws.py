"""
aws.py
AWS provider implementation for sandbox and job machine lifecycles.
"""

from __future__ import annotations

from collections.abc import Iterable
import ipaddress
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.request

from ops.provider import CommandResult, Machine, MachineSpec, merge_str


class AWSProvider:
    """Boto3-backed provider implementation used by ops/sandbox.py and ops/job.py."""

    def __init__(
        self,
        region: str | None = None,
        *,
        iam_instance_profile: str | None = None,
        key_name: str | None = None,
        security_group_id: str | None = None,
        log_group_prefix: str = "/manabot",
    ):
        try:
            import boto3
            from botocore.exceptions import ClientError, WaiterError
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "Missing boto3. Install ops dependencies with: pip install -e '.[ops]'"
            ) from exc

        self._boto3 = boto3
        self._ClientError = ClientError
        self._WaiterError = WaiterError

        self.default_region = region or boto3.Session().region_name or "us-west-2"
        self._session_cache: dict[str, Any] = {}
        self._client_cache: dict[tuple[str, str], Any] = {}

        self.default_iam_instance_profile = iam_instance_profile
        self.default_key_name = key_name
        self.security_group_id = security_group_id
        self.log_group_prefix = log_group_prefix.rstrip("/")

        self.user = self._resolve_caller_identity(self.default_region)
        self.user_slug = _slugify(self.user)

    # ---------------------------------------------------------------------
    # Provider protocol
    # ---------------------------------------------------------------------
    def create(
        self,
        spec: MachineSpec,
        tags: dict[str, str],
        *,
        user_data: str | None = None,
        iam_instance_profile: str | None = None,
    ) -> Machine:
        """Create a machine and tag it as manabot-managed."""

        region = spec.region or self.default_region
        ec2 = self._client("ec2", region)

        security_group_id = self.security_group_id or self._ensure_security_group(
            region
        )
        key_name = (
            spec.key_name or self.default_key_name or self._ensure_key_pair(region)
        )
        profile_name = (
            iam_instance_profile
            or self.default_iam_instance_profile
            or self._ensure_instance_profile()
        )

        self.ensure_log_groups(region)

        managed_tags = merge_str(
            {
                "manabot:user": self.user,
                "manabot:managed": "true",
                "manabot:region": region,
            },
            tags,
        )

        request: dict[str, Any] = {
            "ImageId": spec.ami,
            "InstanceType": spec.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "SecurityGroupIds": [security_group_id],
            "KeyName": key_name,
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": spec.disk_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": _to_aws_tags(managed_tags),
                },
                {
                    "ResourceType": "volume",
                    "Tags": _to_aws_tags(managed_tags),
                },
            ],
            "IamInstanceProfile": {"Name": profile_name},
            "InstanceInitiatedShutdownBehavior": (
                "terminate" if managed_tags.get("manabot:role") == "job" else "stop"
            ),
        }

        if user_data:
            request["UserData"] = user_data

        if spec.spot:
            request["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": merge_str(
                    {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                    {"MaxPrice": spec.max_spot_price},
                ),
            }

        try:
            response = ec2.run_instances(**request)
        except self._ClientError as exc:
            raise RuntimeError(f"Failed to create EC2 instance: {exc}") from exc

        instance = response["Instances"][0]
        return self._machine_from_instance(instance)

    def start(self, machine: Machine) -> Machine:
        """Start a stopped machine."""

        region = self._machine_region(machine)
        ec2 = self._client("ec2", region)
        ec2.start_instances(InstanceIds=[machine.id])
        return self._refresh_machine(machine)

    def wait_until_ready(self, machine: Machine, timeout: int = 300) -> Machine:
        """Wait for EC2 running + public IP assignment."""

        region = self._machine_region(machine)
        ec2 = self._client("ec2", region)
        waiter = ec2.get_waiter("instance_running")

        try:
            waiter.wait(
                InstanceIds=[machine.id],
                WaiterConfig={"Delay": 10, "MaxAttempts": max(1, timeout // 10)},
            )
        except self._WaiterError as exc:
            raise TimeoutError(
                f"Instance {machine.id} did not reach running state within {timeout}s"
            ) from exc

        deadline = time.time() + timeout
        refreshed = self._refresh_machine(machine)
        while refreshed.public_ip is None and time.time() < deadline:
            print("  waiting for public IP...", flush=True)
            time.sleep(5)
            refreshed = self._refresh_machine(machine)
        return refreshed

    def wait_for_ssm(self, machine: Machine, timeout: int = 300) -> None:
        """Wait for SSM agent to report Online."""

        region = self._machine_region(machine)
        ssm = self._client("ssm", region)
        deadline = time.time() + timeout

        while time.time() < deadline:
            response = ssm.describe_instance_information(
                Filters=[
                    {
                        "Key": "InstanceIds",
                        "Values": [machine.id],
                    }
                ]
            )
            instances = response.get("InstanceInformationList", [])
            if instances and instances[0].get("PingStatus") == "Online":
                return
            print("  waiting for SSM agent...", flush=True)
            time.sleep(5)

        raise TimeoutError(
            f"SSM did not become available for {machine.id} in {timeout}s"
        )

    def stop(self, machine: Machine) -> None:
        """Stop a machine."""

        region = self._machine_region(machine)
        ec2 = self._client("ec2", region)
        ec2.stop_instances(InstanceIds=[machine.id])

    def terminate(self, machine: Machine) -> None:
        """Terminate a machine."""

        region = self._machine_region(machine)
        ec2 = self._client("ec2", region)
        ec2.terminate_instances(InstanceIds=[machine.id])

    def list(self, tags: dict[str, str]) -> list[Machine]:
        """List machines matching tags in default region."""

        region = tags.get("manabot:region", self.default_region)
        ec2 = self._client("ec2", region)

        filters = [
            {"Name": f"tag:{key}", "Values": [value]} for key, value in tags.items()
        ]
        filters.append(
            {
                "Name": "instance-state-name",
                "Values": [
                    "pending",
                    "running",
                    "stopping",
                    "stopped",
                ],
            }
        )

        response = ec2.describe_instances(Filters=filters)
        machines: list[Machine] = []
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                machines.append(self._machine_from_instance(instance))

        return machines

    def run_command(
        self,
        machine: Machine,
        command: str,
        timeout: int = 3600,
    ) -> CommandResult:
        """Execute a shell command over SSM and return full output."""

        region = self._machine_region(machine)
        ssm = self._client("ssm", region)
        deadline = time.time() + timeout

        command_id: str | None = None
        while time.time() < deadline:
            try:
                response = ssm.send_command(
                    InstanceIds=[machine.id],
                    DocumentName="AWS-RunShellScript",
                    Parameters={"commands": [command]},
                    CloudWatchOutputConfig={"CloudWatchOutputEnabled": False},
                    TimeoutSeconds=timeout,
                )
                command_id = response["Command"]["CommandId"]
                break
            except self._ClientError as exc:
                if _is_retryable_send_command_error(exc):
                    time.sleep(5)
                    continue
                raise

        if command_id is None:
            raise TimeoutError(
                f"Command could not be sent within {timeout}s for {machine.id}"
            )
        terminal_states = {
            "Success",
            "Cancelled",
            "TimedOut",
            "Failed",
            "Cancelling",
            "Undeliverable",
            "Terminated",
            "Delivery Timed Out",
            "Execution Timed Out",
        }

        while time.time() < deadline:
            try:
                invocation = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=machine.id,
                )
            except self._ClientError as exc:
                message = str(exc)
                if "InvocationDoesNotExist" in message:
                    time.sleep(2)
                    continue
                raise

            status = invocation.get("Status", "Unknown")
            if status in terminal_states:
                return CommandResult(
                    command_id=command_id,
                    status=status,
                    stdout=invocation.get("StandardOutputContent", ""),
                    stderr=invocation.get("StandardErrorContent", ""),
                )

            time.sleep(2)

        raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    def logs(self, group: str, stream_prefix: str, limit: int = 200) -> list[str]:
        """Return latest CloudWatch log lines for the matching stream prefix."""

        logs_client = self._client("logs", self.default_region)
        streams = logs_client.describe_log_streams(
            logGroupName=group,
            logStreamNamePrefix=stream_prefix,
            orderBy="LastEventTime",
            descending=True,
            limit=10,
        ).get("logStreams", [])

        lines: list[str] = []
        for stream in streams:
            if len(lines) >= limit:
                break
            stream_name = stream["logStreamName"]
            response = logs_client.get_log_events(
                logGroupName=group,
                logStreamName=stream_name,
                startFromHead=False,
                limit=limit,
            )
            for event in response.get("events", []):
                lines.append(event.get("message", "").rstrip("\n"))
                if len(lines) >= limit:
                    break

        return lines[-limit:]

    # ---------------------------------------------------------------------
    # Resource preflight helpers
    # ---------------------------------------------------------------------
    def ensure_log_groups(self, region: str | None = None) -> None:
        """Create per-user log groups for sandbox/job command streams."""

        active_region = region or self.default_region
        logs_client = self._client("logs", active_region)
        for role in ("sandbox", "job"):
            group = f"{self.log_group_prefix}/{role}/{self.user}"
            try:
                logs_client.create_log_group(logGroupName=group)
            except self._ClientError as exc:
                if "ResourceAlreadyExistsException" in str(exc):
                    continue
                raise

    def _ensure_security_group(self, region: str) -> str:
        ec2 = self._client("ec2", region)
        name = f"manabot-{self.user_slug}-ssh"

        groups = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [name]},
            ]
        ).get("SecurityGroups", [])
        if groups:
            group_id = groups[0]["GroupId"]
        else:
            vpc_id = self._default_vpc_id(region)
            try:
                response = ec2.create_security_group(
                    GroupName=name,
                    Description="manabot sandbox SSH access",
                    VpcId=vpc_id,
                    TagSpecifications=[
                        {
                            "ResourceType": "security-group",
                            "Tags": _to_aws_tags(
                                {
                                    "Name": name,
                                    "manabot:user": self.user,
                                    "manabot:managed": "true",
                                }
                            ),
                        }
                    ],
                )
                group_id = response["GroupId"]
            except self._ClientError as exc:
                if "InvalidGroup.Duplicate" not in str(exc):
                    raise
                existing = ec2.describe_security_groups(
                    Filters=[{"Name": "group-name", "Values": [name]}]
                ).get("SecurityGroups", [])
                if not existing:
                    raise
                group_id = existing[0]["GroupId"]

        cidr = self._ssh_cidr()
        try:
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [
                            {
                                "CidrIp": cidr,
                                "Description": "manabot operator access",
                            }
                        ],
                    }
                ],
            )
        except self._ClientError as exc:
            if "InvalidPermission.Duplicate" not in str(exc):
                raise

        return group_id

    def _ensure_key_pair(self, region: str) -> str:
        ec2 = self._client("ec2", region)
        name = f"manabot-{self.user_slug}"
        try:
            ec2.describe_key_pairs(KeyNames=[name])
            return name
        except self._ClientError as exc:
            if "InvalidKeyPair.NotFound" not in str(exc):
                raise

        ssh_dir = Path.home() / ".ssh"
        pubkey_path = ssh_dir / "id_ed25519.pub"
        if not pubkey_path.exists():
            import subprocess
            ssh_dir.mkdir(mode=0o700, exist_ok=True)
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(ssh_dir / "id_ed25519"), "-N", ""],
                check=True,
            )

        key_material = pubkey_path.read_text(encoding="utf-8").strip()
        ec2.import_key_pair(KeyName=name, PublicKeyMaterial=key_material)
        return name

    def _ensure_instance_profile(self) -> str:
        iam = self._client("iam", self.default_region)
        role_name = f"manabot-{self.user_slug}-ec2-role"
        profile_name = f"manabot-{self.user_slug}-ec2-profile"

        try:
            iam.get_instance_profile(InstanceProfileName=profile_name)
            return profile_name
        except self._ClientError as exc:
            if "NoSuchEntity" not in str(exc):
                self._raise_iam_error(exc)

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        try:
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="manabot managed EC2 role",
            )
        except self._ClientError as exc:
            if "EntityAlreadyExists" not in str(exc):
                self._raise_iam_error(exc)

        managed_policies = [
            "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
        ]
        for policy_arn in managed_policies:
            try:
                iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            except self._ClientError as exc:
                self._raise_iam_error(exc)

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                    ],
                    "Resource": "*",
                }
            ],
        }
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="manabot-cloudwatch-inline",
            PolicyDocument=json.dumps(inline_policy),
        )

        try:
            iam.create_instance_profile(InstanceProfileName=profile_name)
        except self._ClientError as exc:
            if "EntityAlreadyExists" not in str(exc):
                self._raise_iam_error(exc)

        try:
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )
        except self._ClientError as exc:
            if "LimitExceeded" not in str(exc) and "EntityAlreadyExists" not in str(
                exc
            ):
                self._raise_iam_error(exc)

        # IAM propagation window.
        time.sleep(10)
        return profile_name

    def _raise_iam_error(self, exc: Exception) -> None:
        message = (
            "Unable to create/update IAM role/profile for manabot managed instances. "
            "Either grant iam:* permissions for this role flow, or rerun with a "
            "pre-provisioned profile via --iam-instance-profile / MANABOT_IAM_INSTANCE_PROFILE. "
            f"Original error: {exc}"
        )
        raise RuntimeError(message) from exc

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _resolve_caller_identity(self, region: str) -> str:
        import subprocess
        from botocore.exceptions import (
            ClientError,
            NoCredentialsError,
            TokenRetrievalError,
        )

        sts = self._client("sts", region)
        try:
            identity = sts.get_caller_identity()
        except (ClientError, NoCredentialsError, TokenRetrievalError):
            # SSO token expired — try to refresh automatically
            subprocess.run(["aws", "sso", "login"], check=True)
            # Rebuild the client so it picks up the fresh token
            self._client_cache.pop(("sts", region), None)
            sts = self._client("sts", region)
            identity = sts.get_caller_identity()
        arn = identity.get("Arn", "unknown")
        return _user_from_arn(arn)

    def _default_vpc_id(self, region: str) -> str:
        ec2 = self._client("ec2", region)
        default_vpcs = ec2.describe_vpcs(
            Filters=[{"Name": "isDefault", "Values": ["true"]}]
        ).get("Vpcs", [])
        if default_vpcs:
            return default_vpcs[0]["VpcId"]

        all_vpcs = ec2.describe_vpcs().get("Vpcs", [])
        if not all_vpcs:
            raise RuntimeError(f"No VPC found in region {region}")
        return all_vpcs[0]["VpcId"]

    def _refresh_machine(self, machine: Machine) -> Machine:
        region = self._machine_region(machine)
        ec2 = self._client("ec2", region)
        response = ec2.describe_instances(InstanceIds=[machine.id])
        reservations = response.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            return Machine(
                id=machine.id,
                public_ip=None,
                status="terminated",
                tags=machine.tags,
            )
        return self._machine_from_instance(reservations[0]["Instances"][0])

    def _machine_from_instance(self, instance: dict[str, Any]) -> Machine:
        tag_map = {
            tag["Key"]: tag["Value"]
            for tag in instance.get("Tags", [])
            if "Key" in tag and "Value" in tag
        }
        return Machine(
            id=instance["InstanceId"],
            public_ip=instance.get("PublicIpAddress"),
            status=instance.get("State", {}).get("Name", "unknown"),
            tags=tag_map,
        )

    def _machine_region(self, machine: Machine) -> str:
        return machine.tags.get("manabot:region", self.default_region)

    def _ssh_cidr(self) -> str:
        cidr = os.getenv("MANABOT_SSH_CIDR")
        if cidr:
            return cidr

        public_ip = os.getenv("MANABOT_PUBLIC_IP") or self._resolve_public_ip()
        return f"{public_ip}/32"

    def _resolve_public_ip(self) -> str:
        endpoints = [
            "https://checkip.amazonaws.com",
            "https://api.ipify.org",
        ]
        for endpoint in endpoints:
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    ip = response.read().decode("utf-8").strip()
                ipaddress.ip_address(ip)
                return ip
            except Exception:
                continue

        raise RuntimeError(
            "Could not resolve caller public IP. Set MANABOT_PUBLIC_IP or MANABOT_SSH_CIDR."
        )

    def _session(self, region: str):
        cached = self._session_cache.get(region)
        if cached is not None:
            return cached
        session = self._boto3.Session(region_name=region)
        self._session_cache[region] = session
        return session

    def _client(self, service: str, region: str):
        key = (service, region)
        cached = self._client_cache.get(key)
        if cached is not None:
            return cached
        client = self._session(region).client(service, region_name=region)
        self._client_cache[key] = client
        return client


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", value.lower())[:40]


def _user_from_arn(arn: str) -> str:
    """Extract a stable human-ish user token from caller ARN."""

    if ":assumed-role/" in arn:
        # arn:aws:sts::<acct>:assumed-role/<role>/<session>
        bits = arn.split(":assumed-role/")[-1].split("/")
        if len(bits) >= 2:
            return bits[1]
    if ":user/" in arn:
        return arn.split(":user/")[-1]
    if ":root" in arn:
        return "root"
    return arn.rsplit("/", maxsplit=1)[-1]


def _to_aws_tags(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in tags.items()]


def _is_retryable_send_command_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    code = error.get("Code")
    message = str(error.get("Message", "")).lower()
    return code == "InvalidInstanceId" and "valid state" in message


def choose_single_machine(machines: Iterable[Machine]) -> Machine | None:
    """Choose the newest-looking machine deterministically if multiple match."""

    machine_list = list(machines)
    if not machine_list:
        return None
    # Instance IDs are time-ordered enough for deterministic picks in this workflow.
    return max(machine_list, key=lambda machine: machine.id)
