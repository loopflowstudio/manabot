"""
test_provider.py
Tests for ops provider contracts and spec loading utilities.
"""

from __future__ import annotations

from pathlib import Path

from ops.provider import load_machine_spec, load_runtime_spec, merge_str


def test_load_machine_spec_from_repo_defaults():
    spec = load_machine_spec("sandbox")

    assert spec.instance_type == "g5.xlarge"
    assert spec.region == "us-west-2"
    assert spec.spot is False


def test_load_runtime_spec_from_repo_defaults():
    runtime = load_runtime_spec()

    assert runtime.image.startswith("ghcr.io/")
    assert runtime.log_group_prefix == "/manabot"


def test_merge_str_coerces_strings_and_skips_none():
    merged = merge_str({"a": 1, "b": None}, {"c": True, "a": "override"})

    assert merged == {
        "a": "override",
        "c": "True",
    }


def test_load_machine_spec_from_custom_base_dir(tmp_path: Path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "sandbox.yaml").write_text(
        """
instance_type: g6.xlarge
region: us-west-2
disk_gb: 120
spot: false
ami: ami-custom
""".strip(),
        encoding="utf-8",
    )

    spec = load_machine_spec("sandbox", base_dir=specs_dir)

    assert spec.instance_type == "g6.xlarge"
    assert spec.region == "us-west-2"
    assert spec.disk_gb == 120
    assert spec.ami == "ami-custom"
