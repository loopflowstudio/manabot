"""Conditional scaling must conserve demand and expose unfillable GPU batches."""

from pydantic import ValidationError
import pytest

from manabot.infra.accelerator_model import DeviceService, HostDemand, Scenario, project


def host() -> HostDemand:
    return HostDemand(
        decisions=1000,
        learner_transitions=500,
        seconds=1,
        native_seconds=0.25,
        input_bytes=1_000_000,
        output_bytes=100_000,
        forward_calls=100,
        streams=16,
        rollout_bytes=100_000,
        buffer_bytes=10_000,
    )


def device() -> DeviceService:
    return DeviceService(
        card="test",
        capacity="fixture",
        batch=64,
        inference_seconds=0.01,
        optimizer_seconds=0.1,
        peak_allocated_bytes=1_000_000,
        evidence_sha256="fixture",
        cell="b64",
    )


def test_fast_device_exposes_host_and_more_gpu_cannot_fix_it() -> None:
    one = project(host(), device(), Scenario(actors=64))
    many = project(host(), device(), Scenario(actors=64, gpus=8))
    assert one.bottleneck == many.bottleneck == "host"
    assert one.ideal_overlap_decisions_per_second == 1000
    assert many.ideal_overlap_decisions_per_second == 1000
    assert one.blocking_decisions_per_second < one.ideal_overlap_decisions_per_second


def test_learning_work_reverses_the_bottleneck_and_weak_link_limits_capacity() -> None:
    inference = project(host(), device(), Scenario(cpu_workers=8, actors=64))
    learning = project(
        host(),
        device(),
        Scenario(cpu_workers=8, actors=64, optimizer_observations_per_decision=1),
    )
    assert inference.device_decisions_per_second == 6400
    assert learning.device_decisions_per_second == pytest.approx(64 / 0.11)
    assert (
        learning.ideal_overlap_decisions_per_second
        < inference.ideal_overlap_decisions_per_second
    )
    transfer = project(
        host(),
        device(),
        Scenario(cpu_workers=64, actors=64, transfer_gb_per_second=0.0001),
    )
    assert transfer.bottleneck == "transfer"


def test_finite_actor_inventory_limits_usable_gpus_and_tensor_memory() -> None:
    too_small = project(host(), device(), Scenario(actors=1, gpus=64))
    assert too_small.status == "insufficient_inflight_for_full_batch"
    assert too_small.ideal_overlap_decisions_per_second is None
    assert too_small.usable_gpus == 0
    enough = project(host(), device(), Scenario(actors=8, gpus=64))
    assert enough.usable_gpus == 2
    assert enough.tensor_residency_lower_bound_bytes == 8 * 110_000


def test_nonfinite_assumptions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario(cpu_speed_factor=float("nan"))
