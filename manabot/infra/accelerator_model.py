"""Resource bounds for a host collector coupled to accelerator service curves.

HostDemand comes from real collection replay; DeviceService comes from a named
hardware measurement. project() bounds blocking and ideally overlapped work.
These are conditional capacity estimates, not CUDA emulation or a queueing model.
All rates use surfaced environment decisions, including both seats. Diagnostic
optimizer observations are an explicit assumed demand, not useful RL progress.
"""

from pydantic import BaseModel, ConfigDict, Field


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HostDemand(Measurement):
    decisions: int = Field(gt=0)
    learner_transitions: int = Field(gt=0)
    seconds: float = Field(gt=0)
    native_seconds: float = Field(ge=0)
    input_bytes: int = Field(gt=0)
    output_bytes: int = Field(gt=0)
    forward_calls: int = Field(gt=0)
    streams: int = Field(gt=0)
    rollout_bytes: int = Field(gt=0)
    buffer_bytes: int = Field(gt=0)


class DeviceService(Measurement):
    card: str
    capacity: str
    batch: int = Field(gt=0)
    inference_seconds: float = Field(gt=0)
    optimizer_seconds: float = Field(gt=0)
    # Retained reports take the max across inference/optimizer phases. This is
    # a measured cell footprint, not total VRAM or a guarantee of concurrency.
    peak_allocated_bytes: int = Field(gt=0)
    evidence_sha256: str
    cell: str


class Scenario(Measurement):
    cpu_workers: int = Field(default=1, ge=1)
    cpu_speed_factor: float = Field(default=1, gt=0)
    gpus: int = Field(default=1, ge=1)
    actors: int = Field(default=1, ge=1)
    transfer_gb_per_second: float = Field(default=12, gt=0)
    transfer_latency_us: float = Field(default=20, ge=0)
    optimizer_observations_per_decision: float = Field(default=0, ge=0)


class Projection(Measurement):
    service: DeviceService
    scenario: Scenario
    status: str
    batch_fill_fraction: float
    usable_gpus: int
    cpu_decisions_per_second: float
    device_decisions_per_second: float
    transfer_decisions_per_second: float
    inflight_decisions_per_second: float
    ideal_overlap_decisions_per_second: float | None
    blocking_decisions_per_second: float | None
    bottleneck: str
    tensor_residency_lower_bound_bytes: int
    limits: list[str]


def project(host: HostDemand, service: DeviceService, scenario: Scenario) -> Projection:
    """Apply demand conservation, per-resource capacity and finite actors.

    One full inference batch consumes B host decisions and their H2D/D2H payloads.
    One GPU serializes inference and the assumed diagnostic optimizer work. GPUs
    have independent links; host workers share work ideally. The in-flight bound
    uses request latency without queueing; it is an optimistic Little's-law bound.
    Blocking means one full batch at a time per GPU, with no host/device overlap.
    No estimate extrapolates a model's CUDA service beyond its supplied batch point.
    """
    batch = service.batch
    cpu_per_decision = host.seconds / host.decisions / scenario.cpu_speed_factor
    payload_per_decision = (host.input_bytes + host.output_bytes) / host.decisions
    cpu_batch = cpu_per_decision * batch
    transfer_batch = (
        payload_per_decision * batch / (scenario.transfer_gb_per_second * 1e9)
        + 2 * scenario.transfer_latency_us / 1e6
    )
    device_batch = service.inference_seconds + (
        scenario.optimizer_observations_per_decision * service.optimizer_seconds
    )
    inflight = scenario.actors * host.streams
    usable_gpus = min(scenario.gpus, inflight // batch)
    # One actor's host latency stays serial even when other actors use cores.
    actor_latency = cpu_per_decision * host.streams + transfer_batch + device_batch
    capacities = {
        "host": min(scenario.cpu_workers, scenario.actors) / cpu_per_decision,
        "device": usable_gpus * batch / device_batch,
        "transfer": usable_gpus * batch / transfer_batch,
        "inflight": inflight / actor_latency,
    }
    # Each active actor supplies at most `streams` outstanding observations.
    # A full-batch-only server cannot run even one batch below this inventory.
    fill = inflight / batch
    ready = fill >= 1
    rate = min(capacities.values()) if ready else None
    blocking = (
        min(
            capacities["host"],
            capacities["inflight"],
            usable_gpus
            * batch
            / (
                cpu_batch * usable_gpus / min(scenario.cpu_workers, scenario.actors)
                + transfer_batch
                + device_batch
            ),
        )
        if ready
        else None
    )
    return Projection(
        service=service,
        scenario=scenario,
        status="conditional_bound" if ready else "insufficient_inflight_for_full_batch",
        batch_fill_fraction=min(1.0, fill),
        usable_gpus=usable_gpus,
        cpu_decisions_per_second=capacities["host"],
        device_decisions_per_second=capacities["device"],
        transfer_decisions_per_second=capacities["transfer"],
        inflight_decisions_per_second=capacities["inflight"],
        ideal_overlap_decisions_per_second=rate,
        blocking_decisions_per_second=blocking,
        bottleneck=min(capacities, key=capacities.__getitem__)
        if ready
        else "batch_fill",
        tensor_residency_lower_bound_bytes=scenario.actors
        * (host.rollout_bytes + host.buffer_bytes),
        limits=[
            "Conditional resource bound, not measured GPU or distributed throughput",
            "No queueing, batching timeout, stragglers, policy lag, network or all-reduce model",
            "CPU workers scale ideally from this laptop; cpu_speed_factor is an assumption",
            "GPU inputs are assumed coalesced into one H2D and one D2H payload per batch",
            "Finite-actor latency is an optimistic bound, not batch-formation simulation",
            "Tensor residency excludes native games, Python overhead, staging and model copies",
            "Optimizer service is a diagnostic Adam loss, not the configured RL update",
            "Historical device-resident float32 CUDA curves need current target calibration",
        ],
    )
