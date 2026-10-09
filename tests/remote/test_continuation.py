"""Continuation keeps scientific job identities, costs and creation fences intact."""

from pathlib import Path

import pytest

from manabot.infra.artifacts import StoredArtifact
from manabot.remote.bundle import Bundle, BundleFile
from manabot.remote.cohort import CohortEntry, CohortInsertion, insert_cohort
from manabot.remote.continuation import Continuation, bind_continuation
from manabot.remote.job_store import worker_policy
from manabot.remote.plan import compile_plan
from manabot.remote.snapshots import JobManifest
from manabot.training.models import AtaraxosMoveLearning
from tests.remote.test_cohort import Harness


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    h = Harness(tmp_path, monkeypatch)
    entries = []
    for entry in h.cohort.entries:
        regime = entry.plan.regime.model_copy(deep=True)
        regime.stages[0].learning = AtaraxosMoveLearning(gradient="ataraxos_move")
        plan = compile_plan(
            regime.model_dump_json(),
            entry.plan.spec,
            entry.plan.source,
            entry.plan.seed,
        )
        entries.append(entry.model_copy(update={"plan": plan}))
    h.cohort = h.cohort.model_copy(
        update={"spending_limit": 20, "entries": tuple(entries)}
    )
    return h


def insertion_for(harness: Harness) -> CohortInsertion:
    return CohortInsertion(
        after_job="test-0",
        entries=(
            CohortEntry(
                job_id="continuation",
                plan=harness.cohort.entries[0].plan,
                continuation=Continuation(job_id="test-0", iteration=1),
            ),
        ),
    )


def test_insert_preserves_active_job_and_restart(harness: Harness) -> None:
    first = harness.launch()
    original = harness.cohort.model_dump_json()
    insertion = insertion_for(harness)
    state = insert_cohort(harness.cohort, insertion, store=harness.store)
    assert state.attempts[0].spec == first
    assert [e.job_id for e in state.entries(harness.cohort)] == [
        "test-0",
        "continuation",
        "test-1",
    ]
    assert harness.restart().state.insertion == insertion
    assert insert_cohort(harness.cohort, insertion, store=harness.store) == state
    assert harness.cohort.model_dump_json() == original
    assert harness.provider.created == 1


def appended(insertion: CohortInsertion, job_id: str = "follow-up") -> CohortInsertion:
    entry = insertion.entries[0].model_copy(update={"job_id": job_id})
    return insertion.model_copy(update={"entries": (*insertion.entries, entry)})


def test_bound_insertion_accepts_appended_entry(harness: Harness) -> None:
    first = harness.launch()
    insertion = insertion_for(harness)
    insert_cohort(harness.cohort, insertion, store=harness.store)
    extended = appended(insertion)
    state = insert_cohort(harness.cohort, extended, store=harness.store)
    assert state.attempts[0].spec == first
    assert [e.job_id for e in state.entries(harness.cohort)] == [
        "test-0",
        "continuation",
        "follow-up",
        "test-1",
    ]
    assert harness.restart().state.insertion == extended
    assert insert_cohort(harness.cohort, extended, store=harness.store) == state
    assert harness.provider.created == 1


def test_bound_insertion_rejects_rewrites(harness: Harness) -> None:
    harness.launch()
    insertion = CohortInsertion.model_validate(
        {
            **insertion_for(harness).model_dump(),
            "deadline": 200000,
            "access_expires_at": 200000,
        }
    )
    insert_cohort(harness.cohort, insertion, store=harness.store)
    extended = appended(insertion)
    renamed = extended.entries[0].model_copy(update={"job_id": "renamed"})
    rewrites = (
        insertion.model_copy(update={"entries": (renamed,)}),
        extended.model_copy(update={"entries": (renamed, extended.entries[1])}),
        extended.model_copy(update={"entries": extended.entries[::-1]}),
        extended.model_copy(update={"after_job": "test-1"}),
    )
    for rewrite in rewrites:
        with pytest.raises(ValueError, match="binds another insertion"):
            insert_cohort(harness.cohort, rewrite, store=harness.store)
    for limits in (
        {"deadline": 150000},
        {"deadline": None, "access_expires_at": None},
        {"access_expires_at": 200000 - 1, "deadline": 200000 - 1},
    ):
        with pytest.raises(ValueError, match="shorten admitted limits"):
            insert_cohort(
                harness.cohort,
                extended.model_copy(update=limits),
                store=harness.store,
            )
    assert harness.restart().state.insertion == insertion


def test_appended_entry_cannot_pass_admitted_successor(harness: Harness) -> None:
    first = harness.launch()
    insertion = insertion_for(harness)
    insert_cohort(harness.cohort, insertion, store=harness.store)
    harness.finish(first)
    supervisor = harness.restart()
    supervisor.bind = lambda entry, status, cache: entry.job(harness.clock.now)
    continuation = supervisor.tick().attempts[-1].spec
    assert continuation.job_id == "continuation"
    with pytest.raises(TimeoutError):
        harness.restart().tick()
    harness.finish(continuation)
    assert harness.restart().tick().attempts[-1].spec.job_id == "test-1"
    with pytest.raises(ValueError, match="boundary already passed"):
        insert_cohort(harness.cohort, appended(insertion), store=harness.store)


def test_appended_entry_cannot_reset_budget(harness: Harness) -> None:
    harness.launch()
    insertion = insertion_for(harness)
    insert_cohort(harness.cohort, insertion, store=harness.store)
    extended = appended(insertion)
    plan = extended.entries[1].plan
    expensive = plan.model_copy(
        update={"spec": plan.spec.model_copy(update={"spending_limit": 100})}
    )
    extended = extended.model_copy(
        update={
            "entries": (
                extended.entries[0],
                extended.entries[1].model_copy(update={"plan": expensive}),
            )
        }
    )
    with pytest.raises(ValueError, match="inclusive budget"):
        insert_cohort(harness.cohort, extended, store=harness.store)


def test_insert_rejects_passed_boundary(harness: Harness) -> None:
    first = harness.launch()
    harness.finish(first)
    harness.restart().tick()
    with pytest.raises(ValueError, match="boundary already passed"):
        insert_cohort(harness.cohort, insertion_for(harness), store=harness.store)


def test_insert_cannot_reset_budget(harness: Harness) -> None:
    harness.launch()
    insertion = insertion_for(harness)
    expensive = insertion.entries[0].plan.model_copy(deep=True)
    expensive = expensive.model_copy(
        update={"spec": expensive.spec.model_copy(update={"spending_limit": 100})}
    )
    insertion = insertion.model_copy(
        update={
            "entries": (insertion.entries[0].model_copy(update={"plan": expensive}),)
        }
    )
    with pytest.raises(ValueError, match="inclusive budget"):
        insert_cohort(harness.cohort, insertion, store=harness.store)


def test_insert_lost_cas_cannot_overwrite_admission(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.launch()
    monkeypatch.setattr(harness.store, "replace", lambda *args: False)
    with pytest.raises(RuntimeError, match="advanced during insertion"):
        insert_cohort(harness.cohort, insertion_for(harness), store=harness.store)
    assert harness.restart().state.insertion is None


def test_manifest_binds_all_four_inputs(harness: Harness, tmp_path: Path) -> None:
    first = harness.launch()
    harness.finish(first)
    status = harness.observe(first)
    assert status.record is not None
    status = status.model_copy(
        update={"record": status.record.model_copy(update={"updates": 1})}
    )
    paths = [
        "run/run.json",
        "run/policy-raw.pt",
        "run/policy-ema.pt",
        "run/policy-optimizer.pt",
    ]
    manifest = JobManifest(
        spec_sha256=first.identity,
        generation=1,
        complete=True,
        bundle=Bundle(
            files=tuple(
                BundleFile(
                    producer_path=f"/original/{path}",
                    relative_path=path,
                    size=10,
                    sha256=str(i) * 64,
                )
                for i, path in enumerate(paths)
            )
        ),
        artifacts=tuple(
            StoredArtifact(
                uri=f"{first.prefix}/runtime/{i}",
                sha256=str(i) * 64,
                bytes=10,
                version_id=f"version-{i}",
            )
            for i in range(4)
        ),
    )
    entry = insertion_for(harness).entries[0]
    spec = bind_continuation(entry, status, tmp_path, manifest=manifest)
    assert spec.plan.regime.stages[0].learning_state is not None
    assert spec.learning_inputs["optimizer"].version_id == "version-3"
    assert first.plan.regime.stages[0].learning_state is None
    assert spec.plan.regime.stages[0].learning == first.plan.regime.stages[0].learning
    policy = worker_policy(
        spec.prefix,
        spec.deadline,
        inputs=tuple(a.uri for a in spec.learning_inputs.values()),
    )
    grants = [
        s
        for s in policy["Statement"]
        if s.get("Action") == ["s3:GetObject", "s3:GetObjectVersion"]
    ]
    assert len(grants) == 1
    assert len(grants[0]["Resource"]) == 4
    assert all("*" not in resource for resource in grants[0]["Resource"])
    incomplete = manifest.model_copy(update={"complete": False})
    with pytest.raises(ValueError, match="final predecessor"):
        bind_continuation(entry, status, tmp_path, manifest=incomplete)


def test_inserted_job_finishes_before_original_successor(harness: Harness) -> None:
    first = harness.launch()
    insertion = insertion_for(harness)
    insert_cohort(harness.cohort, insertion, store=harness.store)
    harness.finish(first)
    supervisor = harness.restart()
    supervisor.bind = lambda entry, status, cache: entry.job(harness.clock.now)
    state = supervisor.tick()
    continuation = state.attempts[-1].spec
    assert continuation.job_id == "continuation"
    with pytest.raises(TimeoutError):
        harness.restart().tick()
    harness.finish(continuation)
    state = harness.restart().tick()
    final = state.attempts[-1].spec
    assert final.job_id == "test-1"
    with pytest.raises(TimeoutError):
        harness.restart().tick()
    harness.finish(final)
    state = harness.restart().tick()
    assert state.phase == "completed"
    assert harness.provider.created == 3
    assert [a.spec.job_id for a in state.attempts] == [
        "test-0",
        "continuation",
        "test-1",
    ]


def test_failed_predecessor_does_not_admit_continuation(harness: Harness) -> None:
    first = harness.launch()
    insert_cohort(harness.cohort, insertion_for(harness), store=harness.store)
    harness.finish(first, "failed")
    state = harness.restart().tick()
    assert state.phase == "failed"
    assert len(state.attempts) == 1
    assert harness.provider.created == 1


def test_explicit_deadline_extension_keeps_admitted_job_deadline(
    harness: Harness,
) -> None:
    first = harness.launch()
    insertion = insertion_for(harness)
    extended = CohortInsertion.model_validate(
        {**insertion.model_dump(), "deadline": 200000, "access_expires_at": 200000}
    )
    state = insert_cohort(harness.cohort, extended, store=harness.store)
    assert state.effective_deadline(harness.cohort) == 200000
    assert state.attempts[0].spec.deadline == first.deadline
    assert harness.cohort.deadline == 100000
    with pytest.raises(ValueError, match="access lifetime"):
        CohortInsertion.model_validate({**insertion.model_dump(), "deadline": 200000})
