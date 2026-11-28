"""Runtime effect symbols must reach the production policy's admission tables.

The exhaustive effect inventory lives in Rust. These bounded probes check its
vocabulary sources across the native encoder, admission and embedding-row join.
"""

from dataclasses import replace

import numpy as np
import pytest

from manabot.backends.torch.semantic_cards import SemanticCardEncoder
from manabot.env.observation import (
    FACT_CONTINUATION_PROGRAM,
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_KIND_INDEX,
    FACT_VALUE_INDEX,
    ObservationSpace,
)
from manabot.env.shared_effects import DETAIL_KIND_NAMES
from manabot.policy.admission import Catalog, effect_vocabulary, fact_vocabulary
from manabot.policy.prepare import (
    Layout,
    RecordNamespace,
    Tables,
    record_key,
    record_word,
)
from manabot.policy.readings import legacy_readings
from manabot.policy.spans import full_word
import managym

# Incident keys from diagnostic run 13bb90d47e30, not a second effect inventory.
_KOH_KEYS: list[tuple[int, int]] = [
    (1, 0x31DDFF0E),
    (2, 0x3DECD349),
    (3, 0x193062FF),
]


def _tables(encoder: SemanticCardEncoder) -> Tables:
    buffers = {
        f"semantic_cards.{name}": value.detach().cpu().numpy()
        for name, value in encoder.named_buffers()
    }
    # Naming and ownership are unused by these token-only probes.
    buffers.update(
        {
            "name_candidates.definitions": np.zeros(1, np.int64),
            "name_candidates.program_rows": np.zeros(1, np.int64),
            "_is_agent_template": np.zeros(0, bool),
        }
    )
    return Tables.from_buffers(buffers, Layout(legacy_readings(ObservationSpace())))


def _catalog(tables: Tables) -> Catalog:
    assert tables.definition_rows is not None
    return Catalog(
        tables.definition_rows, tables.record_name_keys, tables.record_token_keys
    )


@pytest.fixture(scope="module")
def encoder() -> SemanticCardEncoder:
    return SemanticCardEncoder("world", 4, records=True, decision_memory=False)


@pytest.fixture(scope="module")
def tables(encoder: SemanticCardEncoder) -> Tables:
    return _tables(encoder)


@pytest.fixture(scope="module")
def catalog(tables: Tables) -> Catalog:
    return _catalog(tables)


def _program_token_pairs() -> set[tuple[int, int]]:
    printed = managym.catalog_tokens()
    return set(zip(printed["token_kind"], printed["token_value"], strict=True)) | {
        (kind, value) for kind, _, value in managym.runtime_tokens()
    }


@pytest.mark.parametrize("decision_memory", [False, True])
def test_program_vocabulary_is_the_printed_and_runtime_union(
    decision_memory: bool,
) -> None:
    tables = _tables(
        SemanticCardEncoder("world", 4, records=True, decision_memory=decision_memory)
    )
    kinds, values = np.array(list(_program_token_pairs()), dtype=np.int64).T
    np.testing.assert_array_equal(
        tables.record_token_keys, np.sort(record_key(kinds, values))
    )
    memory_values = np.array([value for _, value in managym.decision_memory_names()])
    memory_keys = record_key(RecordNamespace.TOKEN, memory_values)
    assert (np.isin(memory_keys, tables.record_name_keys) == decision_memory).all()


def _admit_program(
    catalog: Catalog, pairs: list[tuple[int, int]], *, packed: bool
) -> None:
    kinds, words = np.array(pairs, dtype=np.int32).T
    if packed:
        rows = np.zeros((len(pairs), FACT_DIM), dtype=np.int32)
        rows[:, FACT_DETAIL_INDEX] = words
        word = full_word
    else:
        rows = np.zeros((1, len(pairs), FACT_DIM), dtype=np.float32)
        rows[..., FACT_DETAIL_INDEX] = words.view(np.uint32) >> 16
        rows[..., FACT_DETAIL_INDEX + 1] = words & 0xFFFF
        word = record_word
    rows[..., FACT_KIND_INDEX] = FACT_CONTINUATION_PROGRAM
    rows[..., FACT_VALUE_INDEX] = kinds
    fact_kinds = rows[..., FACT_KIND_INDEX]
    fact_vocabulary(
        rows, fact_kinds, fact_kinds == FACT_CONTINUATION_PROGRAM, catalog, word=word
    )


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "v11"])
def test_facts_admit_printed_and_registered_program_tokens(
    catalog: Catalog, packed: bool
) -> None:
    _admit_program(catalog, sorted(_program_token_pairs()), packed=packed)


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "v11"])
def test_facts_reject_unknown_program_tokens(catalog: Catalog, packed: bool) -> None:
    with pytest.raises(ValueError, match=r"missing keys: \(2, 0xffffffff\)$"):
        _admit_program(catalog, [(2, -1)], packed=packed)


def test_effects_admit_modification_program_tokens_and_reject_unknown_pairs(
    catalog: Catalog,
) -> None:
    names = ("Modification::AddTypes", "Modification::AddColors")
    modifications = np.array([managym.catalog_token_value(name) for name in names])
    closings = np.array([managym.catalog_token_value(f"/{name}") for name in names])
    definition = np.zeros(2, dtype=np.int32)
    structure = DETAIL_KIND_NAMES["program_structure"]
    effect_vocabulary(
        modifications, definition, np.full(2, structure), closings, catalog
    )

    with pytest.raises(
        ValueError,
        match=r"missing keys: \(1, 0xffffffff\), \(3, 0xfffffffe\)$",
    ):
        effect_vocabulary(
            modifications,
            definition,
            np.array([structure, DETAIL_KIND_NAMES["program_variant"]]),
            np.array([-1, -2]),
            catalog,
        )


def test_koh_runtime_tokens_have_admitted_embedding_rows(
    encoder: SemanticCardEncoder, tables: Tables, catalog: Catalog
) -> None:
    kinds, words = np.array(_KOH_KEYS, dtype=np.int64).T
    catalog.require_tokens(kinds, words)
    rows = tables.tokens(kinds, words)
    assert (rows > 0).all()
    assert (rows < encoder.token_embedding.num_embeddings).all()
    assert len(np.unique(rows)) == len(_KOH_KEYS)


@pytest.mark.parametrize(
    ("kind", "word"),
    _KOH_KEYS,
    ids=[
        "closing-/Effect::HaveAbilitiesOf",
        "field-definition",
        "variant-Effect::HaveAbilitiesOf",
    ],
)
def test_missing_koh_registration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, kind: int, word: int
) -> None:
    registrations = managym.runtime_tokens()
    filtered = [
        entry for entry in registrations if (entry[0], entry[2]) != (kind, word)
    ]
    assert len(filtered) < len(registrations)
    monkeypatch.setattr(managym, "runtime_tokens", lambda: filtered)
    tables = _tables(SemanticCardEncoder("world", 4, records=True))
    with pytest.raises(
        ValueError,
        match=rf"program token outside the vocabulary; missing keys: \({kind}, 0x{word:08x}\)$",
    ):
        _catalog(tables).require_tokens(np.array([kind]), np.array([word]))


def test_unseen_numeric_values_use_numeric_embedding_rows(
    tables: Tables, catalog: Catalog
) -> None:
    kinds = np.array([4, 4, 5, 5])
    words = np.array([-(2**31), 1234567, -1234567, 1234567])
    assert not np.isin(record_key(kinds, words), tables.record_token_keys).any()
    catalog.require_tokens(kinds, words)
    np.testing.assert_array_equal(
        tables.tokens(kinds, words), tables.record_number_rows[kinds]
    )
    assert (tables.tokens(kinds, words) > 0).all()


@pytest.mark.parametrize("kind", [7, 8])
def test_definition_reference_and_null_require_exact_pairs(
    encoder: SemanticCardEncoder, tables: Tables, catalog: Catalog, kind: int
) -> None:
    keys = tables.record_token_keys
    key = keys[(keys >> 32) == kind][0]
    kinds, words = np.array([kind]), np.array([key & 0xFFFFFFFF])
    catalog.require_tokens(kinds, words)
    rows = tables.tokens(kinds, words)
    assert 0 < rows[0] < encoder.token_embedding.num_embeddings
    without_pair = replace(catalog, tokens=keys[keys != key])
    with pytest.raises(ValueError, match="program token outside the vocabulary"):
        without_pair.require_tokens(kinds, words)


@pytest.mark.parametrize("kind", [0, 10])
def test_invalid_program_token_kind_is_rejected(catalog: Catalog, kind: int) -> None:
    with pytest.raises(ValueError, match="invalid program token kind"):
        catalog.require_tokens(np.array([kind]), np.array([0]))
