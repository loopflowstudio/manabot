"""Runtime effect symbols must reach the production policy's admission tables.

The exhaustive effect inventory lives in Rust. These bounded probes check its
vocabulary sources across the Python encoder, admission and embedding-row join.
"""

from dataclasses import replace

import numpy as np
import pytest

from manabot.backends.torch._semantic_cards import SemanticCardEncoder
from manabot.env.observation import ObservationSpace
from manabot.policy.admission import Catalog
from manabot.policy.prepare import Layout, Tables, record_key
from manabot.policy.readings import legacy_readings
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
    return SemanticCardEncoder("world", 4, records=True)


def test_koh_runtime_tokens_have_admitted_embedding_rows(
    encoder: SemanticCardEncoder,
) -> None:
    tables = _tables(encoder)
    kinds, words = np.array(_KOH_KEYS, dtype=np.int64).T
    _catalog(tables).require_tokens(kinds, words)
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
    registrations = managym.record_vocabulary()
    filtered = [entry for entry in registrations if entry[3:] != (kind, word)]
    assert len(filtered) < len(registrations)
    monkeypatch.setattr(managym, "record_vocabulary", lambda: filtered)
    tables = _tables(SemanticCardEncoder("world", 4, records=True))
    with pytest.raises(
        ValueError,
        match=rf"program token outside the vocabulary; missing keys: \({kind}, 0x{word:08x}\)$",
    ):
        _catalog(tables).require_tokens(np.array([kind]), np.array([word]))


def test_unseen_numeric_values_use_numeric_embedding_rows(
    encoder: SemanticCardEncoder,
) -> None:
    tables = _tables(encoder)
    kinds = np.array([4, 4, 5, 5])
    words = np.array([-(2**31), 1234567, -1234567, 1234567])
    assert not np.isin(record_key(kinds, words), tables.record_token_keys).any()
    _catalog(tables).require_tokens(kinds, words)
    np.testing.assert_array_equal(
        tables.tokens(kinds, words), tables.record_number_rows[kinds]
    )
    assert (tables.tokens(kinds, words) > 0).all()


@pytest.mark.parametrize("kind", [7, 8])
def test_definition_reference_and_null_require_exact_pairs(
    encoder: SemanticCardEncoder, kind: int
) -> None:
    tables = _tables(encoder)
    keys = tables.record_token_keys
    key = keys[(keys >> 32) == kind][0]
    kinds, words = np.array([kind]), np.array([key & 0xFFFFFFFF])
    _catalog(tables).require_tokens(kinds, words)
    rows = tables.tokens(kinds, words)
    assert 0 < rows[0] < encoder.token_embedding.num_embeddings
    without_pair = replace(_catalog(tables), tokens=keys[keys != key])
    with pytest.raises(ValueError, match="program token outside the vocabulary"):
        without_pair.require_tokens(kinds, words)


@pytest.mark.parametrize("kind", [0, 10])
def test_invalid_program_token_kind_is_rejected(
    encoder: SemanticCardEncoder, kind: int
) -> None:
    with pytest.raises(ValueError, match="invalid program token kind"):
        _catalog(_tables(encoder)).require_tokens(np.array([kind]), np.array([0]))
