"""Explicit native catalog binding and shared encoder dimensions, not strength."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.semantic.learning import BoundSemanticPack, ContentPackBindingError
from manabot.semantic.policy_catalog import bind_policy_catalog


def test_catalog_uses_actual_native_ids_and_shared_encoder() -> None:
    key = "ur-lessons-vs-gw-allies"
    pack = bind_policy_catalog(key)
    encoder = SemanticCardEncoder(key, 8)
    shape = encoder.catalog_shape
    assert shape.definitions == len(pack.ir.definitions)
    assert shape.unique_tokens == len(encoder.token_vocabulary)
    assert shape.expanded_tokens == int(encoder.lengths.sum())
    assert shape.padded_tokens == encoder.tokens.numel()
    assert encoder.token_embedding.num_embeddings == shape.unique_tokens + 1
    definitions = encoder.definitions()
    assert definitions.shape == (shape.definitions + 1, 8)
    for definition, row in pack.definition_row_by_card_def_id.items():
        assert int(encoder._rows(torch.tensor([definition + 1.0]))[0]) == row + 1
    definitions[1:].square().sum().backward()
    assert encoder.token_embedding.weight.grad is not None
    assert encoder.encoder.weight_ih_l0.grad is not None
    assert torch.isfinite(encoder.encoder.weight_ih_l0.grad).all()


def test_unknown_catalog_is_not_a_name_alias() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SemanticCardEncoder("simple-subset", 8)


def test_native_ir_receipt_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    original = BoundSemanticPack.bind

    def altered_binding(
        cls: type[BoundSemanticPack], manifest: dict[str, object], *, ir_path: Path
    ) -> BoundSemanticPack:
        pack = original(manifest, ir_path=ir_path)
        return replace(pack, ir=replace(pack.ir, ir_hash="not-native-ir"))

    monkeypatch.setattr(BoundSemanticPack, "bind", classmethod(altered_binding))
    with pytest.raises(ContentPackBindingError, match="differs from IR"):
        SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)
