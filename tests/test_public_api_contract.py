"""The installed library, contributor contract and documented exports agree."""

import ast
from importlib import import_module
import inspect
from pathlib import Path
import re
import subprocess
import sys

import pytest

from manabot.infra.config import MatchConfig
from scripts.public_api import (
    PUBLIC_MODULES,
    ROOT,
    Contract,
    Symbol,
    deprecated_stub_lines,
    main,
    native_exports,
    python_class,
    resolve,
    stub_class,
    update,
)


def test_public_api_matches_snapshot() -> None:
    subprocess.run(
        [sys.executable, "scripts/public_api.py", "check"], cwd=ROOT, check=True
    )


def test_match_config_publishes_fields_and_methods_without_validator_hooks() -> None:
    members = python_class(MatchConfig).members
    assert members["decklists"].kind == "field"
    assert members["authored"].kind == "classmethod"
    decorators = MatchConfig.__pydantic_decorators__
    assert decorators.field_validators and decorators.model_validators
    assert members.keys().isdisjoint(decorators.field_validators)
    assert members.keys().isdisjoint(decorators.model_validators)


def test_published_symbols_come_from_their_defining_module() -> None:
    owners: dict[str, str] = {}
    for mod in PUBLIC_MODULES:
        module = import_module(mod)
        for name in module.__all__:
            value, warned = resolve(module, name)
            if warned or getattr(value, "__deprecated__", False):
                continue
            assert name not in owners, f"{name} published by {owners[name]} and {mod}"
            owners[name] = mod
            if isinstance(value, type) or inspect.isfunction(value):
                assert value.__module__ == mod, (
                    f"{mod}.{name} is defined in {value.__module__}"
                )


def test_stub_comments_mark_only_the_deprecated_declaration() -> None:
    """Marker comments flag only the declaration they annotate."""
    sample = (
        "# Deprecated: removed in next minor.\n"
        "class OldType:\n"
        "    pass\n"
        "class KeepType:\n"
        "    keep: int\n"
        "    old: int  # Deprecated: use keep.\n"
        "    def active(self) -> None: ...\n"
    )
    marked = deprecated_stub_lines(sample)
    tree = ast.parse(sample)
    exports: dict[str, Symbol] = {}
    for node in tree.body:
        assert isinstance(node, ast.ClassDef)
        exports[node.name] = stub_class(node, exports, marked)
    assert exports["OldType"].deprecated
    assert not exports["KeepType"].deprecated
    assert exports["KeepType"].members["old"].deprecated
    assert not exports["KeepType"].members["keep"].deprecated
    assert not exports["KeepType"].members["active"].deprecated

    live = native_exports()
    assert not any(symbol.deprecated for symbol in live.values())
    assert not live["PlayerConfig"].members["deck"].deprecated
    assert not live["Observation"].members["validate"].deprecated


def test_readme_lists_the_published_exports() -> None:
    section = (
        (ROOT / "manabot/README.md")
        .read_text()
        .split("## Library API\n")[1]
        .split("\n## ")[0]
    )
    documented = {
        module: set(re.findall(r"`([A-Za-z_]\w*)`", exports))
        for module, exports in re.findall(
            r"^\| `(manabot\.[\w.]+)` \| (.*?) \|$", section, re.M
        )
    }
    assert documented == {
        mod: set(import_module(mod).__all__) for mod in PUBLIC_MODULES
    }


def test_only_published_modules_declare_exports() -> None:
    paths = {Path(*mod.split(".")).with_suffix(".py") for mod in PUBLIC_MODULES}
    found = {
        p.relative_to(ROOT)
        for p in (ROOT / "manabot").rglob("*.py")
        if "__all__" in (source := p.read_text())
        and any(
            isinstance(n, ast.Name)
            and n.id == "__all__"
            and isinstance(n.ctx, ast.Store)
            for n in ast.walk(ast.parse(source))
        )
    }
    assert found == paths


@pytest.mark.parametrize(
    ("before", "after", "detail"),
    [
        ("(old)", "(new)", "parameter old removed"),
        (
            "(first, last=...)",
            "(first, middle=..., last=...)",
            "positional parameter order changed",
        ),
        ("(name=...)", "(name)", "parameter name became required"),
        ("(name)", "(*, name)", "parameter name kind changed"),
        ("(name)", "(name, *, count)", "required parameter count added"),
    ],
)
def test_patch_refuses_incompatible_calls(before: str, after: str, detail: str) -> None:
    baseline = Contract(
        line="0.2",
        breaks=[],
        modules={"lib": {"f": Symbol(kind="function", signature=before)}},
    )
    with pytest.raises(ValueError, match=detail):
        update(
            "record",
            baseline,
            {"lib": {"f": Symbol(kind="function", signature=after)}},
            "0.2.6",
            None,
        )


def test_rollover_only_drops_previously_deprecated_api() -> None:
    old = Symbol(kind="method", deprecated=True)
    keep = Symbol(kind="method", signature="(self, x)")
    current = Symbol(kind="method", signature="(self, x=..., *, y=...)")
    baseline = Contract(
        line="0.2",
        breaks=[],
        modules={
            "lib": {"C": Symbol(kind="class", members={"old": old, "keep": keep})}
        },
    )
    modules = {"lib": {"C": Symbol(kind="class", members={"keep": current})}}
    with pytest.raises(ValueError, match="DEPRECATED_REMOVED lib.C.old"):
        update("record", baseline, modules, "0.2.6", None)
    override = update("record", baseline, modules, "0.2.6", "forced removal")
    assert override.breaks[0].reason == "forced removal"
    assert override.breaks[0].changes == ["DEPRECATED_REMOVED lib.C.old: removed"]
    assert update("record", override, modules, "0.3.0", None).breaks == []
    assert update("record", baseline, modules, "0.3.0", None).line == "0.3"
    with pytest.raises(ValueError, match="release line 0.2 -> 0.3"):
        update("check", baseline, baseline.modules, "0.3.0", None)
    with pytest.raises(ValueError, match="BREAKING lib.C.keep"):
        update("record", baseline, {"lib": {"C": Symbol(kind="class")}}, "0.3.0", None)


def test_enum_numbers_and_config_fields_are_published() -> None:
    baseline = Contract(
        line="0.2",
        breaks=[],
        modules={
            "lib": {
                "Zone": Symbol(
                    kind="enum", members={"HAND": Symbol(kind="enum_member", value=1)}
                ),
                "Config": Symbol(
                    kind="class",
                    members={"limit": Symbol(kind="field", required=False)},
                ),
            }
        },
    )
    changed = {
        "lib": {
            "Zone": Symbol(
                kind="enum", members={"HAND": Symbol(kind="enum_member", value=2)}
            ),
            "Config": Symbol(kind="class"),
        }
    }
    with pytest.raises(ValueError) as exc:
        update("record", baseline, changed, "0.2.6", None)
    assert "BREAKING lib.Config.limit: removed" in str(exc.value)
    assert "BREAKING lib.Zone.HAND: enum value 1 changed to 2" in str(exc.value)


@pytest.mark.parametrize("contents", [None, b"{"])
def test_record_requires_a_readable_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contents: bytes | None,
) -> None:
    path = tmp_path / "public_api.json"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr("scripts.public_api.SNAPSHOT", path)
    monkeypatch.setattr(sys, "argv", ["public_api.py", "record"])
    assert main() == 1
    assert "Restore tests/public_api.json" in capsys.readouterr().err
    assert (path.read_bytes() if path.exists() else None) == contents
