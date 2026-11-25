"""Record and compare the structural API promised on a manabot patch line.

Python exports come from the installed modules. Native declarations come from
the managym stub, including its type-only names. A signature retains parameter
names, order, kinds and optionality. An ellipsis stands for any default.
"""

import argparse
import ast
from collections.abc import Callable, Iterator, Mapping
import dataclasses
from importlib import import_module
import inspect
from inspect import Parameter, Signature
from pathlib import Path
import sys
import tomllib
from types import ModuleType
from typing import Literal, TypeAliasType, get_origin
import warnings

from pydantic import BaseModel, Field

from manabot.infra.files import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests/public_api.json"
RECORD = "uv run python scripts/public_api.py record"
PUBLIC_MODULES = (
    "manabot.infra.config",
    "manabot.env.observation",
    "manabot.env.match",
    "manabot.env.memory_store",
    "manabot.training.decks",
    "manabot.backends.torch.policy",
    "manabot.backends.torch.checkpoint",
    "manabot.backends.torch.bot",
    "manabot.backends.torch.arrays",
    "manabot.backends.torch.collection",
    "manabot.backends.torch.rollout",
    "manabot.policy.prepare",
    "manabot.training.rollout.collection",
    "manabot.training.rollout.actions",
    "manabot.training.state.game_errors",
)
OPAQUE_TYPES = {
    "manabot.policy.prepare.Prepared",
    "manabot.env.memory_store.Prefix",
    "manabot.env.memory_store.MemoryArchive",
    "manabot.training.rollout.collection.Collector",
}


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------


class Symbol(BaseModel, extra="forbid"):
    kind: Literal[
        "function",
        "class",
        "opaque",
        "type_alias",
        "value",
        "reexport",
        "property",
        "method",
        "staticmethod",
        "classmethod",
        "attribute",
        "field",
        "enum",
        "enum_member",
        "typed_dict",
    ]
    deprecated: bool = False
    signature: str | None = None
    required: bool | None = None
    value: int | None = None
    members: dict[str, "Symbol"] = Field(default_factory=dict)


class RecordedBreak(BaseModel, extra="forbid"):
    reason: str
    changes: list[str]


class Contract(BaseModel, extra="forbid"):
    line: str
    breaks: list[RecordedBreak]
    modules: dict[str, dict[str, Symbol]]


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------


def signature_text(parameters: list[Parameter]) -> str:
    shape = [
        p.replace(
            annotation=Parameter.empty,
            default=Parameter.empty if p.default is Parameter.empty else None,
        )
        for p in parameters
    ]
    return str(Signature(shape)).replace("=None", "=...")


def stub_parameters(args: ast.arguments) -> list[Parameter]:
    positional = [*args.posonlyargs, *args.args]
    required = len(positional) - len(args.defaults)
    result = [
        Parameter(
            arg.arg,
            Parameter.POSITIONAL_ONLY
            if i < len(args.posonlyargs)
            else Parameter.POSITIONAL_OR_KEYWORD,
            default=Parameter.empty if i < required else ...,
        )
        for i, arg in enumerate(positional)
    ]
    if args.vararg:
        result.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        result.append(
            Parameter(
                arg.arg,
                Parameter.KEYWORD_ONLY,
                default=Parameter.empty if default is None else ...,
            )
        )
    if args.kwarg:
        result.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))
    return result


def callable_symbol(
    function: Callable[..., object],
    kind: Literal["method", "staticmethod", "classmethod", "function"],
) -> Symbol:
    return Symbol(
        kind=kind,
        signature=signature_text(list(inspect.signature(function).parameters.values())),
        deprecated=bool(getattr(function, "__deprecated__", False)),
    )


def python_class(cls: type) -> Symbol:
    if f"{cls.__module__}.{cls.__qualname__}" in OPAQUE_TYPES:
        return Symbol(kind="opaque")
    members: dict[str, Symbol] = {}
    for base in reversed(cls.__mro__):
        if base.__module__.partition(".")[0] not in {"manabot", "managym"}:
            continue
        for name, member in vars(base).items():
            if name.startswith("_") and name != "__init__":
                continue
            if isinstance(member, property):
                members[name] = Symbol(
                    kind="property",
                    deprecated=bool(getattr(member.fget, "__deprecated__", False)),
                )
            elif isinstance(member, staticmethod):
                members[name] = callable_symbol(member.__func__, "staticmethod")
            elif isinstance(member, classmethod):
                members[name] = callable_symbol(member.__func__, "classmethod")
            elif inspect.isfunction(member):
                members[name] = callable_symbol(member, "method")

    if issubclass(cls, BaseModel):
        required = {name: f.is_required() for name, f in cls.model_fields.items()}
    elif dataclasses.is_dataclass(cls):
        required = {
            f.name: f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            for f in dataclasses.fields(cls)
        }
    else:
        required = {}
    members.update(
        {
            name: Symbol(kind="field", required=needed)
            for name, needed in required.items()
            if not name.startswith("_")
        }
    )
    return Symbol(
        kind="class",
        members=members,
        deprecated=bool(getattr(cls, "__deprecated__", False)),
    )


def resolve(module: ModuleType, name: str) -> tuple[object, bool]:
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            value = getattr(module, name)
        except AttributeError as exc:
            raise ValueError(
                f"{module.__name__}.{name} is declared but missing at runtime"
            ) from exc
    return value, any(issubclass(w.category, DeprecationWarning) for w in emitted)


def python_exports(module: ModuleType) -> dict[str, Symbol]:
    names = getattr(module, "__all__", None)
    if names is None:
        raise ValueError(f"{module.__name__} does not define __all__")
    result: dict[str, Symbol] = {}
    for name in names:
        if name.startswith("_") or name in result:
            raise ValueError(f"invalid name in {module.__name__}.__all__: {name}")
        value, warned = resolve(module, name)
        if isinstance(value, type):
            symbol = python_class(value)
        elif inspect.isfunction(value):
            symbol = callable_symbol(value, "function")
        elif isinstance(value, TypeAliasType) or get_origin(value) is not None:
            symbol = Symbol(kind="type_alias")
        else:
            symbol = Symbol(kind="value")
        symbol.deprecated |= warned
        result[name] = symbol
    return result


def decorator_names(node: ast.FunctionDef | ast.ClassDef) -> set[str]:
    return {
        ast.unparse(d.func if isinstance(d, ast.Call) else d)
        for d in node.decorator_list
    }


def is_stub_deprecated(
    node: ast.FunctionDef | ast.ClassDef, decorators: set[str]
) -> bool:
    doc = ast.get_docstring(node) or ""
    return "deprecated" in decorators or doc.startswith("Deprecated")


def stub_function(node: ast.FunctionDef, *, method: bool) -> Symbol:
    decorators = decorator_names(node)
    if "overload" in decorators:
        raise ValueError(
            f"unsupported @overload on public stub declaration: {node.name}"
        )
    kind: Literal["method", "function", "property", "staticmethod", "classmethod"] = (
        "method" if method else "function"
    )
    for decorator in ("property", "staticmethod", "classmethod"):
        if decorator in decorators:
            kind = decorator
    return Symbol(
        kind=kind,
        signature=None
        if kind == "property"
        else signature_text(stub_parameters(node.args)),
        deprecated=is_stub_deprecated(node, decorators),
    )


def stub_class(node: ast.ClassDef, exports: Mapping[str, Symbol]) -> Symbol:
    bases = [ast.unparse(b) for b in node.bases]
    kind: Literal["enum", "class", "typed_dict"] = (
        "enum" if "IntEnum" in bases else "class"
    )
    members: dict[str, Symbol] = {}
    for base in bases:
        if (
            base == "TypedDict"
            or base in exports
            and exports[base].kind == "typed_dict"
        ):
            kind = "typed_dict"
        if base in exports:
            members.update(exports[base].members)
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and (
            item.name == "__init__" or not item.name.startswith("_")
        ):
            members[item.name] = stub_function(item, method=True)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if kind == "typed_dict":
                optional = (
                    isinstance(item.annotation, ast.Subscript)
                    and ast.unparse(item.annotation.value) == "NotRequired"
                )
                member = Symbol(kind="field", required=not optional)
            else:
                member = Symbol(kind="attribute")
            members[item.target.id] = member
        elif kind == "enum" and isinstance(item, ast.Assign):
            members[ast.unparse(item.targets[0])] = Symbol(
                kind="enum_member", value=ast.literal_eval(item.value)
            )
    return Symbol(
        kind=kind,
        members=members,
        deprecated=is_stub_deprecated(node, decorator_names(node)),
    )


def native_exports() -> dict[str, Symbol]:
    module = import_module("managym")
    source = (ROOT / "managym/__init__.pyi").read_text()
    lines = source.splitlines()
    exports: dict[str, Symbol] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            exports[node.name] = stub_class(node, exports)
        elif isinstance(node, ast.FunctionDef):
            exports[node.name] = stub_function(node, method=False)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports[node.target.id] = Symbol(kind="value")
        elif isinstance(node, ast.Assign):
            name = ast.unparse(node.targets[0])
            exports[name] = Symbol(
                kind="type_alias",
                deprecated="Deprecated" in lines[node.lineno - 1],
            )
        elif isinstance(node, ast.ImportFrom) and (
            node.level or (node.module or "").startswith("managym.")
        ):
            for alias in node.names:
                if alias.asname is not None:
                    exports[alias.asname] = Symbol(kind="reexport")
    for name, symbol in exports.items():
        if symbol.kind in {"type_alias", "typed_dict"}:
            continue
        value, warned = resolve(module, name)
        symbol.deprecated |= warned
        if symbol.kind == "enum":
            for member, entry in symbol.members.items():
                if (
                    entry.kind == "enum_member"
                    and int(getattr(value, member)) != entry.value
                ):
                    raise ValueError(
                        f"managym.{name}.{member}: runtime enum value differs from stub ({entry.value})"
                    )
    return exports


# -----------------------------------------------------------------------------
# Compatibility
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Change:
    status: Literal["ADDED", "DEPRECATED_REMOVED", "BREAKING"]
    path: str
    detail: str

    def describe(self) -> str:
        return f"{self.status} {self.path}: {self.detail}"


def parameters(signature: str) -> dict[str, Parameter]:
    fn = ast.parse(f"def f{signature}: pass").body[0]
    assert isinstance(fn, ast.FunctionDef)
    return {p.name: p for p in stub_parameters(fn.args)}


def required(p: Parameter) -> bool:
    return p.default is Parameter.empty and p.kind not in (
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD,
    )


def signature_changes(path: str, before: str, after: str) -> Iterator[Change]:
    old, new = parameters(before), parameters(after)
    for name in sorted(old.keys() - new.keys()):
        yield Change("BREAKING", path, f"parameter {name} removed")
    for name, p in new.items():
        if name not in old:
            if required(p):
                yield Change("BREAKING", path, f"required parameter {name} added")
            else:
                yield Change("ADDED", path, f"optional parameter {name} added")
        elif p.kind != old[name].kind:
            yield Change("BREAKING", path, f"parameter {name} kind changed")
        elif required(p) != required(old[name]):
            yield Change(
                "BREAKING" if required(p) else "ADDED",
                path,
                f"parameter {name} became {'required' if required(p) else 'optional'}",
            )

    def positional(params: Mapping[str, Parameter]) -> list[str]:
        return [
            p.name for p in params.values() if p.kind <= Parameter.POSITIONAL_OR_KEYWORD
        ]

    positions = positional(old)
    if all(name in new and new[name].kind == old[name].kind for name in positions) and (
        positional(new)[: len(positions)] != positions
        or (
            any(p.kind == Parameter.VAR_POSITIONAL for p in old.values())
            and positional(new) != positions
        )
    ):
        yield Change("BREAKING", path, "positional parameter order changed")


def changes(
    before: Mapping[str, Symbol], after: Mapping[str, Symbol], prefix: str
) -> Iterator[Change]:
    for name in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{name}" if prefix else name
        if name not in after:
            yield Change(
                "DEPRECATED_REMOVED" if before[name].deprecated else "BREAKING",
                path,
                "removed",
            )
            continue
        current = after[name]
        if name not in before:
            yield Change(
                "BREAKING" if current.required else "ADDED",
                path,
                f"{current.kind} added",
            )
            continue
        old = before[name]
        if current.kind != old.kind:
            yield Change("BREAKING", path, f"{old.kind} changed to {current.kind}")
            continue
        if current.required != old.required:
            yield Change(
                "BREAKING" if current.required else "ADDED",
                path,
                "field required" if current.required else "field optional",
            )
        if current.deprecated != old.deprecated:
            yield Change(
                "ADDED",
                path,
                "deprecated" if current.deprecated else "deprecation withdrawn",
            )
        if old.value != current.value:
            yield Change(
                "BREAKING", path, f"enum value {old.value} changed to {current.value}"
            )
        if old.signature != current.signature:
            if old.signature and current.signature:
                yield from signature_changes(path, old.signature, current.signature)
            else:
                yield Change("BREAKING", path, "call signature changed")
        yield from changes(old.members, current.members, path)


def update(
    command: str,
    baseline: Contract,
    modules: dict[str, dict[str, Symbol]],
    version: str,
    reason: str | None,
) -> Contract:
    major, minor, _ = version.split(".", 2)
    line = f"{major}.{minor}"
    rollover = line != baseline.line
    if tuple(map(int, line.split("."))) < tuple(map(int, baseline.line.split("."))):
        raise ValueError(f"project line {line} precedes recorded line {baseline.line}")
    delta: list[Change] = []
    for module in sorted(modules.keys() | baseline.modules.keys()):
        delta.extend(
            changes(baseline.modules.get(module, {}), modules.get(module, {}), module)
        )
    blocked = [
        c.describe()
        for c in delta
        if c.status == "BREAKING" or c.status == "DEPRECATED_REMOVED" and not rollover
    ]
    details = "\n".join(c.describe() for c in delta)
    if blocked and not reason:
        raise ValueError(
            f"{details}\n\nmain is on the {line}.x patch line (pyproject.toml: {version}).\n"
            f"Restore the published API or keep a DeprecationWarning alias targeting {major}.{int(minor) + 1}.0.\n"
            f"Only for an unavoidable break: {RECORD} --allow-break --reason '<why>'"
        )
    if command == "check" and (rollover or modules != baseline.modules):
        summary = details or f"release line {baseline.line} -> {line}"
        raise ValueError(f"{summary}\nUpdate the {line} contract with: {RECORD}")
    breaks = [] if rollover else list(baseline.breaks)
    if blocked and reason:
        breaks.append(RecordedBreak(reason=reason, changes=blocked))
    return Contract(line=line, modules=modules, breaks=breaks)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    record = sub.add_parser("record")
    record.add_argument("--allow-break", action="store_true")
    record.add_argument("--reason")
    args = parser.parse_args()
    reason = args.reason.strip() if args.command == "record" and args.reason else None
    if args.command == "record" and args.allow_break != (reason is not None):
        parser.error("--allow-break requires a non-empty --reason '<why>'")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    try:
        try:
            baseline = Contract.model_validate_json(SNAPSHOT.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot read {SNAPSHOT.name}: {exc}\nRestore tests/public_api.json "
                f"from version control before running {RECORD}."
            ) from exc
        modules = {
            module: python_exports(import_module(module)) for module in PUBLIC_MODULES
        }
        modules["managym"] = native_exports()
        result = update(args.command, baseline, modules, version, reason)
        if args.command == "record":
            # Always retain the release line and break history, even when empty.
            atomic_json(SNAPSHOT, result.model_dump(exclude_defaults=True))
            print("recorded tests/public_api.json")
        else:
            print("public API matches tests/public_api.json")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
