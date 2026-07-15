"""Offline compiler for curated, reviewed card semantics."""

from .compiler import (
    IR_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    SemanticCompileError,
    compile_source,
    validate_fixtures,
)

__all__ = [
    "IR_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "SemanticCompileError",
    "compile_source",
    "validate_fixtures",
]
