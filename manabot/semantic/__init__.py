"""Offline compiler for curated, reviewed card semantics."""

from .compiler import (
    IR_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    SemanticCompileError,
    compile_source,
    validate_fixtures,
)
from .learning import (
    ArtifactCompatibilityError,
    BoundSemanticPack,
    ContentPackBindingError,
    LearningSchema,
    SemanticArtifactHeader,
    SemanticProjectionError,
    UnadmittedDefinitionError,
    UnknownOpcodeError,
    UnknownSchemaError,
)

__all__ = [
    "IR_SCHEMA_VERSION",
    "ArtifactCompatibilityError",
    "BoundSemanticPack",
    "ContentPackBindingError",
    "LearningSchema",
    "SOURCE_SCHEMA_VERSION",
    "SemanticArtifactHeader",
    "SemanticCompileError",
    "SemanticProjectionError",
    "UnadmittedDefinitionError",
    "UnknownOpcodeError",
    "UnknownSchemaError",
    "compile_source",
    "validate_fixtures",
]
