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
from .policy import (
    RuntimeObjectRow,
    RuntimeSubject,
    SemanticDecision,
    SemanticDecisionAdapter,
    SemanticDecisionError,
    SubjectBinding,
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
    "SemanticDecision",
    "SemanticDecisionAdapter",
    "SemanticDecisionError",
    "RuntimeObjectRow",
    "RuntimeSubject",
    "SemanticProjectionError",
    "SubjectBinding",
    "UnadmittedDefinitionError",
    "UnknownOpcodeError",
    "UnknownSchemaError",
    "compile_source",
    "validate_fixtures",
]
