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
from .runtime_policy import (
    POLICY_ARMS,
    PolicyTargets,
    RuntimeCatalog,
    RuntimePolicyError,
    RuntimePolicyFeatures,
    RuntimePolicyProjector,
    SemanticRuntimePolicy,
    targets_from_submission,
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
    "RuntimeCatalog",
    "RuntimePolicyError",
    "RuntimePolicyFeatures",
    "RuntimePolicyProjector",
    "SemanticRuntimePolicy",
    "SemanticProjectionError",
    "SubjectBinding",
    "POLICY_ARMS",
    "PolicyTargets",
    "UnadmittedDefinitionError",
    "UnknownOpcodeError",
    "UnknownSchemaError",
    "compile_source",
    "validate_fixtures",
    "targets_from_submission",
]
