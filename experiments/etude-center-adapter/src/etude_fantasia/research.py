"""Import EXP-3's explicitly enumerated citations from exact Fantasia git objects.

Research repository names are workspace aliases. The bundled manifest resolves
`fantasia` through EXP-3's declared repository URL; arbitrary aliases are not
inferred. Definitions/findings themselves remain owned by the research module.
"""

from __future__ import annotations

from pathlib import Path

from etude_client.api import Client
from etude_client.errors import EtudeError
from etude_client.models import (
    Digest,
    ExperimentRegistration,
    Model,
    ObservationBatch,
    ObservationEnvelope,
    RegistrationReceipt,
    ReportObservation,
    SourceKey,
)

from .reader import Reader
from .transports import LocalSource


class Citation(Model):
    experiment_key: str
    title: str
    label: str
    repository: str
    revision: str
    path: str
    sha256: Digest


class CitationManifest(Model):
    source: str
    citations: tuple[Citation, ...]


def manifest() -> CitationManifest:
    return CitationManifest.model_validate_json(
        Path(__file__).with_name("manabot-evidence.json").read_bytes()
    )


def import_citations(client: Client, root: Path) -> tuple[RegistrationReceipt, ...]:
    citations = manifest().citations
    # Freeze complete envelopes before HTTP writes: service latency must not
    # consume the source-read deadline or trigger another read mid-import.
    readers: dict[str, Reader] = {}
    records: list[tuple[Citation, ObservationEnvelope]] = []
    for citation in citations:
        if not citation.path.startswith("experiments/"):
            raise EtudeError("source_path_denied", 403)
        if citation.revision not in readers:
            readers[citation.revision] = Reader(LocalSource(root, citation.revision))
        reader = readers[citation.revision]
        ref = reader.select(citation.path)
        if ref.sha256 != citation.sha256:
            raise EtudeError("research_citation_unavailable_or_changed", 422)
        records.append(
            (
                citation,
                reader.envelope(
                    ref,
                    repository=citation.repository,
                    observation=ReportObservation(
                        source_key=SourceKey(
                            namespace="fantasia", kind="research-citation", source_id=citation.path
                        ),
                        title=citation.label,
                    ),
                ),
            )
        )
    receipts: dict[str, RegistrationReceipt] = {}
    for citation, record in records:
        namespace, kind, source_id = citation.experiment_key.split(":", 2)
        if citation.experiment_key not in receipts:
            experiment_key = SourceKey(namespace=namespace, kind=kind, source_id=source_id)
            existing = client.find_by_key(experiment_key)
            receipts[citation.experiment_key] = (
                RegistrationReceipt(
                    experiment_id=existing.experiment_id, revision_id=existing.revision_id
                )
                if existing
                else client.register_record(
                    ExperimentRegistration(
                        title=citation.title, origin="import", source_key=experiment_key
                    ),
                )
            )
        receipt = receipts[citation.experiment_key]
        client.ingest(ObservationBatch(experiment_id=receipt.experiment_id, records=(record,)))
    return tuple(receipts.values())
