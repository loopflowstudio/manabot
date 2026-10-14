"""One bounded evidence cut for retained files and native object graphs.

Only transports perform I/O. Cache entries are pinned bytes, never mutable keys;
all dependency assertions are checked on every use. Large run summaries cannot
satisfy retained-byte completeness. No observations are published here.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from urllib.parse import urlsplit

from etude_client.collector import ObjectRef, ReadSource
from etude_client.errors import EtudeError
from etude_client.models import (
    MAX_BYTES,
    EvidenceInput,
    Observation,
    ObservationEnvelope,
    sha256,
    utcnow,
)

from .stream_run import OMISSION, ReferencedRun, read_run
from .transports import LocalSource, S3Source
from .values import Object, integer, key, obj, string


class Reader:
    def __init__(
        self,
        source: ReadSource | LocalSource,
        *,
        deadline: float | None = None,
        dependencies: tuple[ObjectRef, ...] | None = None,
    ) -> None:
        self.source = source
        self.deadline = deadline if deadline is not None else time.monotonic() + 45
        self.allowed = {(r.key, r.version): r for r in dependencies or ()}
        if len(self.allowed) != len(dependencies or ()):
            raise EtudeError("invalid", 422)
        self.discover = dependencies is None and isinstance(source, S3Source)
        self.selected: dict[str, ObjectRef] = {}
        self.raw: dict[ObjectRef, bytes] = {}
        self.runs: dict[ObjectRef, ReferencedRun] = {}
        self.total = 0
        self.fetched_at = utcnow()

    def check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            raise EtudeError("timeout", 504)

    def cache(self, ref: ObjectRef, raw: bytes) -> ObjectRef:
        self.check_time()
        if any(r.key == ref.key and r.version == ref.version and r != ref for r in self.raw):
            raise EtudeError("invalid", 422)
        if sha256(raw) != ref.sha256 or self.total + len(raw) > MAX_BYTES:
            raise EtudeError("invalid", 422)
        self.raw[ref] = raw
        self.total += len(raw)
        return ref

    def select(self, path: str) -> ObjectRef:
        """Resolve a mutable path once per cut; another Reader starts a fresh cut."""
        self.check_time()
        if path not in self.selected:
            if not isinstance(self.source, (LocalSource, S3Source)):
                raise EtudeError("source_not_enumerated", 403)
            ref, raw = self.source.select(path, deadline=self.deadline)
            self.selected[path] = ref if ref in self.raw else self.cache(ref, raw)
        return self.selected[path]

    def bytes(self, ref: ObjectRef) -> bytes:
        self.check_time()
        if ref not in self.raw:
            if isinstance(self.source, LocalSource):
                selected, raw = self.source.select(ref.key, deadline=self.deadline)
                if selected != ref:
                    raise EtudeError("invalid", 422)
            else:
                uri = urlsplit(ref.key)
                if (
                    uri.scheme != "s3"
                    or not uri.netloc
                    or uri.query
                    or uri.fragment
                    or ref.version in {"", "null"}
                ):
                    raise EtudeError("invalid", 422)
                raw = self.source.read(ref.key, ref.version, deadline=self.deadline)
            self.cache(ref, raw)
        return self.raw[ref]

    def read(self, ref: ObjectRef) -> Object:
        return obj(json.loads(self.bytes(ref)))

    def dependency(self, value: Object, *, run: bool = False) -> tuple[ObjectRef, Object]:
        ref = ObjectRef(
            key=string(value["uri"]),
            version=string(value["version_id"]),
            sha256=string(value["sha256"]),
        )
        allowed = self.allowed.get((ref.key, ref.version))
        if not self.discover and allowed is None:
            raise EtudeError("forbidden", 403)
        if allowed is not None and allowed != ref:
            raise EtudeError("invalid", 422)
        size = integer(value["bytes"])
        self.check_time()
        if run and size > MAX_BYTES and isinstance(self.source, S3Source):
            if ref not in self.runs:
                with self.source.download(
                    ref.key, ref.version, deadline=self.deadline, large_run=True
                ) as (_, path):
                    self.runs[ref] = read_run(path, ref, size, self.deadline)
            result = self.runs[ref]
            if result.size_bytes != size:
                raise EtudeError("invalid", 422)
            return ref, result.summary
        data = self.bytes(ref)
        if len(data) != size:
            raise EtudeError("invalid", 422)
        return ref, obj(json.loads(data))

    def envelope(
        self,
        ref: ObjectRef,
        observation: Observation,
        *,
        schema: str | None = None,
        source_time: datetime | None = None,
        selector: str = "$",
        generation: int | None = None,
        repository: str = "loopflowstudio/fantasia",
    ) -> ObservationEnvelope:
        native = not isinstance(self.source, LocalSource)
        run = self.runs.get(ref)
        media = "application/json" if native or ref.key.endswith(".json") else "text/markdown"
        return ObservationEnvelope(
            evidence=EvidenceInput(
                repository=repository,
                source_key=key("object" if native else "receipt", ref.key),
                source_uri=ref.key,
                source_version=ref.version if native else None,
                source_revision=ref.version if not native and ref.version != "working" else None,
                source_schema=schema
                or ("fantasia-retained-receipt/1" if media == "application/json" else media),
                media_type=media,
                importer_version="etude-fantasia-native/1" if native else "etude-fantasia/1",
                sha256=ref.sha256,
                content_base64=base64.b64encode(self.bytes(ref)).decode() if run is None else None,
                size_bytes=run.size_bytes if run else None,
                omission_reason=OMISSION if run else None,
                observed_at=source_time if native else None,
                fetched_at=self.fetched_at,
            ),
            observation=observation,
            source_time=source_time,
            selector=selector,
            generation=generation,
        )
