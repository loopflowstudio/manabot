"""Validate large TrainingRun JSON with bounded memory, retaining only lineage.

Dense per-step series are traversed, never accumulated. A complete SHA-256/size
check covers even discarded bytes. The resulting projection is not source bytes.
JobRecord remains the authority for displayed progress and execution summaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from etude_client.collector import ObjectRef
from etude_client.errors import EtudeError
from etude_client.models import MAX_BYTES

from .values import JSON, Object, obj

OMISSION = "Dense per-step series were not imported; complete stage ancestry was extracted."
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class ReferencedRun:
    reference: ObjectRef
    size_bytes: int
    summary: Object


class RunStream:
    """Strict JSON traversal; at most one token and selected lineage in memory."""

    def __init__(self, source: BinaryIO, deadline: float) -> None:
        self.source = source
        self.deadline = deadline
        self.buffer = ""
        self.position = 0
        self.digest = hashlib.sha256()
        self.size = 0
        # Incremental decoding also checks UTF-8 in discarded strings.
        import codecs

        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.eof = False

    def peek(self) -> str:
        while self.position == len(self.buffer) and not self.eof:
            if time.monotonic() >= self.deadline:
                raise EtudeError("timeout", 504)
            raw = self.source.read(65536)
            self.digest.update(raw)
            self.size += len(raw)
            self.eof = not raw
            self.buffer = self.decoder.decode(raw, final=self.eof)
            self.position = 0
        return self.buffer[self.position : self.position + 1]

    def space(self) -> str:
        while (char := self.peek()) and char in " \r\n\t":
            self.position += 1
        return char

    def take(self, expected: str) -> None:
        if self.space() != expected:
            raise ValueError("invalid_json")
        self.position += 1

    def scalar(self) -> JSON:
        char = self.space()
        token: list[str] = []
        if char == '"':
            token.append(char)
            self.position += 1
            escaped = False
            while True:
                char = self.peek()
                if not char:
                    raise ValueError("truncated_json")
                token.append(char)
                self.position += 1
                if char == '"' and not escaped:
                    break
                escaped = char == "\\" and not escaped
                if len(token) > MAX_BYTES:
                    raise ValueError("json_token_bound")
        else:
            while char and char not in " \r\n\t,]}":
                token.append(char)
                self.position += 1
                if len(token) > 1024:
                    raise ValueError("json_token_bound")
                char = self.peek()
        text = "".join(token)
        if (
            not text.startswith('"')
            and text not in {"null", "true", "false"}
            and not _NUMBER.fullmatch(text)
        ):
            raise ValueError("invalid_json_scalar")
        value: JSON = json.loads(text)
        return value

    def value(self, path: tuple[str, ...] = (), depth: int = 0) -> JSON:
        if depth > 100:
            raise ValueError("json_depth_bound")
        # Retain all of each origin, not a guessed subset of ancestry fields.
        keep = (
            path in {("id",), ("schema_version",), ("parent_run_id",)}
            or path[:1] == ("recovery_artifact",)
            or path[:3]
            == (
                "stages",
                "*",
                "learning_state_origin",
            )
        )
        traverse = keep or path in {(), ("stages",), ("stages", "*")}
        char = self.space()
        if char == "{":
            self.take("{")
            result: Object = {}
            keys: set[str] = set()
            if self.space() != "}":
                while True:
                    key = self.scalar()
                    if not isinstance(key, str) or key in keys:
                        raise ValueError("invalid_json_key")
                    keys.add(key)
                    self.take(":")
                    value = self.value((*path, key), depth + 1)
                    if traverse and (keep or value is not None or key == "learning_state_origin"):
                        result[key] = value
                    if self.space() != ",":
                        break
                    self.take(",")
            self.take("}")
            return result if traverse else None
        if char == "[":
            self.take("[")
            values: list[JSON] = []
            if self.space() != "]":
                while True:
                    value = self.value((*path, "*"), depth + 1)
                    if traverse:
                        values.append(value)
                        if len(values) > 10000:
                            raise ValueError("lineage_bound")
                    if self.space() != ",":
                        break
                    self.take(",")
            self.take("]")
            return values if traverse else None
        value = self.scalar()
        return value if keep else None


def read_run(path: Path, reference: ObjectRef, size: int, deadline: float) -> ReferencedRun:
    try:
        with path.open("rb") as source:
            stream = RunStream(source, deadline)
            summary = obj(stream.value())
            if (
                stream.space()
                or stream.size != size
                or stream.digest.hexdigest() != reference.sha256
            ):
                raise ValueError("invalid_run_digest_or_size")
            if len(json.dumps(summary).encode()) > MAX_BYTES:
                raise ValueError("lineage_bound")
            return ReferencedRun(reference, size, summary)
    except (ValueError, UnicodeError, RecursionError):
        raise EtudeError("invalid", 422) from None
