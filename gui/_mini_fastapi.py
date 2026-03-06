"""
_mini_fastapi.py
Tiny fallback surface for environments without FastAPI installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import re
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = int(status_code)
        self.detail = detail
        super().__init__(str(detail))


class WebSocketDisconnect(Exception):
    pass


_DISCONNECT = object()


class WebSocket:
    def __init__(self):
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        self._outgoing: asyncio.Queue[Any] = asyncio.Queue()
        self._accepted = False

    async def accept(self) -> None:
        self._accepted = True

    async def receive_json(self) -> Any:
        payload = await self._incoming.get()
        if payload is _DISCONNECT:
            raise WebSocketDisconnect()
        return payload

    async def send_json(self, payload: Any) -> None:
        await self._outgoing.put(payload)

    async def close(self) -> None:
        return None


@dataclass
class _Route:
    method: str
    path: str
    pattern: re.Pattern[str]
    param_names: list[str]
    handler: Callable[..., Any]


class FastAPI:
    def __init__(self, *_, **__):
        self.http_routes: list[_Route] = []
        self.websocket_routes: dict[str, Callable[..., Any]] = {}

    def get(self, path: str):
        pattern, param_names = _compile_path(path)

        def decorator(handler: Callable[..., Any]):
            self.http_routes.append(
                _Route(
                    method="GET",
                    path=path,
                    pattern=pattern,
                    param_names=param_names,
                    handler=handler,
                )
            )
            return handler

        return decorator

    def websocket(self, path: str):
        def decorator(handler: Callable[..., Any]):
            self.websocket_routes[path] = handler
            return handler

        return decorator


class _Response:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class _WebSocketSession:
    def __init__(self, handler: Callable[[WebSocket], Any]):
        self._handler = handler
        self._loop = asyncio.new_event_loop()
        self._websocket = WebSocket()
        self._thread_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._handler(self._websocket))
        except Exception as exc:  # pragma: no cover - surfaced on close
            self._thread_error = exc
        finally:
            self._loop.close()

    def send_json(self, payload: Any) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._websocket._incoming.put(payload),
            self._loop,
        )
        future.result(timeout=30)

    def receive_json(self) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._websocket._outgoing.get(),
            self._loop,
        )
        return future.result(timeout=30)

    def close(self) -> None:
        if self._loop.is_closed():
            return

        with suppress_exceptions():
            asyncio.run_coroutine_threadsafe(
                self._websocket._incoming.put(_DISCONNECT),
                self._loop,
            ).result(timeout=30)

        self._thread.join(timeout=30)

        if self._thread_error and not isinstance(
            self._thread_error,
            WebSocketDisconnect,
        ):
            raise self._thread_error

    def __enter__(self) -> "_WebSocketSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class TestClient:
    def __init__(self, app: FastAPI):
        self.app = app

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def get(self, url: str) -> _Response:
        split = urlsplit(url)
        path = split.path
        query = parse_qs(split.query)

        for route in self.app.http_routes:
            if route.method != "GET":
                continue
            match = route.pattern.match(path)
            if not match:
                continue

            kwargs: dict[str, Any] = match.groupdict()
            kwargs.update(_coerce_query_params(route.handler, query))

            try:
                payload = asyncio.run(route.handler(**kwargs))
            except HTTPException as exc:
                return _Response({"detail": exc.detail}, status_code=exc.status_code)
            return _Response(payload, status_code=200)

        return _Response({"detail": "Not Found"}, status_code=404)

    def websocket_connect(self, path: str) -> _WebSocketSession:
        split = urlsplit(path)
        route_path = split.path
        handler = self.app.websocket_routes.get(route_path)
        if handler is None:
            raise RuntimeError(f"WebSocket route not found: {route_path}")
        return _WebSocketSession(handler)


class suppress_exceptions:
    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> bool:
        return True


def _compile_path(path: str) -> tuple[re.Pattern[str], list[str]]:
    if path == "/":
        return re.compile(r"^/$"), []

    parts = [part for part in path.split("/") if part]
    pattern_parts: list[str] = []
    param_names: list[str] = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            name = part[1:-1]
            param_names.append(name)
            pattern_parts.append(fr"(?P<{name}>[^/]+)")
        else:
            pattern_parts.append(re.escape(part))

    pattern = re.compile(r"^/" + "/".join(pattern_parts) + r"$")
    return pattern, param_names


def _coerce_query_params(
    handler: Callable[..., Any],
    query: dict[str, list[str]],
) -> dict[str, Any]:
    signature = inspect.signature(handler)
    result: dict[str, Any] = {}

    for key, values in query.items():
        if key not in signature.parameters or not values:
            continue

        raw = values[-1]
        annotation = signature.parameters[key].annotation
        annotation_name = annotation if isinstance(annotation, str) else None
        if annotation is bool or annotation_name == "bool":
            result[key] = raw.lower() in {"1", "true", "yes", "on"}
        elif annotation is int or annotation_name == "int":
            result[key] = int(raw)
        else:
            result[key] = raw

    return result
