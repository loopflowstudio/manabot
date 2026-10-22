"""Hosted play entrypoint: the experience server plus the built frontend.

Wraps academy.server's FastAPI app and mounts the static SPA build so the
client, /api, and /ws/play share one origin — the same shape ./scripts/play
gives locally via the Vite proxy.
"""

import os
from pathlib import Path
from urllib.parse import quote

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from academy.server import app

<<<<<<< HEAD
BUILD_DIR = Path(os.environ.get("ACADEMY_FRONTEND_BUILD", "/app/frontend/build"))
=======
BUILD_DIR = Path(os.environ.get("ACADEMY_FRONTEND_BUILD", "/app/academy/ui/build"))
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f


class PlayDomainMiddleware(BaseHTTPMiddleware):
    """Move legacy HTTP traffic; existing WebSocket clients remain supported."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.hostname == "play.etude.gg":
            # Starlette URL.path is decoded; retain encoded slashes in old links.
            raw_path = request.scope.get("raw_path")
            path = (
                raw_path.decode("ascii")
                if isinstance(raw_path, bytes)
                else quote(request.url.path, safe="/")
            )
            return RedirectResponse(
                str(
                    request.url.replace(
                        scheme="https", netloc="play.manabot.gg", path=path
                    )
                ),
                status_code=308,
            )
        return await call_next(request)


class SpaStaticFiles(StaticFiles):
    """Serve the SPA fallback for client-routed paths (/replay).

    Vite emits hashed filenames under _app/immutable (card art included), so
    those get a year-long immutable Cache-Control — browsers and the
    Cloudflare edge fetch each asset once. The fallback shell revalidates.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
            if response.status_code == 404:
                path = "index.html"
                response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            path = "index.html"
            response = await super().get_response(path, scope)
        if path.startswith("_app/immutable/"):
            response.headers["cache-control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["cache-control"] = "no-cache"
        return response


app.add_middleware(PlayDomainMiddleware)
app.mount("/", SpaStaticFiles(directory=BUILD_DIR, html=True), name="frontend")
