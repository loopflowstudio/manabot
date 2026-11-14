"""Serve the packaged Academy SPA on the same origin as its API."""

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope


class SpaStaticFiles(StaticFiles):
    """Serve the SPA fallback for client-routed paths (/replay).

    Vite emits hashed filenames under _app/immutable (card art included), so
    those get a year-long immutable Cache-Control — browsers and the
    Cloudflare edge fetch each asset once. The fallback shell revalidates.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
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


def frontend_directory() -> Path:
    override = os.environ.get("ACADEMY_FRONTEND_BUILD")
    return (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).parent / "site"
    )
