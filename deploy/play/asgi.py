"""Hosted play entrypoint: the experience server plus the built frontend.

Wraps etude.server's FastAPI app and mounts the static SPA build so the
client, /api, and /ws/play share one origin — the same shape ./scripts/play
gives locally via the Vite proxy.
"""

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from etude.server import app

BUILD_DIR = Path(os.environ.get("ETUDE_FRONTEND_BUILD", "/app/frontend/build"))


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


app.mount("/", SpaStaticFiles(directory=BUILD_DIR, html=True), name="frontend")
