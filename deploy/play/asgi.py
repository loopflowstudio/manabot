"""Hosted play entrypoint: the experience server plus the built frontend.

Wraps academy.server's FastAPI app and mounts the static SPA build so the
client, /api, and /ws/play share one origin — the same shape ./scripts/play
gives locally via the Vite proxy.
"""

from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from managym.academy.server import app
from managym.academy.site import SpaStaticFiles, frontend_directory

BUILD_DIR = frontend_directory()


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




app.add_middleware(PlayDomainMiddleware)
app.mount("/", SpaStaticFiles(directory=BUILD_DIR, html=True), name="frontend")
