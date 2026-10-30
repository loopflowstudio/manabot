"""manabot.gg: static blog essays, canonical-host redirects and machine routes.

`/` redirects to the current essay; each essay is one self-contained HTML file
in `blog/`.
"""

import os
from pathlib import Path
from urllib.parse import quote

from fasthtml.common import *
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

CANONICAL_HOST = "manabot.gg"
BASE_URL = f"https://{CANONICAL_HOST}"
REDIRECT_HOSTS = {
    "www.manabot.gg",
    "etude.gg",
    "www.etude.gg",
    "etudefantasia.com",
    "www.etudefantasia.com",
    "etudefantasia.gg",
    "www.etudefantasia.gg",
}

class CanonicalHostMiddleware(BaseHTTPMiddleware):
    """301 the alternate domains onto the canonical host."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        host = request.headers.get("host", "").split(":")[0].lower()
        if host in REDIRECT_HOSTS:
            # Starlette URL.path is decoded; retain encoded slashes in old links.
            raw_path = request.scope.get("raw_path")
            path = (
                raw_path.decode("ascii")
                if isinstance(raw_path, bytes)
                else quote(request.url.path, safe="/")
            )
            return RedirectResponse(
                str(request.url.replace(
                    scheme="https", netloc=CANONICAL_HOST, path=path
                )),
                status_code=301,
            )
        return await call_next(request)


app, rt = fast_app(default_hdrs=False, canonical=False)
app.add_middleware(CanonicalHostMiddleware)


# Routes


# The blog is the site.
@rt("/")
def get():
    return RedirectResponse("/blog/lol-cube-kickoff", status_code=302)


async def _robots_handler(request):
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\n\nSitemap: {BASE_URL}/sitemap.xml\n"
    )


async def _sitemap_handler(request):
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(
            f"<url><loc>{BASE_URL}/blog/{p.stem}</loc></url>\n"
            for p in sorted(BLOG_DIR.glob("*.html"))
        )
        + "</urlset>\n"
    )
    return Response(body, media_type="application/xml")


BLOG_DIR = Path(__file__).parent / "blog"


async def _blog_handler(request):
    """Static essays, one self-contained HTML file per slug."""
    slug = request.path_params["slug"]
    f = BLOG_DIR / f"{slug}.html"
    if "/" in slug or ".." in slug or not f.is_file():
        return PlainTextResponse("Not found", status_code=404)
    return Response(f.read_text(), media_type="text/html")


# Insert machine-readable routes ahead of FastHTML's catch-all handlers.
async def _healthz_handler(request):
    return PlainTextResponse("ok")


app.routes.insert(0, Route("/healthz", _healthz_handler, methods=["GET"]))
app.routes.insert(0, Route("/robots.txt", _robots_handler, methods=["GET"]))
app.routes.insert(0, Route("/blog/{slug}", _blog_handler, methods=["GET"]))
app.routes.insert(0, Route("/sitemap.xml", _sitemap_handler, methods=["GET"]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))
