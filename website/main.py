"""Etude Fantasia landing site.

A single leaf in the Sepia Etude language: parchment grounds, three inks,
bronze display serif, one Mountain-red action, and the WUBRG weave as the
banner's one moment of full color. Copy lives in content.yaml.
"""

import os
from pathlib import Path

import yaml
from fasthtml.common import *
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

BASE_URL = "https://etude.gg"
CANONICAL_HOST = "etude.gg"
REDIRECT_HOSTS = {
    "www.etude.gg",
    "etudefantasia.com",
    "www.etudefantasia.com",
    "etudefantasia.gg",
    "www.etudefantasia.gg",
}

CONTENT = yaml.safe_load((Path(__file__).parent / "content.yaml").read_text())

WUBRG = ["W", "U", "B", "R", "G"]


class CanonicalHostMiddleware(BaseHTTPMiddleware):
    """301 the alternate domains onto the canonical host."""

    async def dispatch(self, request, call_next):
        host = request.headers.get("host", "").split(":")[0].lower()
        if host in REDIRECT_HOSTS:
            return RedirectResponse(
                f"{BASE_URL}{request.url.path}", status_code=301
            )
        return await call_next(request)


DESCRIPTION = CONTENT["homepage"]["hero"]["subline"].strip()

app, rt = fast_app(
    htmlkw={"lang": "en"},
    default_hdrs=False,  # no htmx/CDN scripts — the page is pure document
    canonical=False,  # we set the canonical host explicitly
    hdrs=(
        Meta(charset="utf-8"),
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Meta(name="description", content=DESCRIPTION),
        Meta(property="og:title", content="Etude Fantasia"),
        Meta(property="og:description", content=DESCRIPTION),
        Meta(property="og:image", content=f"{BASE_URL}/static/board-developed.png"),
        Meta(property="og:url", content=BASE_URL + "/"),
        Meta(property="og:type", content="website"),
        Link(rel="icon", href="/static/favicon.svg", type="image/svg+xml"),
        Link(rel="canonical", href=BASE_URL + "/"),
        Link(rel="stylesheet", href="/static/style.css"),
    ),
)
app.add_middleware(CanonicalHostMiddleware)


# Components


def SkipLink():
    return A("Skip to main content", href="#main-content", cls="skip-link")


def Weave():
    """The banner weave: the brand's one fixed-rich stripe, in both modes."""
    return Div(
        *[Span(cls=f"weave-band weave-{c.lower()}") for c in WUBRG],
        cls="weave",
        **{"aria-hidden": "true"},
    )


def Pips():
    return Span(
        *[
            Span(
                Img(src=f"/static/mana/{c}.svg", alt="", width="10", height="10"),
                cls=f"pip pip-{c.lower()}",
            )
            for c in WUBRG
        ],
        cls="pips",
        **{"aria-hidden": "true"},
    )


def Masthead(hero):
    return Header(
        Weave(),
        Nav(
            Div(
                Span("Etude Fantasia", cls="brand"),
                Pips(),
                cls="brand-group",
            ),
            Ul(
                Li(
                    A(
                        "GitHub",
                        href=hero["repo_url"],
                        target="_blank",
                        rel="noopener noreferrer",
                        **{"aria-label": "Etude Fantasia on GitHub (opens in new tab)"},
                    )
                ),
                Li(A("Play", href=hero["play_url"], cls="btn btn-primary")),
                cls="nav-links",
            ),
            cls="container masthead",
            **{"aria-label": "Main navigation"},
        ),
    )


def Hero(hero, showcase):
    return Section(
        Div(
            H1(hero["tagline"], cls="hero-tagline"),
            P(hero["subline"], cls="hero-subline"),
            Div(
                A("Play now", href=hero["play_url"], cls="btn btn-primary"),
                A(
                    "View the source",
                    href=hero["repo_url"],
                    target="_blank",
                    rel="noopener noreferrer",
                    cls="btn btn-secondary",
                    **{"aria-label": "View the source on GitHub (opens in new tab)"},
                ),
                cls="hero-actions",
            ),
            cls="container",
        ),
        Figure(
            Img(
                src=showcase["image"],
                alt=showcase["image_alt"],
                loading="eager",
            ),
            Figcaption(showcase["caption"]),
            cls="container showcase",
        ),
        cls="hero",
    )


def Pillars(pillars):
    return Section(
        Div(
            H2(pillars["heading"]),
            Div(
                *[
                    Div(
                        H3(item["title"]),
                        P(item["description"]),
                        cls=f"pillar register-{item['register']}",
                    )
                    for item in pillars["items"]
                ],
                cls="pillar-grid",
            ),
            cls="container",
        ),
        cls="pillars",
    )


def Steps(play, train):
    def step(block, anchor=None):
        return Div(
            H2(block["heading"], id=anchor),
            P(block["intro"]),
            Pre(Code(block["code"].strip())),
            cls="step",
        )

    return Section(
        Div(
            step(play, anchor="play"),
            step(train),
            cls="container step-grid",
        ),
        cls="steps",
    )


def Research(research):
    return Section(
        Div(
            H2(research["heading"]),
            P(research["body"]),
            P(
                A(
                    research["link_label"],
                    href=research["link_url"],
                    target="_blank",
                    rel="noopener noreferrer",
                    cls="quiet-link",
                    **{"aria-label": f"{research['link_label']} (opens in new tab)"},
                )
            ),
            cls="container",
        ),
        cls="research",
    )


def Colophon(colophon):
    return Footer(
        Div(
            Div(
                P("Etude Fantasia", cls="colophon-title"),
                P(colophon["identity"], cls="colophon-identity"),
                P(
                    A(
                        colophon["built_by"],
                        href=colophon["built_by_url"],
                        target="_blank",
                        rel="noopener noreferrer",
                        cls="quiet-link",
                        **{"aria-label": "Loopflow Studio (opens in new tab)"},
                    ),
                    cls="colophon-built-by",
                ),
            ),
            P(
                "Etude Fantasia is unofficial Fan Content permitted under the ",
                A(
                    "Wizards of the Coast Fan Content Policy",
                    href=colophon["fan_content_policy_url"],
                    rel="external noopener",
                ),
                ". Card art and mana symbols are © Wizards of the Coast. "
                "Not approved or endorsed by Wizards.",
                cls="colophon-legal",
            ),
            cls="container colophon-grid",
        ),
        cls="colophon",
    )


# Routes


@rt("/")
def get():
    home = CONTENT["homepage"]
    return (
        Title("Etude Fantasia — the study of a game that improvises"),
        SkipLink(),
        Masthead(home["hero"]),
        Main(
            Hero(home["hero"], home["showcase"]),
            Pillars(home["pillars"]),
            Steps(home["play"], home["train"]),
            Research(home["research"]),
            id="main-content",
        ),
        Colophon(CONTENT["colophon"]),
    )


async def _robots_handler(request):
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\n\nSitemap: {BASE_URL}/sitemap.xml\n"
    )


async def _sitemap_handler(request):
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"<url><loc>{BASE_URL}/</loc></url>\n"
        "</urlset>\n"
    )
    return Response(body, media_type="application/xml")


# Insert machine-readable routes at the beginning to avoid the static handler
app.routes.insert(0, Route("/robots.txt", _robots_handler, methods=["GET"]))
app.routes.insert(0, Route("/sitemap.xml", _sitemap_handler, methods=["GET"]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))
