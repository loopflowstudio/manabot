# Provide Exact Study Fork and Return

## Contract

Game already owns the canonical `erd1` replay address and the exact
viewer-safe row it restores. Rules already owns exact `Env::fork` clones and
the structured offer/command path. This slice joins those seams without
serializing private engine state or replaying legacy action indices.

At each deliberate canonical decision, `GameSession` retains an
authority-private environment clone under the row's global ordinal. Once the
match is finalized, a `StudyForkProvider` validates those roots against the
canonical replay. An authorized `erd1` address creates an ephemeral branch
from its retained root. The branch exposes only the recorded viewer's
structured offers and fixed-viewer observations, applies the ordinary
structured submission path, and consumes itself when returning the canonical
replay row and presentation cursor.

The provider uses the selected production representation: compact full clone.
Branch commands never append to the recorded trace, canonical decisions, or
presentation tracks. Unknown and cross-viewer addresses fail identically.

## Scope

- In-process roots for a completed, retained `GameSession`.
- Player-0 Study authorization at the current local human-versus-bot surface.
- Structured priority/attacker decisions currently supported by managym.
- One fixed-viewer observation hook on `Env` and its Python binding.

Durable engine-root persistence/reconstruction, Study HTTP/UI navigation,
opponent automation inside a branch, and broader structured decision coverage
remain outside this slice.

## Done when

- A pinned completed match resolves a canonical player-0 address to an exact
  branch root.
- A normal structured alternative advances only the branch and returns a
  fixed-player projection with no opposing hand identities.
- One return yields the byte-identical recorded frame, offer, command, and
  presentation cursor, closes the branch, and a fresh fork reproduces the
  original offer surface.
- Malformed, missing, and cross-viewer addresses fail closed.
- Focused Rust debug tests and Etude branch integration tests pass.
