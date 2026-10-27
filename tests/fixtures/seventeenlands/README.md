# Pinned offline parser

`mtga_follower.py` is unmodified from rconroy293/mtga-log-client commit
`b87160b65cfe5be1ff9f526cecacf6405147b198`, path
`src/python/seventeenlands/mtga_follower.py`, fetched 2026-10-23.
SHA-256: `71117bdd8b1ef75cc019bbddfaeab9a69a02c5ef9efd741ec05a2e660918f77d`.
Upstream's GPL-3.0 license is retained in `LICENSE`.

This test-only source is not imported by Manabot or shipped in its wheel.
The subprocess harness replaces the submission client with an in-memory sink,
refuses sockets, and reads no account credentials. Every parser error fails the
conformance test. No synthetic submissions leave the harness.

The adjacent `arena-card-log` directory retains the full Scryfall printing
responses for ANA Mountain (70504) and Forest (70505), fetched 2026-10-23.
Their `uri` fields identify the source; native fixture keys bind these explicit
printings. These are parser goldens, not independent Arena reference logs or
proof of downstream 17Lands metric parity. The fixture intentionally includes
return and selection hand sightings that are not native GIH draws.
