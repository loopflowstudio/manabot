# Etude center adapter (not run)

This folder holds the code that publishes Fantasia's experiment records to the
Etude research center. **Nothing here runs by default.** It is not installed,
not imported by manabot or the play product, and not collected by `pytest` or
CI (`testpaths = ["tests"]`). Fantasia does not depend on Etude.

It was written in the private repository now named `loopflowstudio/etude-studio`
(formerly `loopflowstudio/etude`, EXP-2) and
moved here on 2026-10-10 at Jack Heart's direction, because code that reads
Fantasia's record formats belongs with Fantasia. Its last commit there is
`53a56cb`. The Etude center keeps a general API and no Fantasia-specific code.

## What it does

`etude_fantasia` reads retained receipts from a Fantasia checkout, or explicitly
enumerated native job objects from S3 or pinned HTTPS links, and posts them as
registrations, observations and evidence references through `etude_client`, the
Etude center's open client. It only reads Fantasia's storage: no writes, no
contact with jobs, no listing beyond what the caller names.

- `reader.py` owns byte reading, size and digest checks and evidence envelopes.
- `fantasia.py` translates retained historical records; `live_fantasia.py`
  translates native job records and walks their dependencies.
- `transports.py` holds the local Git/file and S3 read transports.
- `stream_run.py` is a streaming parser for run records too large to load
  (one is 242 MB); it hashes the whole source and keeps ancestry, not series.
- `research.py` and `manabot-evidence.json` publish revision-pinned citations.
- `cli.py` is the single entry point.

## What is needed to run it

- The legacy `etude_client` API used at `etude-studio` revision `53a56cb`.
  The public `loopflowstudio/etude` repository now provides the distinct
  `etude-sdk` package; do not assume it satisfies this adapter's dependency.
  `pyproject.toml` names `etude_client`, but nothing resolves it here.
- `ETUDE_URL` and `ETUDE_TOKEN_FILE` pointing at a private write token for an
  Etude organization. Never a database connection string.
- Read access to the records being published.

```sh
etude-fantasia retained PATH_TO_FANTASIA capacity-recovery-v3 --revision COMMIT
etude-fantasia s3 EXPERIMENT_UUID s3://BUCKET/PREFIX JOB_ID
```

`--revision` pins an import to an exact commit. Without it, identical bytes read
at a newer revision are recorded as new provenance.

## Tests

`tests/` is kept as a record of the behaviour that was verified, not as a
runnable suite. Several tests import the Etude server and its PostgreSQL test
fixtures to check ingestion end to end, and those do not exist here. They last
passed in the Etude repository at `53a56cb`. Making them runnable here means
replacing those parts with calls against a running Etude center.
