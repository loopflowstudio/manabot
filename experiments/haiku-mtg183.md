# Haiku versus the final MTG-183 policy

Completed 2026-10-27. Jack Heart requested a pilot against the final MTG-183
checkpoint, then a high-reasoning ten-game screen, then expansion to 100 games.

**Claude Haiku 5.5 won 51 games; MTG-183 won 49.** There were no draws, failed
games or replay mismatches. This sample establishes no clear playing-strength
advantage. Reported OpenRouter charges were **$37.0833537**, before credit fees.

| Measure | Observed |
|---|---:|
| Games | 100 |
| Native Commands, each replayed against a clone | 66,275 |
| Haiku decisions | 33,477 |
| Decisions requiring one protocol repair | 5 |
| Random fallbacks / unpriced attempts | 0 / 0 |
| Prompt tokens | 334,552,252 |
| Completion tokens, including reasoning | 7,256,257 |
| Reasoning tokens (subset of completion) | 7,122,064 |
| Cache-read / cache-write tokens | 0 / 0 |
| Median / p95 model decision latency | 1.386 / 8.780 seconds |
| Mean / minimum / maximum game API cost | $0.37083 / $0.10283 / $0.94343 |

## Protocol and identities

- World w13, pack `elemental-world-v2`, all 100 ordered (Haiku deck, checkpoint
  deck) pairs including ten mirrors. Each pair has one shuffle seed and one seat
  assignment; this is not a repeated, paired-seed tournament. Haiku played first
  50 times and second 50 times. The initial ten-game schedule was retained when
  expanding to the full grid; all seeds and assignments are in `manifest.json`.
- Opponent: MTG-183 final raw policy, run `f3748bcabac8`, update **5,950**,
  331,180,341 learner transitions. Checkpoint SHA-256:
  `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`.
  It has **196,482 parameters**, two attention layers, width 64 and four heads.
  The checkpoint is 1,071,166 bytes. World/input/roster admission passed.
- The policy sampled on CPU with one Torch thread, an isolated RNG and seed
  `opponent_seed + decision_index`; it retained its saved v1 observation encoder.
- Haiku: `anthropic/claude-haiku-5.5`, served by `Claude Platform on AWS`,
  high reasoning effort, 8,192 maximum output tokens including reasoning,
  120-second decision deadline, explicit `win-game` objective, no forced
  temperature, no cache-write opt-in. Hosted slugs are not immutable weights.
- Both agents had public history disabled. Haiku received the complete current
  viewer-safe text observation and legal actions. Its text includes continuation
  details absent from the old v1 policy inputs: these are not identical inputs.
  There was no writable memory or cross-decision conversation transcript.
- Every decision, including forced choices, went through the declared agent.
  Haiku returned one exact legal reference. One invalid/incomplete answer could
  receive a protocol-only repair; a second would trigger recorded seeded uniform
  fallback. HTTP/provider failures stop. No game was retried or overwritten.
- Game bounds were 2,000 Commands, one hour, and a $2 reported-spend stop checked
  before another hosted decision. Up to ten games overlapped remote waiting.
  A failure or $50 reported campaign spend would stop new launches; active games
  could finish. These are launch/decision stop thresholds, not hard billing caps.

The frozen source/content/native snapshot has file-manifest SHA-256
`0dee1f0947a520b52f2bb4e9d0617218e04705dd0ec434a4bf2681ac2ee21bcb`.
It predates the publication cleanup. Results remain bound to those original
bytes; publication does not claim that cleanup reran the paid games.

## Deck outcomes

Rows are Haiku's deck; columns are MTG-183's deck. **H** means Haiku won; **M**
means MTG-183 won. Color abbreviations use B black, G green, R red, U blue, W white.
Each cell is one game and cannot estimate that matchup's win probability.

| Haiku deck | BG | GR | GW | RB | UB | UG | UR | UW | WB | WR | Wins |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| BG | M | M | H | H | H | M | M | H | H | H | 6 |
| GR | M | H | H | H | H | M | H | H | H | H | 8 |
| GW | H | M | H | M | M | M | M | H | H | H | 5 |
| RB | M | H | M | M | H | M | M | H | M | M | 3 |
| UB | M | H | H | M | M | M | M | M | H | H | 4 |
| UG | M | H | H | H | M | H | M | H | M | H | 6 |
| UR | M | M | M | H | H | H | M | H | H | H | 6 |
| UW | M | M | M | M | M | H | M | M | H | M | 2 |
| WB | M | M | H | H | M | H | M | H | M | H | 5 |
| WR | M | H | H | H | H | M | M | H | H | M | 6 |

Haiku won 21/50 on the play and 30/50 on the draw. Deck pairing and shuffle differ
between seats, so this does not establish a causal seat advantage.

## What the initial pilot did and did not establish

The separate low-effort pilot used a protocol-only instruction asking for a
legal action, without an explicit winning objective. MTG-183's red/black beat
Haiku's green/red 23–0 on turn 22. Haiku made 290 decisions, spent $0.2560459 and
used 9,357 reasoning tokens. This pilot is **excluded** from the 100-game result.

A full reconstruction matched every viewer capture, rendered prompt, complete
legal-reference list, provider answer, executed Command, seeded policy choice
and terminal digest. No wiring bug was found by that audit. The instruction was
underspecified for strength evaluation; the subsequent batch changed both the
winning objective and reasoning effort. Different matchups/seeds and both prompt
changes prevent attributing the result difference to reasoning alone.

## Cost and caching implications

The measured requests averaged about **9,992 prompt tokens per attempt** and
reached 21,716 in a single prompt. These full-roster games are substantially larger
than the earlier tiny-deck planning assumption of 3,500 tokens and 200 decisions.
At this batch's usage, linear extrapolation gives **$370.83 for 1,000 games** and
**$3,708.34 for 10,000**, before credit fees. These are workload extrapolations,
not bounds or predictions for other models, prompts or game distributions.

About $33.46 of the batch cost was input and $3.63 was output. Reasoning tokens
are already included in completion tokens and must not be charged twice. Cache
reads/writes were zero. The unchanged renderer puts varying state early, so a
separately versioned, viewer-safe stable-prefix layout remains a useful next
experiment; this batch establishes no cache savings. See the
[adapter and cache discussion](../docs/llm.md#hosted-models-openrouter).

## Retrieve the evidence

The [experiment-data manifest](data.manifest.json) records content hashes and
sizes. Bytes live in the shared experiment cache/R2 store, not git. With the
team's R2 credentials configured, fetch the small aggregate or the entire bundle:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/haiku-mtg183-20261027/analysis.json
uv run --extra artifacts python -m manabot.data fetch experiments/data/haiku-mtg183-20261027
```

The directory contains the exact original `analysis.json`, `manifest.json` and
`source-files.json`, an archive receipt, and `evidence.tar.gz`. The lossless archive
preserves `.runs/haiku-pilot/`, `.runs/haiku-ten/` and `.runs/haiku-hundred/`,
including every capture, prompt, selection, Command, transition, game summary,
checkpoint, source snapshot and original campaign scripts. Python bytecode caches
and temporary `.pending` files are omitted. Credential-pattern scanning found no
OpenRouter keys or long Bearer-token values; credentials were supplied only via
the process environment during execution.

The archive contains 369,119 files (9,799,314,858 uncompressed file bytes) in
1,040,916,791 compressed bytes. Its SHA-256 is
`daa2f074f5c03723a12a1e032f62287ea3447fea36504c8923577b4f5416c656`.
`cost-probe.json` separately retains the earlier tiny-deck tokenizer/layout probe;
it is not part of the 100-game sample.

Extract into a fresh directory to preserve any existing local evidence:

```sh
mkdir -p .runs/haiku-results-restored
tar -xzf experiments/data/haiku-mtg183-20261027/evidence.tar.gz -C .runs/haiku-results-restored
```

Original absolute paths inside receipts are provenance and are not rewritten.
The first ten games are under `haiku-ten/game-00` through `game-09`; the other
90 are under `haiku-hundred/game-010` through `game-099`, beneath the extracted
`.runs/`. Original orchestration scripts retain those historical paths and are
not a portable campaign API. Inspecting cached evidence makes no hosted requests.
For new paid games use the [current runner](../docs/llm.md#hosted-models-openrouter), a new output
directory and explicit model, checkpoint, match and cost limits.
