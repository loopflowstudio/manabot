# Observations and the policy network

At every decision, the network receives what the acting seat can see and returns
two things: a probability for each legal action, and an estimate of who will win.
[How a manabot learns](learning.md) covers what training does with those outputs.

![A viewer-safe observation becomes typed object embeddings, mixed by self-attention. Legal actions gather their focus objects to produce one logit each, and pooled objects produce the value estimate.](images/policy-network.svg)

The [illustrated walkthrough](observation-architecture.html) goes further, with
an interactive tensor explorer and a fact-capacity example. GitHub shows its
source, so download it or open it from a checkout. It is self-contained and
works offline.

## What the network sees

managym owns this boundary. It supplies viewer-safe state and the complete legal
decision before the network sees either, so nothing in the encoder or browser
filters hidden information.

| Input | What it carries |
| --- | --- |
| Players, visible cards, permanents | Typed objects. Permanent rows already include effective characteristics. |
| Local facts and links | Open-ended details such as counters, plus attachment and blocking links. |
| Current continuation | The pending decision: stack entries, targets, modes and suspended payments. |
| Shared effects | Each active effect once, with the objects it affects. |
| Known hand and disclosed decklists | What this seat is allowed to know. |
| Legal actions and focus objects | Every offered action and up to two objects it refers to. |
| Card definitions | Typed rules tokens derived from each card's native definition. |

## What the network does

1. **Embed.** Each visible object gets a typed projection. A card's rules tokens
   pass through a token embedding and a GRU, so the model can read what a card
   does as well as its name. Facts, links and effects add to the object they
   belong to.
2. **Mix.** Self-attention lets objects inform each other, with padding masked.
3. **Score actions.** Each action gathers the post-attention embeddings of its
   focus objects, and a small head gives it one logit. With `policy_film`, a
   pooled game state first scales each action embedding. Illegal slots are
   masked, so every decision type shares one action head.
4. **Estimate value.** Objects are projected, pooled by a masked mean and read
   by the value head. The output is a scalar or loss/draw/win logits.

The GRU encodes a card's rules program, not game history. The network has no
recurrent game memory, and the optional recent-event encoder is off by default.
Each decision is judged from its current observation alone.

## Sizes in the TLA pilot

Each object has 64 channels. Two self-attention blocks, each with four heads, mix
up to 202 padded object slots: two players, 120 cards and 80 permanents. An
action concatenates its 64-channel projection with two 64-channel focus objects,
giving 192 channels, and passes through 192 → 64 → 64 to the policy head.

V7 encodes each complete shared effect once and sums its vector into every
explicitly named affected object before attention. Duplicate memberships retain
multiplicity, and every affected object's loss reaches the shared encoder. Local
facts remain a separate sum. Neither path adds attention tokens.

Complete shared records retain duration, timestamp, controller and exact source
bindings that can change future behavior.

## Input v7 capacity

Select `ObservationConfig(policy_input_version=7)` for the new transport: 128
headers × 18 columns, 512 detail tokens × 3, and 2,048 memberships × 2, plus a
configurable local-fact table (128 rows by default). Overflow fails explicitly.
The retained 140/270-fragment Drill positions become 7/9 complete payloads and
70/135 memberships. All original timestamps and duplicate edges survive; no
rules records are deleted or semantically coalesced.

The three padded effect arrays use 31,744 bytes per observation; local facts add
6,144. This exceeds v6/256's 12,288-byte fact array, although the retained
270-fragment case needs only 1,944 active effect bytes. The walkthrough documents
the layout, routing, parameters and sum/normalization tradeoff. These engineering
proofs establish no learning-quality advantage. Persistent memory remains future
work. Fresh defaults remain v6; frozen v1–v6 checkpoints and the TLA pilot's
protocols/receipts keep their original meanings.
