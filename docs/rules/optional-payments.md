# Optional payments

A resolving optional cost is authored as `Effect::OptionalPayment` with a
`PaymentCost` and `if_paid` / `if_declined` effect lists. `PaymentCost::Mana`
uses ordinary mana; `PaymentCost::Waterbend` permits creature/artifact taps
against a generic obligation. For example, another optional waterbend with a
life penalty can use `Waterbend { generic: 2 }`, an empty paid list, and
`[Effect::DealDamageToController { amount: 2 }]` as its declined list. No new
decision or card-name dispatch is needed.

`flow::decision::PaymentProgress` owns the payer, remaining cost, exact objects
already contributed to waterbend, and outcome branches. The existing
`EffectFrame` owns the source and its incarnation, controller, targets and
remaining effect sequence. Completion prepends exactly one outcome branch to
that frame. Counter-unless-paid effects use the same payment progress, with
the target spell's controller as payer and a counter effect on decline.

Waterbending Lesson draws three cards before offering waterbend {2}; decline
prepends the ordinary mandatory discard decision. It cannot switch to discard
after a waterbend tap. Activating a mana ability does not itself close decline:
its activation costs, taps, sacrifices and chosen output are committed even
when the optional payment is subsequently declined. Waterbend taps are cost
payment, never tapping for mana. Existing mana feasibility and production own
restrictions, granted abilities and source consumption; clients do not duplicate
them. Chosen-color mana suspends and resumes the same payment and effect frame.

The authored opcode is 65, `optional_payment`, for example:

```json
{
  "op": "optional_payment",
  "cost": {"waterbend": 2},
  "if_paid": [],
  "if_declined": [{"op": "discard"}]
}
```

Use `{"mana": "{2}{U}"}` for an ordinary mana obligation. Both branches are
recursively compiled, validated and projected into policy input. Frozen opcode
61 (`waterbend_or_discard`) remains readable only through load-time lowering to
`Step::OptionalPayment` at the IR read boundary; it has no separate Step,
Effect or Decision.

The ten-deck pack is `elemental-world-v2`. Its w11 source, IR, fixtures and
learning schema 7 were in `content/semantic/w11` and now live at the revisions
that recorded them; current decklists are in
`content/semantic/current/setups.json`. Academy's version 2 asset manifest binds
the same ten exact setups and local treatments to the new content. Native action kinds,
order and observation tensor dimensions remain unchanged: ordinary optional
mana exposes PayOrNot; waterbend exposes Waterbend; immediate colors expose
Modal; decline may expose Discard. All choices use the same native legal offers
for humans and policies. No human-only payment workflow is introduced.

World w11 and match-state hash version 8 identify changed serialized payment
state and content. Old checkpoints are rejected; historical receipts, `erd1`
addresses and previous conformance corpora are retained without rewriting.
The original optional-payment replay evidence belongs to `experiments/data/archive/conformance/semantic-kernel-w11-v1`;
current-runtime replay evidence belongs to `experiments/data/archive/conformance/semantic-kernel-w13-v1`.
Native clone/undo and Command replay include payment progress, nested mana
choices and contributed-object evidence. Recovery replays the ordinary Commands;
it does not reconstruct payment by interpreting the card's text.

Upfront mana costs use the same payment transition model as offers and ordinary
production. `Decision::ManaUpfront` reserves its exact source and carries a
recursive parent continuation. A Treasure color can finish a filter's upfront
cost, exposing the filter color before restoring the original optional payment.
Waterbend contributions still close decline; mana activation costs remain
committed when declining is legal. No resolving frame or announced target list
is replaced by a mana ability's empty rule frame.

The repaired source has separate replay evidence at
`experiments/data/archive/conformance/semantic-kernel-w11-payment-v1`. The earlier w11 and w10 corpora
retain their bytes and source meanings. The correction introduces no tensor or
authored-content change relative to current w11 and makes no new scientific
checkpoint compatibility claim.
