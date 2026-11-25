// payment.rs
// Paying a cost. An announced cast or activation is a `Payment`: its
// `Operation`, the `CostComponent`s still owed and the receipt of what has
// paid. A payment made during resolution is a `PaymentProgress` owing one
// component. Both pay pips through `pay_pip_with` and share the waterbend and
// life steps; a new cost kind is one `CostComponent` variant, its line in
// `Game::cost_components` and its arms here.

use crate::{
    agent::action::{Action, AgentError},
    flow::{
        cast_permission::PlayRoute,
        decision::{Decision, EffectFrame, FrameCompletion, ResolutionRef, SpendCompletion},
        game::{ActivationSource, CastChoices, Game, PendingChoice},
        proposed_event::{CounterProvenance, EventBinding, EventCause, ProposedEvent},
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{Effect, PaymentCost},
        card::{ActivatedAbilityDefinition, CardType, SacrificeCost, TapCost},
        game_object::{ObjectId, PermanentId, PlayerId, Target},
        mana::{Color, ManaCost},
        payment::{How, PaymentReceipt, Toward},
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// A fully announced spell or activation (CR 601.2a-e, 602.2).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Operation {
    Spell {
        #[serde(rename = "card")]
        object: ObjectId,
        #[serde(flatten)]
        cast: CastChoices,
        mode: Option<usize>,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    },
    Activation {
        source: ActivationSource,
        ability_index: usize,
        targets: Vec<Target>,
        /// The clause of each target (`ActivatedTargeting::target_req_indices`).
        #[serde(skip_serializing_if = "Vec::is_empty")]
        target_req_indices: Vec<usize>,
        /// The announced value of X (CR 602.2b, 601.2b).
        #[serde(skip_serializing_if = "crate::state::is_zero")]
        x: i32,
    },
}

impl Operation {
    /// The spell being cast, if this is a cast.
    pub fn spell(&self) -> Option<ObjectId> {
        match self {
            Self::Spell { object, .. } => Some(*object),
            Self::Activation { .. } => None,
        }
    }
}

/// One part of a total cost. The payer chooses how to pay each component up
/// to and including `Mana`; the ones after it (life and what the source
/// itself pays) have no choice and are paid as the payment completes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum CostComponent {
    /// Tap other untapped creatures until their total power reaches `power`
    /// (crew, CR 702.122a). `power` is what is still owed.
    TapPower {
        power: i32,
        toward: Toward,
    },
    /// "Tap N untapped [predicate] you control"; `count` is what is still
    /// owed, and the second value is the part of the cost the taps pay. A
    /// station tap is another creature than the source (CR 702.184a).
    Tap(TapCost, Toward),
    /// "Discard a card".
    Discard,
    /// Exile a matching card from hand rather than pay the mana cost
    /// (`AdditionalCost::ExileFromHandAlternative`).
    ExileFromHand,
    /// Sacrifice a matching permanent, or pay its `or_mana` instead; the
    /// second value is the part of the cost the sacrifice pays.
    Sacrifice(SacrificeCost, Toward),
    /// Forage (CR 701.61a) or pay `or_mana`: sacrifice a Food, or exile
    /// three cards from the payer's graveyard, `exiled` of them so far. The
    /// first card exiled commits to that branch.
    Forage {
        or_mana: ManaCost,
        exiled: u8,
    },
    /// The mana still owed, and how its payer may pay part of it without
    /// mana. Each such payment is a contribution and reduces `cost`.
    Mana {
        cost: ManaCost,
        help: Help,
    },
    /// The artifact or creature just tapped could pay more than one thing
    /// of the mana owed (`Game::help_taps`): choose which, such as a color
    /// or {1} by convoke (CR 702.51a), or {1} by convoke or by waterbending.
    TapPays {
        tapped: ObjectId,
        options: Vec<TapPayment>,
    },
    /// Remove `remaining` more counters, of any kinds, from among creatures
    /// the payer controls (Quilled Greatwurm), one at a time. Each removal
    /// commits at once. `choosing_from` is the creature just chosen while
    /// its payer names which of its several kinds to remove.
    RemoveCounters {
        remaining: u8,
        choosing_from: Option<ObjectId>,
    },
    /// "Pay N life" (CR 119.4): an ability's life cost, or a spell's life
    /// paid rather than its other cost.
    Life(u32),
    #[serde(rename = "RemoveCounters")]
    RemoveSourceCounters(CounterKind, i32),
    /// "Put N [kind] counters on this" (CR 118.3): always payable while the
    /// source is on the battlefield.
    AddCounters(CounterKind, u16),
    /// What the source itself pays: its own {T} (`Tapped`), "sacrifice this"
    /// (`Sacrificed`), "exile this" (`Exiled`) or, for an ability that
    /// functions from hand, "discard this card" (`Discarded`).
    Source(How),
    ExileGrantor(ObjectId),
    TapGrantor(ObjectId),
}

impl CostComponent {
    /// Mana owed with no help.
    pub fn mana(cost: ManaCost) -> Self {
        Self::Mana {
            cost,
            help: Help::default(),
        }
    }

    /// What a resolving effect's `cost` owes as its payment decision is
    /// made: the one place a definition's `PaymentCost` becomes a component.
    /// A waterbend cost is its generic mana, all of which taps may pay.
    pub fn resolving(cost: &PaymentCost) -> Self {
        match cost {
            PaymentCost::Mana(cost) => Self::mana(cost.clone()),
            PaymentCost::Waterbend { generic } => Self::Mana {
                cost: ManaCost::default().with_generic(*generic),
                help: Help {
                    waterbend: *generic,
                    ..Help::default()
                },
            },
            PaymentCost::Discard => Self::Discard,
            PaymentCost::Life(life) => Self::Life(*life),
        }
    }
}

/// `object` paid one pip of `cost` without mana, a pip of `color` or {1} for
/// `Color::Generic`: the cost shrinks and `receipt` gains the contribution.
pub(crate) fn pay_pip_with(
    cost: &mut ManaCost,
    receipt: &mut PaymentReceipt,
    object: ObjectId,
    how: How,
    toward: Toward,
    color: Color,
) {
    *cost = cost.less_pip(color);
    receipt.contribute(object, how, toward);
}

/// The ways a payer may pay part of a mana cost with something other than
/// mana, fixed as the cost is announced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Help {
    /// Generic mana that taps of untapped artifacts and creatures may pay,
    /// {1} each (waterbend), as announced; taps made are the receipt's
    /// `Toward::Waterbend` contributions.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub waterbend: u8,
    /// Each creature tapped pays {1} or one mana of its color (CR 702.51).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub convoke: bool,
    /// Each artifact tapped pays {1} (CR 702.126).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub improvise: bool,
    /// Each card exiled from the payer's graveyard pays {1} (CR 702.66).
    /// Holds the spell being cast, which its own delve cannot exile.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::state::game_object::exact"
    )]
    pub delve: Option<ObjectId>,
}

impl Help {
    /// Whether any tap or exile may pay beside waterbend's.
    pub fn keyworded(self) -> bool {
        self.convoke || self.improvise || self.delve.is_some()
    }

    /// The generic part of `cost` that waterbend taps may still pay: what
    /// was announced, less the taps already on `receipt`.
    pub fn waterbend_left(self, cost: &ManaCost, receipt: &PaymentReceipt) -> u8 {
        let tapped = receipt.contributed(How::Tapped, Toward::Waterbend).count() as u8;
        self.waterbend.saturating_sub(tapped).min(cost.generic())
    }
}

/// What one direct tap of an artifact or creature pays of a mana cost: one
/// pip of `color`, or {1} for `Color::Generic`, under `toward` (convoke,
/// improvise or waterbend), which the receipt keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TapPayment {
    pub toward: Toward,
    pub color: Color,
}

impl TapPayment {
    /// The `ChooseMode` code naming this payment in a tap-pays choice, the
    /// same in every decision: 0-4 one {W}/{U}/{B}/{R}/{G} pip by convoke,
    /// 5 {1} by convoke, 6 {1} by improvise, 7 {1} of the waterbend part.
    /// Only feasible payments are offered, so a choice's modes are sparse.
    pub fn mode(self) -> usize {
        match (self.toward, self.color) {
            (Toward::Convoke, Color::Generic) => 5,
            (Toward::Convoke, color) => {
                debug_assert!((color as usize) < 5, "convoke pays a colored pip or {{1}}");
                color as usize
            }
            (Toward::Improvise, _) => 6,
            (Toward::Waterbend, _) => 7,
            (toward, _) => unreachable!("a direct tap does not pay toward {toward:?}"),
        }
    }
}

/// One cast or activation being paid for (CR 601.2f-h): the announced
/// operation, the cost components still owed in the order they are paid, and
/// the receipt of what has been paid so far. The total cost is fixed when
/// `Game::cost_components` builds `owed`; payment reads `owed`, never the
/// definition again. There is exactly one `Mana` component: it stays until
/// the mana is spent, and what follows it is paid as the payment completes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Payment {
    pub operation: Operation,
    pub owed: Vec<CostComponent>,
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub receipt: PaymentReceipt,
}

impl Payment {
    fn mana_component(&self) -> (&ManaCost, Help) {
        self.owed
            .iter()
            .find_map(|component| match component {
                CostComponent::Mana { cost, help } => Some((cost, *help)),
                _ => None,
            })
            .expect("a payment has its mana component")
    }

    /// Whether every choice before the mana is made: the step is the mana.
    pub fn paying_mana(&self) -> bool {
        matches!(self.owed[0], CostComponent::Mana { .. })
    }

    /// The life this payment takes as it completes.
    pub fn life(&self) -> u32 {
        self.owed
            .iter()
            .map(|component| match component {
                CostComponent::Life(life) => *life,
                _ => 0,
            })
            .sum()
    }

    /// Whether completing this payment pays `source_cost` (a `Source`
    /// component).
    pub fn owes(&self, source_cost: &CostComponent) -> bool {
        self.owed.contains(source_cost)
    }

    /// The mana still owed and the generic part of it that waterbend taps
    /// may still pay.
    pub fn mana(&self) -> (&ManaCost, u8) {
        let (cost, help) = self.mana_component();
        (cost, help.waterbend_left(cost, &self.receipt))
    }

    /// How the mana owed may be paid without mana, as announced.
    pub fn help(&self) -> Help {
        self.mana_component().1
    }

    /// Whether the cost being paid is a waterbend cost, however it is paid.
    pub fn waterbends(&self) -> bool {
        self.help().waterbend > 0
    }

    /// The mana owed, its help and the receipt, for a step that pays part
    /// of the mana.
    pub(crate) fn mana_step(&mut self) -> (&mut ManaCost, Help, &mut PaymentReceipt) {
        let (cost, help) = self
            .owed
            .iter_mut()
            .find_map(|component| match component {
                CostComponent::Mana { cost, help } => Some((cost, *help)),
                _ => None,
            })
            .expect("a payment has its mana component");
        (cost, help, &mut self.receipt)
    }

    /// Owe `extra` mana as well (a sacrifice cost's "or pay" branch).
    pub(crate) fn owe_mana(&mut self, extra: &ManaCost) {
        let cost = self.mana_step().0;
        *cost = cost.plus(extra);
    }

    /// `object` paid one pip of the mana owed without mana: a pip of `color`,
    /// or {1} for `Color::Generic`.
    pub(crate) fn help_paid(&mut self, object: ObjectId, how: How, toward: Toward, color: Color) {
        let (cost, _, receipt) = self.mana_step();
        pay_pip_with(cost, receipt, object, how, toward, color);
    }
}

/// What one payment made during resolution still owes, what has paid toward
/// it and the branches its outcome selects. Mana activation costs commit
/// immediately through the mana authority. Only a waterbend contribution
/// closes the decline branch (the existing contract). The receipt names the
/// exact objects that paid, as evidence; they are never followed or untapped.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PaymentProgress {
    /// Acceptance of an optional payment precedes choosing its sources in auto mode.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub committed: bool,
    pub player: PlayerId,
    /// `Mana`, whose help allows waterbend taps for a waterbend cost, or
    /// `Life` or `Discard` (`CostComponent::resolving`).
    pub owed: CostComponent,
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub receipt: PaymentReceipt,
    pub if_paid: Vec<Effect>,
    pub if_declined: Vec<Effect>,
    /// The once-each-turn option `(exact source, printed face, ability index)` paying
    /// spends ("Do this only once each turn").
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub spends: Option<OnceEachTurnKey>,
    /// The spell or ability declining counters ("counter it unless that
    /// player pays"): ward answers both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counters_on_decline: Option<Target>,
}

impl PaymentProgress {
    pub fn new(
        player: PlayerId,
        owed: CostComponent,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    ) -> Self {
        Self {
            player,
            owed,
            receipt: PaymentReceipt::default(),
            committed: false,
            if_paid,
            if_declined,
            spends: None,
            counters_on_decline: None,
        }
    }

    pub fn can_decline(&self) -> bool {
        !self.committed && self.receipt.contributions().is_empty()
    }

    /// Whether the cost is a waterbend cost, however it is paid.
    pub fn waterbends(&self) -> bool {
        matches!(self.owed, CostComponent::Mana { help, .. } if help.waterbend > 0)
    }
}

impl Game {
    /// The cost of a resolution payment is paid: spend its once-each-turn
    /// option, then run the "if paid" branch before the rest of `frame`.
    pub(crate) fn finish_resolution_payment(
        &mut self,
        payment: PaymentProgress,
        mut frame: EffectFrame,
    ) {
        if let Some(key) = payment.spends {
            self.spend_once_each_turn(key);
        }
        if payment.waterbends() {
            self.record_bending(payment.player, crate::flow::turn::Bending::Water);
        }
        frame.prepend_effects(payment.if_paid.into_iter());
        self.run_frame(frame);
    }
}

impl Game {
    pub(crate) fn payment_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
        cost: &SacrificeCost,
    ) -> Vec<PermanentId> {
        self.sacrifice_candidates(player, &cost.predicate)
            .into_iter()
            .filter(|id| {
                !cost.another
                    || !matches!(operation, Operation::Activation { source: ActivationSource::Permanent(permanent), .. } if permanent == id)
            })
            .collect()
    }

    pub(super) fn exile_payment_objects(
        &self,
        player: PlayerId,
        payment: &Payment,
    ) -> Vec<ObjectId> {
        let Operation::Spell { object, cast, .. } = payment.operation else {
            unreachable!("an exile alternative pays for a spell")
        };
        self.exile_alternative_objects(player, object, cast)
    }

    /// What `player` may choose to forage with for `payment`: a Food to
    /// sacrifice, or a graveyard card to exile while three can be. Once a
    /// card is exiled only cards are offered. A Food whose mana the rest of
    /// the cost needs is left out.
    pub(crate) fn forage_objects(&self, player: PlayerId, payment: &Payment) -> Vec<ObjectId> {
        let CostComponent::Forage { exiled, .. } = payment.owed[0] else {
            return Vec::new();
        };
        let mana = payment.mana().0;
        let mut objects = Vec::new();
        if exiled == 0 {
            objects.extend(
                self.sacrifice_candidates(player, &food())
                    .into_iter()
                    .filter(|id| self.can_pay_with_mana_abilities(player, mana, Some(*id)))
                    .filter_map(|id| self.permanent_object(id)),
            );
        }
        let Operation::Spell { object: spell, .. } = payment.operation else {
            unreachable!("forage pays for a spell")
        };
        let mut graveyard = self.graveyard_payers(player, spell);
        graveyard.sort_unstable_by_key(|object| (self.state[*object].definition_id, *object));
        if graveyard.len() >= usize::from(FORAGE_CARDS - exiled)
            && self.can_pay_with_mana_abilities(player, mana, None)
        {
            objects.extend(graveyard);
        }
        objects
    }

    /// The cards in `player`'s graveyard that may be exiled to pay for
    /// `spell` (delve, forage), in graveyard order. A spell cast from the
    /// graveyard waits there until it is paid for, but it is being cast and
    /// is no card its own cost can exile (CR 601.2a).
    pub(crate) fn graveyard_payers(&self, player: PlayerId, spell: ObjectId) -> Vec<ObjectId> {
        self.state
            .zones
            .zone_objects(ZoneType::Graveyard, player)
            .iter()
            .copied()
            .filter(|object| *object != spell)
            .collect()
    }

    /// The cards `player` may discard to pay a "discard a card" cost: the
    /// hand, less the spell being cast from it (`casting`).
    pub(crate) fn discard_candidates(
        &self,
        player: PlayerId,
        casting: Option<ObjectId>,
    ) -> Vec<ObjectId> {
        self.state
            .zone_cards(ZoneType::Hand, player)
            .map(|card| self.state.object_of(card))
            .filter(|object| Some(*object) != casting)
            .collect()
    }

    /// Whether `player` could forage for `spell` beside paying `cost` in mana.
    pub(super) fn can_forage(&self, player: PlayerId, spell: ObjectId, cost: &ManaCost) -> bool {
        self.graveyard_payers(player, spell).len() >= usize::from(FORAGE_CARDS)
            && self.can_pay_with_mana_abilities(player, cost, None)
            || self
                .sacrifice_candidates(player, &food())
                .into_iter()
                .any(|id| self.can_pay_with_mana_abilities(player, cost, Some(id)))
    }

    /// The creatures `player` controls that hold a counter, in battlefield
    /// order: what a counter-removal cost removes from.
    pub(crate) fn counter_removal_candidates(&self, player: PlayerId) -> Vec<ObjectId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| self.permanent_matches_predicate(*id, &CardPredicate::creature()))
            .filter_map(|id| {
                let permanent = self.state.permanents[id].as_ref()?;
                (!permanent.counters.is_empty()).then_some(permanent.object)
            })
            .collect()
    }

    /// The counters, of any kinds, on creatures `player` controls.
    pub(crate) fn removable_counters(&self, player: PlayerId) -> u32 {
        self.counter_removal_candidates(player)
            .into_iter()
            .flat_map(|object| self.counters_on(object))
            .map(|(_, count)| count.max(0) as u32)
            .sum()
    }

    /// The counters on `object`, in `CounterKind` order; none once it has
    /// left the battlefield.
    fn counters_on(&self, object: ObjectId) -> impl Iterator<Item = (CounterKind, i32)> + '_ {
        self.state
            .permanent_of(object)
            .and_then(|id| self.state.permanents[id].as_ref())
            .into_iter()
            .flat_map(|permanent| permanent.counters.iter())
    }

    /// Remove one counter of `kind` from `object` as part of a cost, then
    /// owe one fewer. Removing counters fires no trigger here.
    fn remove_cost_counter(
        &mut self,
        payment: &mut Payment,
        object: ObjectId,
        kind: CounterKind,
        remaining: u8,
    ) {
        self.commit_counter_change(object, kind, -1, None);
        if remaining > 1 {
            payment.owed[0] = CostComponent::RemoveCounters {
                remaining: remaining - 1,
                choosing_from: None,
            };
        } else {
            payment.owed.remove(0);
        }
    }

    /// Name the kind of counter to remove from the creature just chosen for
    /// a counter-removal cost; `mode` is the kind's `CounterKind` index.
    pub(crate) fn remove_counter_kind_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?.clone();
        let CostComponent::RemoveCounters {
            remaining,
            choosing_from: Some(object),
        } = payment.owed[0]
        else {
            return Err(AgentError("no counter kind is being chosen".into()));
        };
        let kind = self
            .removable_kind(object, mode)
            .ok_or_else(|| AgentError("that creature has no counter of this kind".into()))?;
        self.pending_choice = None;
        self.remove_cost_counter(&mut payment, object, kind, remaining);
        self.advance_payment(player, payment)
    }

    /// The kinds of counter on `object`, in `CounterKind` order.
    pub(crate) fn removable_kinds(&self, object: ObjectId) -> Vec<CounterKind> {
        self.counters_on(object).map(|(kind, _)| kind).collect()
    }

    /// The kind of counter on `object` whose `CounterKind` index is `mode`.
    pub(crate) fn removable_kind(&self, object: ObjectId, mode: usize) -> Option<CounterKind> {
        self.counters_on(object)
            .map(|(kind, _)| kind)
            .find(|kind| *kind as usize == mode)
    }

    /// The parts of `operation`'s total cost its payer chooses how to pay,
    /// in the order they are paid, ending with its mana. The total is fixed
    /// here, once announcement is over (CR 601.2f, 602.2b).
    pub(crate) fn cost_components(
        &self,
        player: PlayerId,
        operation: &Operation,
    ) -> Vec<CostComponent> {
        let mut owed = Vec::new();
        match operation {
            Operation::Spell {
                object,
                cast,
                targets,
                ..
            } => {
                if self.uses_exile_alternative(*object, *cast) {
                    owed.push(CostComponent::ExileFromHand);
                }
                let card = cast.spell(&self.state[*object]);
                if let Some(cost) = card.sacrifice_cost() {
                    owed.push(CostComponent::Sacrifice(cost.clone(), Toward::Printed));
                }
                if cast.kicked && card.bargain() {
                    owed.push(CostComponent::Sacrifice(bargain_cost(), Toward::Bargain));
                }
                if let Some(or_mana) = card.forage_or() {
                    owed.push(CostComponent::Forage {
                        or_mana: or_mana.clone(),
                        exiled: 0,
                    });
                }
                if card.discard_cost() {
                    owed.push(CostComponent::Discard);
                }
                if cast.route == Some(PlayRoute::PrintedGraveyard) {
                    if let Some(remaining) = card.graveyard_counter_cast() {
                        owed.push(CostComponent::RemoveCounters {
                            remaining,
                            choosing_from: None,
                        });
                    }
                }
                let cost = self
                    .effective_spell_cost(player, *object, *cast, Some(targets))
                    .unwrap_or_default();
                let help = self.spell_help(player, *object, *cast, &cost);
                owed.push(CostComponent::Mana { cost, help });
                if let (false, Some(life)) = (cast.kicked, card.life_alternative()) {
                    owed.push(CostComponent::Life(life));
                }
            }
            Operation::Activation {
                source,
                ability_index,
                x,
                ..
            } => {
                let ability = self
                    .activation_ability(*source, *ability_index)
                    .expect("announced activation source");
                if let Some((power, toward)) = ability.tap_power().filter(|(power, _)| *power > 0) {
                    owed.push(CostComponent::TapPower { power, toward });
                }
                if let Some((cost, toward)) = ability.chosen_tap() {
                    owed.push(CostComponent::Tap(cost, toward));
                }
                if ability.discard_card {
                    owed.push(CostComponent::Discard);
                }
                if let Some(cost) = ability.sacrifice {
                    owed.push(CostComponent::Sacrifice(cost, Toward::Printed));
                }
                let cost = if ability.mana_cost.x == 0 {
                    ability.mana_cost
                } else {
                    ability
                        .mana_cost
                        .with_x(u8::try_from(*x).expect("payable mana X"))
                };
                let help = Help {
                    waterbend: if ability.waterbend { cost.generic() } else { 0 },
                    ..Default::default()
                };
                owed.push(CostComponent::Mana { cost, help });
                if let Some(loyalty) = ability.loyalty {
                    let delta = loyalty.counter_change(*x);
                    if delta < 0 {
                        owed.push(CostComponent::RemoveSourceCounters(
                            CounterKind::Loyalty,
                            -delta,
                        ));
                    } else if delta > 0 {
                        owed.push(CostComponent::AddCounters(
                            CounterKind::Loyalty,
                            delta as u16,
                        ));
                    }
                }
                if let Some((kind, count)) = ability.remove_counters {
                    owed.push(CostComponent::RemoveSourceCounters(kind, i32::from(count)));
                }
                if let Some((kind, count)) = ability.add_counters {
                    owed.push(CostComponent::AddCounters(kind, count));
                }
                if ability.life_cost > 0 {
                    owed.push(CostComponent::Life(ability.life_cost));
                }
                if ability.exile_grantor {
                    owed.push(CostComponent::ExileGrantor(
                        ability.granted_by.expect("attached ability grant").grantor,
                    ));
                }
                if ability.tap_grantor {
                    owed.push(CostComponent::TapGrantor(
                        ability.granted_by.expect("attached ability grant").grantor,
                    ));
                }
                for (owes, how) in [
                    (ability.tap_source, How::Tapped),
                    (ability.sacrifice_source, How::Sacrificed),
                    (ability.exile_source, How::Exiled),
                    (ability.return_source, How::Returned),
                    (ability.discard_source, How::Discarded),
                ] {
                    if owes {
                        owed.push(CostComponent::Source(how));
                    }
                }
            }
        }
        owed
    }

    /// Announcement is over: fix the operation's cost and start paying it.
    pub(super) fn begin_payment(
        &mut self,
        player: PlayerId,
        operation: Operation,
    ) -> Result<(), AgentError> {
        if let Operation::Activation {
            source,
            ability_index,
            ..
        } = &operation
        {
            if self.activation_ability(*source, *ability_index).is_none() {
                return Err(AgentError("missing source".into()));
            }
        }
        let owed = self.cost_components(player, &operation);
        self.advance_payment(
            player,
            Payment {
                operation,
                owed,
                receipt: PaymentReceipt::default(),
            },
        )
    }

    /// Offer the next owed component. The payment is over once its mana
    /// step owes no mana, whatever else could still be tapped: surplus or
    /// unusable mana stays in the pool for the priority that follows. While
    /// mana is owed and no source, mana-ability or help choice remains, pay
    /// it from the pool and finish. A waterbend cost always offers its taps
    /// beside paying with mana, until a plain one (no convoke, improvise or
    /// delve) owes no more waterbend: then, unless the player floats extra
    /// mana by hand, it pays the rest when it can. A remaining colored cost
    /// that needs a mana ability, such as sacrificing Treasure, stays open.
    pub(crate) fn advance_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
    ) -> Result<(), AgentError> {
        let paying = payment.paying_mana();
        let paid = paying && payment.mana().0.mana_value == 0;
        let chooses = !paying || payment.waterbends();
        let waterbent = paying
            && payment.waterbends()
            && !payment.help().keyworded()
            && payment.mana().1 == 0
            && !self.offers_extra_payment_taps(player)
            && self.can_pay_mana_cost(player, payment.mana().0);
        self.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        let finishes = paid
            || waterbent
            || !chooses && {
                let _payments = self.payment_scan();
                !self.has_mana_payment_choices(player) && self.help_actions(player).is_empty()
            };
        if finishes {
            return self.finish_mana_payment(player);
        }
        Ok(())
    }

    /// The payment in progress and the component being paid.
    pub(crate) fn paying(&self) -> Option<(PlayerId, &Payment, &CostComponent)> {
        match &self.pending_choice {
            Some(PendingChoice::Pay { player, payment }) => {
                Some((*player, payment, &payment.owed[0]))
            }
            _ => None,
        }
    }

    /// The receipt of the cost being paid. While a decision is suspended it
    /// is the innermost payment's in that decision's caller chain: a spend
    /// choice carries its payment, with the units it has already spent
    /// appended as completion appends them, and a mana ability's own cost or
    /// output choice returns to the payment that activated it. Otherwise it
    /// is the announced payment's.
    pub(crate) fn pending_receipt(&self) -> Option<PaymentReceipt> {
        fn receipt_of(caller: ResolutionRef<'_>) -> Option<PaymentReceipt> {
            let ResolutionRef::Decision(decision) = caller else {
                return None;
            };
            match decision {
                Decision::Payment(PaymentProgress { receipt, .. }) => Some(receipt.clone()),
                Decision::ManaSpend {
                    completion, spent, ..
                } => match completion.as_ref() {
                    SpendCompletion::Announced(Payment { receipt, .. })
                    | SpendCompletion::Resolution(PaymentProgress { receipt, .. }) => {
                        let mut receipt = receipt.clone();
                        receipt.record_mana(spent.mana().iter().cloned());
                        Some(receipt)
                    }
                    SpendCompletion::Upfront(_) | SpendCompletion::Foretell(_) => None,
                },
                _ => None,
            }
        }
        self.state
            .suspended_decision
            .as_ref()
            .and_then(|suspended| {
                suspended
                    .continuation()
                    .into_iter()
                    .rev()
                    .find_map(receipt_of)
            })
            .or_else(|| match &self.pending_choice {
                Some(PendingChoice::Pay { payment, .. }) => Some(payment.receipt.clone()),
                _ => None,
            })
    }

    pub(crate) fn pending_payment(&self, player: PlayerId) -> Result<&Payment, AgentError> {
        match self.paying() {
            Some((payer, payment, _)) if payer == player => Ok(payment),
            Some(_) => Err(AgentError("wrong payer".into())),
            None => Err(AgentError("no cost payment is pending".into())),
        }
    }

    /// The untapped permanents `player` controls matching a tap cost.
    pub(crate) fn tap_candidates(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| {
                self.state.permanents[*id]
                    .as_ref()
                    .is_some_and(|permanent| !permanent.tapped)
                    && self.permanent_matches_predicate(*id, predicate)
            })
            .collect()
    }

    /// The untapped permanents `player` controls that can pay a `Tap`
    /// component of `operation`. Station taps another creature than its
    /// source, whatever its power.
    pub(crate) fn tap_component_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
        cost: &TapCost,
        toward: Toward,
    ) -> Vec<PermanentId> {
        let mut candidates = self.tap_candidates(player, &cost.predicate);
        if let (
            Toward::Station,
            Operation::Activation {
                source: ActivationSource::Permanent(source),
                ..
            },
        ) = (toward, operation)
        {
            candidates.retain(|id| id != source);
        }
        candidates
    }

    /// The untapped permanents `player` controls that can pay `ability`'s
    /// tap cost.
    pub(crate) fn tap_cost_candidates(
        &self,
        player: PlayerId,
        ability: &ActivatedAbilityDefinition,
    ) -> Vec<PermanentId> {
        ability
            .tap_cost
            .as_ref()
            .map(|cost| self.tap_candidates(player, &cost.predicate))
            .unwrap_or_default()
    }

    pub(crate) fn pay_discard_cost(
        &mut self,
        player: PlayerId,
        casting: Option<ObjectId>,
        object: ObjectId,
        receipt: &mut PaymentReceipt,
    ) -> Result<(), AgentError> {
        if !self.discard_candidates(player, casting).contains(&object) {
            return Err(AgentError("discard cost must be a card in hand".into()));
        }
        let discarded = self.move_object(object, ZoneType::Graveyard);
        receipt.contribute(discarded, How::Discarded, Toward::Printed);
        Ok(())
    }

    /// Pay the component being paid with `object`: tap it, discard it, exile
    /// it, or commit to sacrificing it. Taps, discards and exiles happen now;
    /// a sacrifice is recorded and happens as the whole cost completes, so no
    /// departure trigger sees a half-paid cost.
    pub(crate) fn pay_with_object(
        &mut self,
        player: PlayerId,
        object: ObjectId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?.clone();
        let permanent = self.state.permanent_of(object);
        match payment.owed[0].clone() {
            CostComponent::TapPower { power, toward } => {
                let Operation::Activation {
                    source: ActivationSource::Permanent(vehicle),
                    ..
                } = payment.operation
                else {
                    unreachable!("a permanent's ability taps creatures for power")
                };
                let creature = permanent
                    .filter(|id| self.crew_candidates(player, vehicle).contains(id))
                    .ok_or_else(|| AgentError("creature cannot crew".into()))?;
                let power = power - self.characteristics(creature).power;
                self.pending_choice = None;
                self.tap_permanent(creature, false);
                payment.receipt.contribute(object, How::Tapped, toward);
                if power > 0 {
                    payment.owed[0] = CostComponent::TapPower { power, toward };
                } else {
                    payment.owed.remove(0);
                }
            }
            CostComponent::Tap(cost, toward) => {
                let tapped = permanent
                    .filter(|id| {
                        self.tap_component_candidates(player, &payment.operation, &cost, toward)
                            .contains(id)
                    })
                    .ok_or_else(|| {
                        AgentError("tap cost must be an untapped permanent you control".into())
                    })?;
                if !self.tap_activation_payment_possible(player, &payment, Some(tapped)) {
                    return Err(AgentError(
                        "tap cost selection cannot complete the activation".into(),
                    ));
                }
                self.pending_choice = None;
                self.tap_permanent(tapped, false);
                payment.receipt.contribute(object, How::Tapped, toward);
                if cost.count > 1 {
                    payment.owed[0] = CostComponent::Tap(
                        TapCost {
                            count: cost.count - 1,
                            ..cost
                        },
                        toward,
                    );
                } else {
                    payment.owed.remove(0);
                }
            }
            CostComponent::Discard => {
                self.pay_discard_cost(
                    player,
                    payment.operation.spell(),
                    object,
                    &mut payment.receipt,
                )?;
                self.pending_choice = None;
                payment.owed.remove(0);
            }
            CostComponent::ExileFromHand => {
                if !self
                    .exile_payment_objects(player, &payment)
                    .contains(&object)
                {
                    return Err(AgentError(
                        "not a card this alternative cost can exile".into(),
                    ));
                }
                self.pending_choice = None;
                let exiled = self.move_object(object, ZoneType::Exile);
                payment
                    .receipt
                    .contribute(exiled, How::Exiled, Toward::Alternative);
                payment.owed.remove(0);
            }
            CostComponent::Sacrifice(cost, toward) => {
                if !permanent.is_some_and(|id| {
                    self.payment_candidates(player, &payment.operation, &cost)
                        .contains(&id)
                }) {
                    return Err(AgentError("illegal sacrifice cost".into()));
                }
                payment.receipt.contribute(object, How::Sacrificed, toward);
                payment.owed.remove(0);
            }
            CostComponent::Forage { or_mana, exiled } => {
                if !self.forage_objects(player, &payment).contains(&object) {
                    return Err(AgentError(
                        "not a Food or graveyard card to forage with".into(),
                    ));
                }
                if permanent.is_some() {
                    payment
                        .receipt
                        .contribute(object, How::Sacrificed, Toward::Forage);
                    payment.owed.remove(0);
                } else {
                    self.pending_choice = None;
                    let card = self.move_object(object, ZoneType::Exile);
                    payment
                        .receipt
                        .contribute(card, How::Exiled, Toward::Forage);
                    if exiled + 1 < FORAGE_CARDS {
                        payment.owed[0] = CostComponent::Forage {
                            or_mana,
                            exiled: exiled + 1,
                        };
                    } else {
                        payment.owed.remove(0);
                    }
                }
            }
            CostComponent::RemoveCounters {
                remaining,
                choosing_from: None,
            } => {
                if !self.counter_removal_candidates(player).contains(&object) {
                    return Err(AgentError(
                        "not a creature you control with a counter".into(),
                    ));
                }
                self.pending_choice = None;
                match self.removable_kinds(object)[..] {
                    [kind] => self.remove_cost_counter(&mut payment, object, kind, remaining),
                    _ => {
                        payment.owed[0] = CostComponent::RemoveCounters {
                            remaining,
                            choosing_from: Some(object),
                        }
                    }
                }
            }
            CostComponent::RemoveCounters { .. } => {
                return Err(AgentError("choose which counter to remove".into()));
            }
            CostComponent::Mana { cost, help } => {
                // Delve: a card exiled from the payer's graveyard pays {1}.
                if !(cost.generic() > 0
                    && help.delve.is_some_and(|spell| {
                        self.graveyard_payers(player, spell).contains(&object)
                    }))
                {
                    return Err(AgentError("no object pays this cost".into()));
                }
                self.pending_choice = None;
                let exiled = self.move_object(object, ZoneType::Exile);
                payment.help_paid(exiled, How::Exiled, Toward::Delve, Color::Generic);
            }
            CostComponent::TapPays { .. } => {
                return Err(AgentError("choose what the tapped permanent pays".into()));
            }
            CostComponent::Life(_)
            | CostComponent::RemoveSourceCounters(..)
            | CostComponent::AddCounters(..)
            | CostComponent::Source(_)
            | CostComponent::ExileGrantor(_)
            | CostComponent::TapGrantor(_) => {
                unreachable!("paid as the payment completes, after its mana")
            }
        }
        self.advance_payment(player, payment)
    }

    /// Mana `player` just produced while paying an announced cost's mana
    /// pays toward it at once (CR 601.2g-h): the pool's units from `start`
    /// on, in the order produced. Each unit is committed to the cost, not to
    /// a chosen pip: it takes the most flexible pip it can pay whose rest
    /// stays payable (`ManaCost::paid_with`), joins the receipt with its
    /// origin and shrinks what is owed, as a convoke tap does. A unit stays
    /// in the pool when its restriction forbids this cost, when no pip it
    /// pays leaves the rest payable, so the surplus of a source that makes
    /// more than is owed floats, or when pooled mana of another class or
    /// color could pay that pip instead. Mana floating before the
    /// production is untouched: the spend choice at "Pay the cost" decides
    /// it.
    pub(crate) fn commit_produced_mana(&mut self, player: PlayerId, start: usize) {
        let mut index = start;
        loop {
            let Some((payer, payment, CostComponent::Mana { cost, help })) = self.paying() else {
                return;
            };
            let waterbend = payment.mana().1;
            let pool = &self.state.players[player.0].mana_pool;
            if payer != player || index >= pool.unit_count() || cost.mana_value == 0 {
                return;
            }
            let unit = pool.unit(index);
            let help = *help;
            let rest = if unit.spendable_units(cost) == 0 {
                None
            } else {
                let mut left = self.pooled_mana(player);
                left.spend(&unit);
                cost.paid_with(pool.unit_color(index))
                    .into_iter()
                    .find(|rest| self.payable_with_help(player, &left, rest, help, waterbend, None))
                    // Pooled mana of another class or color could pay this
                    // pip instead, with other consequences (a rider applied
                    // or kept, a restriction used up, the colors spent and
                    // left floating): the spend choice at "Pay the cost"
                    // stays the payer's (CR 601.2h).
                    .filter(|rest| left.other_than(&unit).spendable_units(&cost.less(rest)) == 0)
            };
            let Some(rest) = rest else {
                index += 1;
                continue;
            };
            let purpose = cost.purpose.clone();
            let spent = self.state.mana_pool_mut(player).spend_at(index, &purpose);
            let pooled = self.state.players[player.0].mana_pool.total();
            if let Some(exposure) = self.card_capture.as_mut().and_then(|c| c.exposure.as_mut()) {
                exposure.mana[player.0].remove(1, pooled, false);
                exposure.paid(player, 1);
            }
            let Some(PendingChoice::Pay { payment, .. }) = &mut self.pending_choice else {
                unreachable!("an announced payment is being paid")
            };
            let (cost, _, receipt) = payment.mana_step();
            *cost = rest;
            receipt.record_mana([spent]);
        }
    }

    /// `PayCost` while a cost is being paid: take a sacrifice cost's mana
    /// branch, or pay the mana still owed.
    pub(crate) fn pay_cost_action(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let payment = self.pending_payment(player)?;
        let extra = match &payment.owed[0] {
            CostComponent::Sacrifice(cost, _) => cost
                .or_mana
                .as_ref()
                .ok_or_else(|| AgentError("sacrifice required".into()))?,
            CostComponent::Forage { or_mana, exiled: 0 } => or_mana,
            CostComponent::Mana { .. } => return self.finish_mana_payment(player),
            _ => return Err(AgentError("this cost is not paid with mana".into())),
        };
        let mut payment = payment.clone();
        payment.owed.remove(0);
        payment.owe_mana(extra);
        self.advance_payment(player, payment)
    }

    /// Pay the mana the pending payment still owes, then cast the spell or
    /// activate the ability.
    fn finish_mana_payment(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let payment = self.pending_payment(player)?;
        if !payment.paying_mana() {
            return Err(AgentError("no mana payment pending".into()));
        }
        let cost = payment.mana().0.clone();
        // The solver reads the pending payment, so leave it installed until
        // production succeeds.
        self.produce_mana(player, &cost)?;
        // Commit the complete announced cost once, before sacrifice triggers
        // can observe a departing permanent or a spell reaches the stack.
        let Some(PendingChoice::Pay { payment, .. }) = self.pending_choice.take() else {
            unreachable!("automatic mana production does not replace the pending payment")
        };
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Announced(*payment),
            EffectFrame::new(player, Vec::new()),
        )
    }

    /// The whole announced cost is paid: pay what `owed` holds after the
    /// mana (life and the source's own costs), sacrifice what was chosen or
    /// is owed, then cast the spell or put the ability on the stack with the
    /// receipt. The ability is captured before its source taps or leaves.
    pub(crate) fn complete_announced_payment(
        &mut self,
        player: PlayerId,
        mut payment: Payment,
    ) -> Result<(), AgentError> {
        if payment.waterbends() {
            self.record_bending(player, crate::flow::turn::Bending::Water);
        }
        let life = payment.life();
        self.pay_life_cost(player, life, &mut payment.receipt);
        // The source's own discard or exile; its sacrifice is on the receipt.
        let mut source_costs = Vec::new();
        let activated = match &payment.operation {
            Operation::Spell { .. } => None,
            Operation::Activation {
                source,
                ability_index,
                targets,
                target_req_indices,
                x,
            } => {
                let mut ability = match *source {
                    ActivationSource::Permanent(permanent) => self.prepare_activated_ability(
                        player,
                        permanent,
                        *ability_index,
                        targets.clone(),
                        *x,
                        payment.owes(&CostComponent::Source(How::Tapped)),
                    )?,
                    ActivationSource::Card(card) => self.prepare_card_activated_ability(
                        player,
                        card,
                        *ability_index,
                        targets.clone(),
                    )?,
                };
                ability.target_req_indices = target_req_indices.clone();
                let source_object = ability.source_object;
                let counter_cost = |kind, delta| ProposedEvent::CounterChange {
                    provenance: CounterProvenance::new(EventCause::Cost),
                    target: source_object,
                    kind,
                    delta,
                    by: Some(player),
                };
                for component in &payment.owed {
                    match *component {
                        CostComponent::RemoveSourceCounters(kind, count) => {
                            let available = self
                                .state
                                .permanent_of(source_object)
                                .and_then(|id| self.state.permanents[id].as_ref())
                                .map_or(0, |permanent| permanent.counter_count(kind));
                            if available < count {
                                return Err(AgentError("cannot pay counter cost".into()));
                            }
                            source_costs.push(counter_cost(kind, -count));
                        }
                        CostComponent::AddCounters(kind, count) => {
                            source_costs.push(counter_cost(kind, i32::from(count)));
                        }
                        CostComponent::TapGrantor(object) => {
                            let grantor = self
                                .state
                                .permanent_of(object)
                                .ok_or_else(|| AgentError("the grantor has left".into()))?;
                            if !self.tap_permanent(grantor, false) {
                                return Err(AgentError("cannot tap the grantor".into()));
                            }
                        }
                        CostComponent::Source(how) => {
                            payment
                                .receipt
                                .contribute(source_object, how, Toward::Printed);
                            let to = match how {
                                How::Tapped | How::Sacrificed => continue,
                                How::Discarded => ZoneType::Graveyard,
                                How::Exiled => ZoneType::Exile,
                                How::Returned => ZoneType::Hand,
                            };
                            source_costs.push(self.zone_move_proposal(
                                source_object,
                                to,
                                Default::default(),
                            ));
                        }
                        CostComponent::ExileGrantor(object) => {
                            payment
                                .receipt
                                .contribute(object, How::Exiled, Toward::Printed);
                            source_costs.push(self.zone_move_proposal(
                                object,
                                ZoneType::Exile,
                                Default::default(),
                            ));
                        }
                        _ => {}
                    }
                }
                Some((ability, *source))
            }
        };
        let events = payment
            .receipt
            .paid_as(How::Sacrificed)
            .filter(|object| self.state.permanent_of(*object).is_some())
            .map(|target| ProposedEvent::Sacrifice { target })
            .chain(source_costs);
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.batch = Some(self.event_batch(EventBinding::None, events));
        frame.completion = match activated {
            None => FrameCompletion::Paid { player, payment },
            Some((mut ability, source)) => {
                ability.paid = payment.receipt;
                FrameCompletion::Activated {
                    ability,
                    source_after_cost: match source {
                        ActivationSource::Card(card) => Some(self.state.object_of(card)),
                        ActivationSource::Permanent(_) => None,
                    },
                }
            }
        };
        self.run_frame(frame);
        Ok(())
    }

    /// Other untapped creatures `player` controls that can help crew
    /// `vehicle`. Summoning sickness doesn't matter (crew is not a {T}
    /// cost); creatures without positive power add nothing and are omitted.
    pub(crate) fn crew_candidates(
        &self,
        player: PlayerId,
        vehicle: PermanentId,
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|&id| {
                id != vehicle
                    && self.state.permanents[id]
                        .as_ref()
                        .is_some_and(|permanent| !permanent.tapped)
                    && {
                        let characteristics = self.characteristics(id);
                        characteristics.is_creature() && characteristics.power > 0
                    }
            })
            .collect()
    }

    pub(crate) fn crew_power_available(&self, player: PlayerId, vehicle: PermanentId) -> i32 {
        self.crew_candidates(player, vehicle)
            .into_iter()
            .map(|id| self.characteristics(id).power)
            .sum()
    }

    /// Untapped artifacts/creatures `player` controls, as the layers see
    /// them — the permanents that can each pay {1} of a waterbend cost.
    pub(crate) fn waterbend_candidates(&self, player: PlayerId) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| {
                self.state.permanents[*id]
                    .as_ref()
                    .is_some_and(|permanent| !permanent.tapped)
                    && {
                        let view = self.characteristics(*id);
                        view.types.is_artifact() || view.is_creature()
                    }
            })
            .collect()
    }

    /// The card whose spell or ability `payment` pays for.
    pub(crate) fn payment_source_object(&self, operation: &Operation) -> Option<ObjectId> {
        match operation {
            Operation::Spell { object, .. } => Some(*object),
            Operation::Activation {
                source: ActivationSource::Card(card),
                ..
            } => Some(self.state.object_of(*card)),
            Operation::Activation {
                source: ActivationSource::Permanent(permanent),
                ..
            } => self.state.permanents[*permanent]
                .as_ref()
                .map(|permanent| permanent.object),
        }
    }

    /// The permanent whose ability `payment` activates when that ability
    /// also costs {T}: it can't tap for both.
    pub(crate) fn payment_tapped_source(&self, operation: &Operation) -> Option<PermanentId> {
        let Operation::Activation {
            source: ActivationSource::Permanent(source),
            ability_index,
            ..
        } = operation
        else {
            return None;
        };
        self.characteristics(*source)
            .activated_abilities
            .get(*ability_index)
            .is_some_and(|ability| ability.tap_source)
            .then_some(*source)
    }

    /// The untapped artifacts and creatures a tap may pay `operation` with
    /// (waterbend, convoke or improvise), less a source it taps anyway.
    pub(crate) fn help_tap_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
    ) -> Vec<PermanentId> {
        let mut candidates = self.waterbend_candidates(player);
        if let Some(source) = self.payment_tapped_source(operation) {
            candidates.retain(|candidate| *candidate != source);
        }
        candidates
    }

    /// Pay `life` as a cost (CR 119.4) and record it on `receipt`. The
    /// caller has offered it only to a player who can pay it.
    pub(crate) fn pay_life_cost(
        &mut self,
        player: PlayerId,
        life: u32,
        receipt: &mut PaymentReceipt,
    ) {
        self.change_life(player, -(life as i32));
        receipt.pay_life(life);
    }

    /// What tapping `tapped` may pay of the pending cost; see `help_taps`.
    /// Empty unless `tapped` is a tap the payment offers.
    pub(crate) fn help_tap_options(
        &self,
        player: PlayerId,
        payment: &Payment,
        tapped: PermanentId,
    ) -> Vec<TapPayment> {
        if !self
            .help_tap_candidates(player, &payment.operation)
            .contains(&tapped)
        {
            return Vec::new();
        }
        let (cost, waterbend) = payment.mana();
        let also_tapped = self.payment_tapped_source(&payment.operation);
        self.help_taps(
            player,
            cost,
            payment.help(),
            waterbend,
            &[tapped],
            also_tapped,
        )
        .pop()
        .map_or_else(Vec::new, |(_, options)| options)
    }

    /// The taps and exiles that may pay part of the pending mana cost by
    /// convoke, improvise, waterbend or delve, each leaving the rest payable.
    pub(crate) fn help_actions(&self, player: PlayerId) -> Vec<Action> {
        let Some((payer, payment, CostComponent::Mana { cost, help })) = self.paying() else {
            return Vec::new();
        };
        if payer != player {
            return Vec::new();
        }
        let (_, waterbend) = payment.mana();
        let mut actions: Vec<Action> = self
            .help_taps(
                player,
                cost,
                *help,
                waterbend,
                &self.help_tap_candidates(player, &payment.operation),
                self.payment_tapped_source(&payment.operation),
            )
            .into_iter()
            .map(|(permanent, _)| Action::WaterbendTap { player, permanent })
            .collect();
        if let Some(spell) = help.delve.filter(|_| cost.generic() > 0) {
            actions.extend(
                self.graveyard_payers(player, spell)
                    .into_iter()
                    .map(|object| Action::SelectCard { player, object }),
            );
        }
        actions
    }

    /// Tap `tapped` to pay part of the pending cost: {1} or a pip of its
    /// color by convoke, {1} by improvise or {1} of a waterbend cost (CR
    /// 702.51a, 702.126a, 701.67). Tapping pays without activating a mana
    /// ability. A permanent that could pay more than one thing is tapped now
    /// and what it pays chosen next.
    pub(crate) fn help_tap_action(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?.clone();
        if !payment.paying_mana() {
            return Err(AgentError("no mana payment is pending".to_string()));
        }
        let options = self.help_tap_options(player, &payment, tapped);
        let object = self
            .permanent_object(tapped)
            .filter(|_| !options.is_empty())
            .ok_or_else(|| AgentError("permanent cannot help pay this cost".to_string()))?;
        self.pending_choice = None;
        self.tap_permanent(tapped, false);
        match options[..] {
            [paid] => payment.help_paid(object, How::Tapped, paid.toward, paid.color),
            _ => {
                payment.owed.insert(
                    0,
                    CostComponent::TapPays {
                        tapped: object,
                        options,
                    },
                );
                return self.advance_payment(player, payment);
            }
        }
        self.advance_payment(player, payment)
    }

    /// Choose what the permanent just tapped pays, by its mode
    /// (`TapPayment::mode`); a mode not among its options is rejected.
    pub(crate) fn tap_pays_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?.clone();
        let CostComponent::TapPays { tapped, options } = payment.owed[0].clone() else {
            return Err(AgentError("no tap payment is being chosen".into()));
        };
        let paid = *options
            .iter()
            .find(|paid| paid.mode() == mode)
            .ok_or_else(|| AgentError("not something this tap can pay".into()))?;
        payment.owed.remove(0);
        payment.help_paid(tapped, How::Tapped, paid.toward, paid.color);
        self.advance_payment(player, payment)
    }

    /// Tap `tapped` to pay {1} of a payment made during resolution: only
    /// waterbend helps one (`CostComponent::resolving`), so a tap pays one
    /// thing. The receipt names the permanent tapped.
    pub(crate) fn resolution_help_tap(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
        cost: &mut ManaCost,
        help: Help,
        receipt: &mut PaymentReceipt,
    ) -> Result<(), AgentError> {
        let waterbend = help.waterbend_left(cost, receipt);
        let paid = self
            .waterbend_candidates(player)
            .contains(&tapped)
            .then(|| self.help_taps(player, cost, help, waterbend, &[tapped], None))
            .and_then(|mut found| found.pop())
            .and_then(|(_, options)| options.first().copied());
        let (Some(paid), Some(object)) = (paid, self.permanent_object(tapped)) else {
            return Err(AgentError(
                "permanent cannot be tapped for waterbend".to_string(),
            ));
        };
        self.tap_permanent(tapped, false);
        pay_pip_with(cost, receipt, object, How::Tapped, paid.toward, paid.color);
        Ok(())
    }
}

/// The cards forage exiles from a graveyard (CR 701.61a).
const FORAGE_CARDS: u8 = 3;

fn food() -> CardPredicate {
    CardPredicate {
        subtype: Some("Food".into()),
        ..Default::default()
    }
}

/// What bargain sacrifices: an artifact, enchantment or token (CR 702.166a).
pub(super) fn bargain_cost() -> SacrificeCost {
    SacrificeCost {
        predicate: CardPredicate {
            any_of: vec![
                CardPredicate {
                    card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                    ..Default::default()
                },
                CardPredicate {
                    token: Some(true),
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        another: false,
        or_mana: None,
    }
}
