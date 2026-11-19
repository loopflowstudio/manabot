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
        decision::{Decision, EffectFrame, FrameCompletion, ResolutionRef, SpendCompletion},
        game::{ActivationSource, CastChoices, Game, PendingChoice},
        proposed_event::ProposedEvent,
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{Effect, PaymentCost},
        card::{ActivatedAbilityDefinition, CardType, SacrificeCost, TapCost},
        game_object::{ObjectId, PermanentId, PlayerId, Target},
        mana::{Color, ManaCost},
        payment::{How, PaymentReceipt, Toward},
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
        /// The announced value of X (CR 602.2b, 601.2b).
        #[serde(skip_serializing_if = "crate::state::is_zero")]
        x: u8,
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
    TapPower { power: i32, toward: Toward },
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
    Forage { or_mana: ManaCost, exiled: u8 },
    /// The mana still owed, and how its payer may pay part of it without
    /// mana. Each such payment is a contribution and reduces `cost`.
    Mana { cost: ManaCost, help: Help },
    /// The convoking creature just tapped has several colors the cost still
    /// owes: choose which one it pays (CR 702.51a).
    ConvokeColor {
        creature: ObjectId,
        colors: Vec<Color>,
    },
    /// "Pay N life" (CR 119.4): an ability's life cost, or a spell's life
    /// paid rather than its other cost.
    Life(u32),
    /// What the source itself pays: its own {T} (`Tapped`), "sacrifice this"
    /// (`Sacrificed`), "exile this" (`Exiled`) or, for an ability that
    /// functions from hand, "discard this card" (`Discarded`).
    Source(How),
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

    /// Whether taps pay for any of the generic mana rather than only the
    /// announced waterbend part.
    pub fn taps_any_generic(self) -> bool {
        self.convoke || self.improvise
    }

    /// The keyword under which tapping a permanent pays part of the cost:
    /// convoke for a creature, improvise for an artifact.
    pub fn tap_toward(self, creature: bool, artifact: bool) -> Option<Toward> {
        if self.convoke && creature {
            Some(Toward::Convoke)
        } else if self.improvise && artifact {
            Some(Toward::Improvise)
        } else {
            None
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
    /// `Life` (`CostComponent::resolving`).
    pub owed: CostComponent,
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub receipt: PaymentReceipt,
    pub if_paid: Vec<Effect>,
    pub if_declined: Vec<Effect>,
    /// The once-each-turn option (exact source, ability index) paying
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
                let card = &self.state[*object];
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
                let cost = self
                    .effective_spell_cost(player, *object, *cast, Some(targets))
                    .unwrap_or_default();
                let help = self.spell_help(player, *object, *cast, &cost);
                owed.push(CostComponent::Mana { cost, help });
                if let (false, Some(life)) = (cast.kicked, self.state[*object].life_alternative()) {
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
                let cost = ability.mana_cost.with_x(*x);
                let help = Help {
                    waterbend: if ability.waterbend { cost.generic() } else { 0 },
                    ..Default::default()
                };
                owed.push(CostComponent::Mana { cost, help });
                if ability.life_cost > 0 {
                    owed.push(CostComponent::Life(ability.life_cost));
                }
                for (owes, how) in [
                    (ability.tap_source, How::Tapped),
                    (ability.sacrifice_source, How::Sacrificed),
                    (ability.exile_source, How::Exiled),
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

    /// Offer the next owed component. Once only mana is owed and no source
    /// or mana-ability choice remains, pay it and finish.
    fn advance_payment(&mut self, player: PlayerId, payment: Payment) -> Result<(), AgentError> {
        // A waterbend cost always offers its taps beside paying with mana.
        let chooses = !payment.paying_mana() || payment.waterbends();
        self.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        let offers = chooses || {
            let _payments = self.payment_scan();
            self.has_mana_payment_choices(player) || !self.help_actions(player).is_empty()
        };
        if offers {
            return Ok(());
        }
        self.finish_mana_payment(player)
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
    /// choice carries its payment, and a mana ability's own cost or output
    /// choice returns to the payment that activated it. Otherwise it is the
    /// announced payment's.
    pub(crate) fn pending_receipt(&self) -> Option<&PaymentReceipt> {
        fn receipt_of(caller: ResolutionRef<'_>) -> Option<&PaymentReceipt> {
            let ResolutionRef::Decision(decision) = caller else {
                return None;
            };
            match decision {
                Decision::Payment(PaymentProgress { receipt, .. }) => Some(receipt),
                Decision::ManaSpend { completion, .. } => match completion.as_ref() {
                    SpendCompletion::Announced(Payment { receipt, .. })
                    | SpendCompletion::Resolution(PaymentProgress { receipt, .. }) => Some(receipt),
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
            .or(match &self.pending_choice {
                Some(PendingChoice::Pay { payment, .. }) => Some(&payment.receipt),
                _ => None,
            })
    }

    fn pending_payment(&self, player: PlayerId) -> Result<Payment, AgentError> {
        match self.paying() {
            Some((payer, payment, _)) if payer == player => Ok(payment.clone()),
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

    /// Pay the component being paid with `object`: tap it, discard it, exile
    /// it, or commit to sacrificing it. Taps, discards and exiles happen now;
    /// a sacrifice is recorded and happens as the whole cost completes, so no
    /// departure trigger sees a half-paid cost.
    pub(crate) fn pay_with_object(
        &mut self,
        player: PlayerId,
        object: ObjectId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
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
                if !self
                    .discard_candidates(player, payment.operation.spell())
                    .contains(&object)
                {
                    return Err(AgentError("discard cost must be a card in hand".into()));
                }
                self.pending_choice = None;
                let discarded = self.move_object(object, ZoneType::Graveyard);
                payment
                    .receipt
                    .contribute(discarded, How::Discarded, Toward::Printed);
                payment.owed.remove(0);
            }
            CostComponent::ExileFromHand => {
                let Operation::Spell { object: spell, .. } = payment.operation else {
                    unreachable!("an exile alternative pays for a spell")
                };
                if !self
                    .state
                    .card_of(object)
                    .is_some_and(|card| self.exile_alternative_cards(player, spell).contains(&card))
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
            CostComponent::ConvokeColor { .. } => {
                return Err(AgentError("choose the color this creature pays".into()));
            }
            CostComponent::Life(_) | CostComponent::Source(_) => {
                unreachable!("paid as the payment completes, after its mana")
            }
        }
        self.advance_payment(player, payment)
    }

    /// `PayCost` while a cost is being paid: take a sacrifice cost's mana
    /// branch, or pay the mana still owed.
    pub(crate) fn pay_cost_action(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        let extra = match payment.owed[0].clone() {
            CostComponent::Sacrifice(cost, _) => cost
                .or_mana
                .ok_or_else(|| AgentError("sacrifice required".into()))?,
            CostComponent::Forage { or_mana, exiled: 0 } => or_mana,
            CostComponent::Mana { .. } => return self.finish_mana_payment(player),
            _ => return Err(AgentError("this cost is not paid with mana".into())),
        };
        payment.owed.remove(0);
        payment.owe_mana(&extra);
        self.advance_payment(player, payment)
    }

    /// Pay the mana the pending payment still owes, then cast the spell or
    /// activate the ability.
    pub(crate) fn finish_mana_payment(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let payment = self.pending_payment(player)?;
        if !payment.paying_mana() {
            return Err(AgentError("no mana payment pending".into()));
        }
        let cost = payment.mana().0.clone();
        self.produce_mana(player, &cost)?;
        // Commit the complete announced cost once, before sacrifice triggers
        // can observe a departing permanent or a spell reaches the stack.
        self.pending_choice = None;
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Announced(payment),
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
        let mut leaves = Vec::new();
        let activated = match &payment.operation {
            Operation::Spell { .. } => None,
            Operation::Activation {
                source,
                ability_index,
                targets,
                x,
            } => {
                let ability = match *source {
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
                let source_object = ability.source_object;
                for component in &payment.owed {
                    let CostComponent::Source(how) = *component else {
                        continue;
                    };
                    payment
                        .receipt
                        .contribute(source_object, how, Toward::Printed);
                    let to = match how {
                        How::Tapped | How::Sacrificed => continue,
                        How::Discarded => ZoneType::Graveyard,
                        How::Exiled => ZoneType::Exile,
                    };
                    leaves.push(self.zone_move_proposal(source_object, to, Default::default()));
                }
                Some((ability, *source))
            }
        };
        let events: Vec<_> = payment
            .receipt
            .sacrificed()
            .filter(|object| self.state.permanent_of(*object).is_some())
            .map(|target| ProposedEvent::Sacrifice { target })
            .chain(leaves)
            .collect();
        let completion = match activated {
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
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.batch = Some(self.event_batch(events));
        frame.completion = completion;
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

    /// Can `source`'s waterbend ability be paid, its own {T} included?
    pub(crate) fn can_pay_waterbend_activation(
        &self,
        player: PlayerId,
        source: PermanentId,
        ability: &ActivatedAbilityDefinition,
    ) -> bool {
        // An X cost is payable when its least X is.
        let cost = ability.mana_cost.with_x(ability.x_min);
        if !ability.tap_source {
            return self.can_pay_with_waterbend(player, &cost);
        }
        !self
            .waterbend_taps_keeping_payable(player, &cost, cost.generic(), vec![source], None)
            .is_empty()
    }

    /// Can `player` cover `cost` with untapped waterbend candidates
    /// ({1} each, generic only) plus available mana?
    pub(crate) fn can_pay_with_waterbend(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.waterbend_payment_possible(player, cost, cost.generic())
    }

    /// Tap `tapped`, one of `candidates`, to pay {1} of the waterbend part
    /// of `cost` (CR 701.67): the one waterbend tap, for a cast, an
    /// activation and a payment made during resolution. It pays without
    /// activating a mana ability; `receipt` names the permanent tapped.
    pub(crate) fn waterbend_tap(
        &mut self,
        tapped: PermanentId,
        candidates: &[PermanentId],
        cost: &mut ManaCost,
        help: Help,
        receipt: &mut PaymentReceipt,
    ) -> Result<(), AgentError> {
        if help.waterbend_left(cost, receipt) == 0 {
            return Err(AgentError("waterbend cost is already paid".to_string()));
        }
        let object = self
            .permanent_object(tapped)
            .filter(|_| candidates.contains(&tapped))
            .ok_or_else(|| AgentError("permanent cannot be tapped for waterbend".to_string()))?;
        self.tap_permanent(tapped, false);
        pay_pip_with(
            cost,
            receipt,
            object,
            How::Tapped,
            Toward::Waterbend,
            Color::Generic,
        );
        Ok(())
    }

    /// The waterbend taps among `candidates` that leave the rest of `cost`
    /// payable, `waterbend` of its generic mana being payable by taps.
    /// `also_tapped` is the source a cost taps as well.
    pub(crate) fn waterbend_tap_offers(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
        candidates: Vec<PermanentId>,
        also_tapped: Option<PermanentId>,
    ) -> Vec<Action> {
        if waterbend == 0 {
            return Vec::new();
        }
        self.waterbend_taps_keeping_payable(
            player,
            &cost.reduced_generic(1),
            waterbend - 1,
            candidates,
            also_tapped,
        )
        .into_iter()
        .map(|permanent| Action::WaterbendTap { player, permanent })
        .collect()
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

    /// Tap `tapped` to pay {1} of the pending waterbend cost.
    pub(crate) fn waterbend_tap_action(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        if payment.paying_mana() && payment.help().taps_any_generic() {
            return self.help_tap_action(player, tapped, payment);
        }
        if !payment.paying_mana() || !payment.waterbends() {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        }
        let candidates = self.help_tap_candidates(player, &payment.operation);
        let (cost, help, receipt) = payment.mana_step();
        self.waterbend_tap(tapped, &candidates, cost, help, receipt)?;
        let (owed, waterbend) = payment.mana();
        // A remaining colored cost can still need a mana ability such as
        // sacrificing Treasure. Leave the committed payment open for it.
        let finished = waterbend == 0
            && (owed.mana_value == 0
                || (!self.offers_extra_payment_taps(player)
                    && self.can_pay_mana_cost(player, owed)));
        self.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        if finished {
            return self.finish_mana_payment(player);
        }
        Ok(())
    }

    /// The keyword under which tapping `tapped` pays part of a cost with
    /// `help`: convoke for a creature, improvise for an artifact.
    pub(crate) fn help_tap_toward(&self, help: Help, tapped: PermanentId) -> Option<Toward> {
        let view = self.characteristics(tapped);
        help.tap_toward(view.is_creature(), view.types.is_artifact())
    }

    /// The pips `tapped` may pay of the pending cost by convoke or
    /// improvise; see `help_taps`. `tapped` is a tap the payment offers.
    pub(crate) fn help_tap_options(
        &self,
        player: PlayerId,
        payment: &Payment,
        tapped: PermanentId,
    ) -> Vec<Color> {
        self.help_taps(player, payment.mana().0, payment.help(), &[tapped])
            .pop()
            .map_or_else(Vec::new, |(_, colors)| colors)
    }

    /// The taps and exiles that may pay part of the pending mana cost by
    /// convoke, improvise or delve, each leaving the rest payable.
    pub(crate) fn help_actions(&self, player: PlayerId) -> Vec<Action> {
        let Some((payer, payment, CostComponent::Mana { cost, help })) = self.paying() else {
            return Vec::new();
        };
        if payer != player || !help.keyworded() {
            return Vec::new();
        }
        let mut actions = Vec::new();
        if help.taps_any_generic() {
            let candidates = self.help_tap_candidates(player, &payment.operation);
            actions.extend(
                self.help_taps(player, cost, *help, &candidates)
                    .into_iter()
                    .map(|(permanent, _)| Action::WaterbendTap { player, permanent }),
            );
        }
        if let Some(spell) = help.delve.filter(|_| cost.generic() > 0) {
            actions.extend(
                self.graveyard_payers(player, spell)
                    .into_iter()
                    .map(|object| Action::SelectCard { player, object }),
            );
        }
        actions
    }

    /// Tap `tapped` to pay one pip of the pending cost by convoke or
    /// improvise. A creature that could pay more than one owed color is
    /// tapped now and its color chosen next.
    fn help_tap_action(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
        mut payment: Payment,
    ) -> Result<(), AgentError> {
        let colors = if self
            .help_tap_candidates(player, &payment.operation)
            .contains(&tapped)
        {
            self.help_tap_options(player, &payment, tapped)
        } else {
            Vec::new()
        };
        let object = self
            .permanent_object(tapped)
            .filter(|_| !colors.is_empty())
            .ok_or_else(|| AgentError("permanent cannot help pay this cost".to_string()))?;
        let toward = self
            .help_tap_toward(payment.help(), tapped)
            .expect("a tap with options pays by a keyword");
        self.pending_choice = None;
        // Convoke and improvise pay a cost without activating a mana ability.
        self.tap_permanent(tapped, false);
        match colors[..] {
            [color] => payment.help_paid(object, How::Tapped, toward, color),
            _ => payment.owed.insert(
                0,
                CostComponent::ConvokeColor {
                    creature: object,
                    colors,
                },
            ),
        }
        self.advance_payment(player, payment)
    }

    /// Choose which owed color the convoking creature just tapped pays.
    pub(crate) fn convoke_color_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        let CostComponent::ConvokeColor { creature, colors } = payment.owed[0].clone() else {
            return Err(AgentError("no convoke color is being chosen".into()));
        };
        let color = *colors
            .get(mode)
            .ok_or_else(|| AgentError("not a color this creature can pay".into()))?;
        payment.owed.remove(0);
        payment.help_paid(creature, How::Tapped, Toward::Convoke, color);
        self.advance_payment(player, payment)
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
