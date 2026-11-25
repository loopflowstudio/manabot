// decision.rs
// Mid-resolution decisions: a resolving spell or ability can suspend, surface
// a choice to a specific player as an ActionSpace, and resume when the choice
// action arrives. The suspended state (remaining effects + the pending
// decision) lives on GameState so clones/forks carry it.

use crate::state::card::SpellProfile;
#[cfg(test)]
use crate::state::player::ManaFloats;
use std::collections::{HashSet, VecDeque};

use rand::seq::SliceRandom;

use crate::cards::names::NameOption;
use crate::state::binding::{ObjectBinding, ZoneSelection};

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        event::GameEvent,
        game::Game,
        payment::{CostComponent, Payment, PaymentProgress},
        play::CastAuthority,
        proposed_event::{
            BatchProgress, EventBatch, EventBinding, EventResult, ReplacementCandidate, ZoneEntry,
        },
        sba::SbaProgress,
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{DistinctSearchEnd, Effect, LibraryEnd, LinkedGrant},
        game_object::{CardId, ObjectId, PermanentId, PlayerId, PublicId, Target},
        knowledge::LibraryKnowledge,
        mana::ManaCost,
        payment::PaymentReceipt,
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// The caller awaiting a frame's completion. A child owns exactly one return
/// destination; unfinished effects never grant priority or expose a saved
/// decision. Spells leave the stack as the final part of resolution (CR 608.2m).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum FrameCompletion {
    /// A completed root ability restarts the priority round.
    Priority,
    /// A turn action or rule batch returns to the existing tick/priority state.
    Continue,
    StateBased {
        next: Box<FrameCompletion>,
    },
    /// After stack exile, finish state-based actions before starting cleanup.
    EndTurn,
    /// The spell's whole cost is paid; `payment.receipt` is what it actually
    /// took (CR 601.2h).
    Paid {
        player: PlayerId,
        payment: Payment,
    },
    Activated {
        ability: crate::state::stack_object::ActivatedAbilityOnStack,
        source_after_cost: Option<ObjectId>,
    },
    /// An iteration child returns to its interrupted scope.
    Frame(Box<EffectFrame>),
    /// Announce a completed rules operation before resuming its caller.
    /// Keyword events happen after every selection and ordering choice.
    Event {
        event: GameEvent,
        frame: Box<EffectFrame>,
    },
    /// Cost effects finish before the saved caller decision becomes available.
    /// Its frame, output and parent payment remain owned here across choices.
    Decision(Box<SuspendedResolution>),
    /// Spells: remove the stack object, then put the card into the
    /// battlefield (permanents) or graveyard, through the ordinary entry
    /// batch and its replacement choices, then announce the resolution. A
    /// resolving Aura spell enters attached to `attach_to`, the object its
    /// attach instruction chose (CR 303.4f).
    Spell {
        #[serde(rename = "card")]
        object: ObjectId,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(serialize_with = "crate::state::game_object::exact")]
        attach_to: Option<ObjectId>,
    },
}

impl FrameCompletion {
    fn then(&mut self, completion: FrameCompletion) {
        match self {
            Self::Continue => *self = completion,
            Self::StateBased { next } => next.then(completion),
            Self::Frame(frame) | Self::Event { frame, .. } => frame.completion.then(completion),
            Self::Decision(caller) => caller.frame.completion.then(completion),
            _ => panic!("rule continuation must return to its caller"),
        }
    }
}

/// A chosen target's clause and eligibility at resolution start (CR 608.2b).
/// Keep illegal slots: positional instructions must not reinterpret another
/// chosen object as the final target. Presence is checked separately on use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ResolutionTarget {
    pub target: Target,
    pub requirement: usize,
    pub legal: bool,
}

/// Everything a resolving spell/ability needs while its effects execute,
/// including what's left to do if it suspends for a decision.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct EffectFrame {
    /// Original dynamically granted program, retained across branch selection.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub program: Vec<Effect>,
    /// The current instruction owns this batch through replacement and child choices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) batch: Option<EventBatch>,
    /// An iteration owns its remaining subjects while one child scope runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration: Option<TargetIteration>,
    /// Exact subjects captured at delayed-trigger creation, independent of event context.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub remembered: Vec<ObjectBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<crate::state::card::CardDefId>,
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub source_transforms: u32,
    /// The single exact source, inherited unchanged by child scopes.
    #[serde(
        flatten,
        serialize_with = "crate::state::game_object::serialize_frame_source"
    )]
    pub source_identity: Option<crate::state::game_object::AbilitySource>,
    /// The player the effects resolve for ("you").
    pub controller: PlayerId,
    /// For triggered abilities: how many times this ability has resolved
    /// this turn, counting this resolution. Zero for spells and activated
    /// abilities.
    pub resolutions_this_turn: u32,
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub triggering_amount: i32,
    /// Whether the resolving spell was kicked.
    pub kicked: bool,
    /// The resolving spell's or ability's announced X; 0 without one.
    pub x: u8,
    /// What was paid to cast the resolving spell, to cast the permanent whose
    /// entering triggered the resolving ability, or to activate the
    /// resolving ability; empty otherwise.
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub spent: PaymentReceipt,
    /// Excess damage (CR 120.4a) the latest damage instruction dealt to its
    /// creature, read by a later "the excess damage dealt this way".
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub excess_damage: i32,
    /// Committed loss across all players in the scope's latest LoseLife.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub life_lost: i32,
    /// The cards the resolution's latest discard or mill moved ("this way").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub this_way: Vec<ObjectId>,
    /// Exact objects selected by the enclosing selection operation.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub selected: Vec<ObjectId>,
    /// Private membership from which the selected exact objects were chosen.
    /// Inherited child scopes keep it until those objects leave that zone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_from: Option<ZoneSelection>,
    /// The objects the resolution's latest `Move` or `Sacrifice` moved, as
    /// they now exist ("it", "if you do").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub moved: Vec<ObjectId>,
    /// Whether `Effect::Reveal` revealed the `selected` objects; a later
    /// placement of one on top of its library keeps it public ("reveal it,
    /// then shuffle and put that card on top"). Inline as a flag: the frame
    /// is part of `GameState`'s size budget.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub selected_revealed: bool,
    /// The permanent the resolving player chose for the instruction now
    /// running over a `Chosen` selector; cleared once it has run.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub chosen: Option<ObjectId>,
    /// All exact objects from the latest creation, including replacement
    /// multiplicity. A following "that token" refers to the whole result.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub created: Vec<ObjectId>,
    /// Chosen slots, with eligibility retained across instructions and choices.
    pub targets: Vec<ResolutionTarget>,
    pub active_requirement: usize,
    /// Trigger-context object (e.g. the spell that targeted a warded
    /// permanent) — used as the primary target when none was chosen. For an
    /// activated ability of an Aura or Equipment, the object it was attached to.
    pub context_target: Option<Target>,
    /// The trigger's player ("they": the caster of the triggering spell).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triggering_player: Option<PlayerId>,
    /// The exact object that granted the resolving ability (CR 400.7: once
    /// it leaves the battlefield, references to it find nothing).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub grantor: Option<ObjectId>,
    /// The printed index of the resolving triggered ability on its source,
    /// which keys its "do this only once each turn" use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ability: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ability_face: Option<crate::state::card::CardDefId>,
    /// Effects still to execute.
    pub queue: VecDeque<Effect>,
    pub completion: FrameCompletion,
}

/// Runtime progress through one authored `ForEachTarget`, not another program.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TargetIteration {
    pub remaining: VecDeque<ResolutionTarget>,
    pub effects: Vec<Effect>,
}

impl EffectFrame {
    pub fn source(&self) -> Option<ObjectId> {
        self.source_identity.map(|source| source.object())
    }
    pub fn source_ref(&self) -> Option<ObjectId> {
        self.source_identity.and_then(|source| source.reference())
    }
    pub(crate) fn bind_source(&mut self, source: Option<ObjectId>, reference: Option<ObjectId>) {
        self.source_identity =
            crate::state::game_object::AbilitySource::optional(source, reference);
    }

    /// Give the current instruction an isolated scope and retain its suffix.
    /// Completing the scope emits its fact exactly once through `run_frame`.
    pub(crate) fn complete_with_event(&mut self, event: GameEvent) {
        let child = self.child_scope(Vec::new());
        let parent = std::mem::replace(self, child);
        self.completion = FrameCompletion::Event {
            event,
            frame: Box::new(parent),
        };
    }

    /// A fresh scope with no source, results or stack cleanup. Callers bind
    /// their source and targets; only a root spell supplies finalization.
    pub(crate) fn new(controller: PlayerId, effects: Vec<Effect>) -> Self {
        Self {
            program: Vec::new(),
            batch: None,
            iteration: None,
            remembered: Vec::new(),
            inherited_from: None,
            source_transforms: 0,
            source_identity: None,
            controller,
            resolutions_this_turn: 0,
            triggering_amount: 0,
            kicked: false,
            x: 0,
            spent: Default::default(),
            excess_damage: 0,
            life_lost: 0,
            this_way: Vec::new(),
            selected: Vec::new(),
            selected_revealed: false,
            selected_from: None,
            moved: Vec::new(),
            chosen: None,
            created: Vec::new(),
            targets: Vec::new(),
            active_requirement: 0,
            context_target: None,
            triggering_player: None,
            grantor: None,
            ability: None,
            ability_face: None,
            queue: effects.into(),
            completion: FrameCompletion::Priority,
        }
    }

    fn child_scope(&self, effects: Vec<Effect>) -> Self {
        Self {
            program: self.program.clone(),
            selected: self.selected.clone(),
            selected_revealed: self.selected_revealed,
            selected_from: self.selected_from.clone(),
            remembered: self.remembered.clone(),
            inherited_from: self.inherited_from,
            source_transforms: self.source_transforms,
            source_identity: self.source_identity,
            resolutions_this_turn: self.resolutions_this_turn,
            triggering_amount: self.triggering_amount,
            kicked: self.kicked,
            x: self.x,
            spent: self.spent.clone(),
            targets: self.targets.clone(),
            active_requirement: self.active_requirement,
            context_target: self.context_target,
            triggering_player: self.triggering_player,
            grantor: self.grantor,
            ability: self.ability,
            ability_face: self.ability_face,
            ..Self::new(self.controller, effects)
        }
    }

    pub(crate) fn select_scope(
        &mut self,
        selected: Vec<ObjectId>,
        from: Option<ZoneSelection>,
        effects: Vec<Effect>,
    ) {
        let mut child = self.child_scope(effects);
        child.selected = selected;
        child.selected_revealed = false;
        child.selected_from = from;
        let parent = std::mem::replace(self, child);
        self.completion = FrameCompletion::Frame(Box::new(parent));
    }

    /// Start the next child with an isolated result scope. The interrupted
    /// scope keeps its targets, results, suffix and finalization unchanged.
    fn next_iteration(mut self) -> Self {
        let iteration = self.iteration.as_mut().expect("pending iteration");
        let Some(target) = iteration.remaining.pop_front() else {
            self.iteration = None;
            return self;
        };
        let effects = iteration.effects.clone();
        let mut child = self.child_scope(effects);
        child.targets = vec![ResolutionTarget {
            requirement: 0,
            ..target
        }];
        child.active_requirement = 0;
        child.context_target = None;
        child.completion = FrameCompletion::Frame(Box::new(self));
        child
    }

    /// Run a selected branch before the remaining effects, in authored order.
    pub fn prepend_effects(&mut self, effects: impl DoubleEndedIterator<Item = Effect>) {
        for effect in effects.rev() {
            self.queue.push_front(effect);
        }
    }

    /// Eligible chosen slots for the active clause, in original order.
    pub fn active_targets(&self) -> impl Iterator<Item = ResolutionTarget> + '_ {
        self.targets
            .iter()
            .copied()
            .filter(|slot| slot.legal && slot.requirement == self.active_requirement)
    }

    /// Trigger context is untargeted; it cannot replace an illegal chosen slot.
    pub fn primary_target(&self) -> Option<Target> {
        self.active_targets()
            .next()
            .map(|slot| slot.target)
            .or_else(|| self.context_target.filter(|_| self.targets.is_empty()))
    }
}

/// Where a `CastDuringResolution` decision's candidates come from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ResolutionCastOffer {
    Graveyard {
        predicate: CardPredicate,
        exile_instants_and_sorceries: bool,
    },
    /// Spells in the deciding player's hand matching `predicate`, read when
    /// offered (Yue, the Moon Spirit).
    Hand { predicate: CardPredicate },
    /// One exiled card, put into its owner's hand if not cast (Solstice
    /// Revelations).
    Exiled {
        #[serde(rename = "card")]
        object: ObjectId,
        below: i32,
    },
    /// The top card of the deciding player's library, which they have looked
    /// at and which stays there if not cast (Planetarium of Wan Shi Tong).
    LibraryTop {
        #[serde(rename = "card")]
        object: ObjectId,
    },
    /// Cards matching `predicate` the deciding player owns that are exiled
    /// with the exact `source` (Boiling Rock Rioter).
    ExiledWith {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        source: ObjectId,
        predicate: CardPredicate,
    },
    /// Exact appearances exiled together, cast in the chooser's order.
    AmongExiled { objects: Vec<ObjectId> },
    /// Cascade's hit `card`, among every card the cascade `exiled`; those not
    /// cast go on the bottom of the library in a random order.
    Cascade {
        #[serde(rename = "card")]
        object: ObjectId,
        exiled: Vec<ObjectId>,
        below: u8,
    },
}

/// The operation completed by a mana payment. It owns its saved caller;
/// nested fees cannot contribute to an outer spell's receipt.
// `Decision::ManaSpend` holds this in a `Box`, so the largest variant costs
// nothing inline.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum SpendCompletion {
    /// The mana of an announced spell or activation; its receipt gains the
    /// units spent as the spend completes.
    Announced(Payment),
    Upfront(Box<Decision>),
    Resolution(PaymentProgress),
    Foretell(CardId),
}

/// What a `ManaChoice` adds mana for.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ManaProduction {
    /// The mana ability's output, from `source` (absent for a choice with no
    /// permanent source). `tapped` is present when the ability tapped
    /// `source` for mana (CR 106.12); the triggered mana abilities that
    /// triggers are read from the board when this output is produced
    /// (CR 106.12a).
    Output {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        source: Option<ObjectId>,
        tapped: Option<TappedForMana>,
    },
    /// The type that `source`'s triggered mana ability adds after
    /// `tapped`'s ability produced `produced`, chosen by the tapping player.
    /// `pending` are the triggered mana abilities of the same production
    /// still to resolve after this choice (CR 605.4a).
    Bonus {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        source: ObjectId,
        #[serde(serialize_with = "crate::state::game_object::exact")]
        tapped: ObjectId,
        produced: crate::state::mana::Mana,
        pending: Vec<PendingManaBonus>,
    },
}

/// A permanent tapped for mana, as it was once tapped (CR 106.12): the
/// triggered mana ability predicates on the board then that it matched. A
/// permanent the same ability sacrifices answers those triggers from this
/// record, its last known information.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct TappedForMana {
    pub matched: Vec<CardPredicate>,
}

/// A triggered mana ability's output waiting to resolve after a production
/// (CR 605.4a), with the permanent whose ability it is.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PendingManaBonus {
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub source: ObjectId,
    pub output: crate::state::card::TriggeredManaOutput,
}

/// A pending mid-resolution decision, including its own bookkeeping so a
/// multi-step choice (scry card-by-card, pick-by-pick selection) re-suspends
/// with updated state.
/// The retained input meaning of migrated selections. This records their
/// original domain, never an alternate executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum SelectionOrigin {
    Milled,
    Library,
    Hand,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Decision {
    AssignCombatDamage {
        player: PlayerId,
        attacker: ObjectId,
        blocker: ObjectId,
        minimum: i32,
        remaining: i32,
        last: bool,
        pass: crate::flow::game::CombatDamagePass,
        assigned: Vec<i32>,
    },
    OrderReplacement {
        player: PlayerId,
        candidates: Vec<ReplacementCandidate>,
    },
    /// "You may have this creature enter as a copy of [a creature]" (CR
    /// 707.5): the entering card's controller chooses one of `candidates`
    /// or declines before its move commits; the copy's own entry
    /// replacements then apply (CR 614.12).
    EnterAsCopy {
        player: PlayerId,
        #[serde(rename = "card")]
        object_id: ObjectId,
        candidates: Vec<PermanentId>,
    },
    /// Choose a name from the immutable installed vocabulary, one ordinary
    /// decision per level: a scope larger than `NAME_BRANCHING` offers its
    /// balanced child groups, and choosing one narrows `scope` without
    /// committing anything. Only choosing a name in a leaf scope materializes
    /// an effect. Clones retain constant-size bookkeeping.
    NameCard {
        player: PlayerId,
        count: usize,
        names: std::sync::Arc<crate::cards::names::NameVocabulary>,
        scope: crate::cards::names::NameScope,
    },
    /// Select the generic X of an optional resolution payment, or decline.
    ChoosePaymentX {
        player: PlayerId,
        if_paid: Vec<Effect>,
    },
    /// Privately search the owner's graveyard, hand and library for up to
    /// `remaining` cards of a named identity. Every pick is an exact card;
    /// stopping or reaching the bound shuffles the searched library.
    SearchNamed {
        player: PlayerId,
        owner: PlayerId,
        name: String,
        remaining: usize,
    },
    Payment(PaymentProgress),
    /// Spend one exact mana unit. Every option preserves saved obligations.
    ManaSpend {
        player: PlayerId,
        cost: ManaCost,
        spent: PaymentReceipt,
        completion: Box<SpendCompletion>,
    },
    /// A mana ability's printed upfront cost, before its output exists.
    ManaUpfront {
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
        #[serde(skip_serializing_if = "crate::state::is_zero")]
        tap_remaining: usize,
        resume: Option<Box<Decision>>,
    },
    /// A choice of mana to add during a mana production: the mana ability's
    /// own output, or a triggered mana ability's type after it
    /// (`production`).
    ManaChoice {
        player: PlayerId,
        choices: Vec<crate::state::mana::Mana>,
        production: ManaProduction,
        resume: Option<Box<Decision>>,
    },
    /// `player` chooses a permanent they control matching `predicate` for
    /// `effect`, the instruction over a `Chosen` selector ("sacrifice a
    /// creature", "tap a creature you control"); `effect` then runs over
    /// the choice.
    ChooseObject {
        player: PlayerId,
        predicate: CardPredicate,
        /// The permanent an "another" or "all other" leaves out, never a
        /// candidate.
        #[serde(skip_serializing_if = "Option::is_none")]
        excluded: Option<PermanentId>,
        /// "You may sacrifice": declining is allowed.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        optional: bool,
        effect: Box<Effect>,
    },
    /// Place one of `remaining` `kind` counters on a tapped creature an
    /// opponent of `player` controls; candidates are read at each placement.
    DistributeCounters {
        player: PlayerId,
        kind: CounterKind,
        remaining: usize,
    },
    /// Select exact objects without moving them until the subset is complete.
    SelectCards {
        origin: SelectionOrigin,
        player: PlayerId,
        from: Option<ZoneSelection>,
        #[serde(serialize_with = "crate::state::game_object::exact")]
        candidates: Vec<ObjectId>,
        #[serde(serialize_with = "crate::state::game_object::exact")]
        selected: Vec<ObjectId>,
        min: usize,
        max: usize,
        effects: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    /// Choose cards in their desired top/bottom order (CR 701.26).
    Scry {
        player: PlayerId,
        /// All inspected cards stay pinned until the whole reorder commits.
        looked: Vec<ObjectId>,
        remaining: Vec<ObjectId>,
        top: Vec<ObjectId>,
        bottom: Vec<ObjectId>,
    },
    /// Surveil (CR 701.25): select cards for the graveyard, then order
    /// the kept cards on top through the shared library-order decision.
    Surveil {
        player: PlayerId,
        /// All inspected cards remain private and in the library until the
        /// partition is complete. Choosing one does not commit a zone move.
        looked: Vec<ObjectId>,
        graveyard: Vec<ObjectId>,
    },
    /// Look at top N, select up to K matching cards to hand, rest to the
    /// bottom of the library in the specified order.
    LookAndSelect {
        random_order: bool,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        reveal: bool,
        /// Selected cards go onto the battlefield, not into the hand.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        to_battlefield: bool,
        player: PlayerId,
        /// Looked-at cards still in the library, top first.
        looked: Vec<ObjectId>,
        predicate: CardPredicate,
        selected: usize,
        min_select: usize,
        max_select: usize,
    },
    /// Order inspected cards, first to draw first. `cards[..next]` is the
    /// committed prefix; the suffix contains the remaining choices. Top
    /// ordering keeps the whole inspected segment pinned and visible.
    OrderLibrary {
        player: PlayerId,
        #[serde(rename = "cards")]
        object_ids: Vec<ObjectId>,
        next: usize,
        end: LibraryEnd,
    },
    /// Search for up to `remaining` more [predicate] cards with names
    /// different from every one already `selected`, revealing each. The
    /// revealed cards stay in the library until `end` places them; an
    /// opponent's split is `OpponentSplitsRevealed`.
    SearchDistinctNames {
        player: PlayerId,
        predicate: CardPredicate,
        remaining: usize,
        selected: Vec<ObjectId>,
        end: DistinctSearchEnd,
    },
    /// `player` chooses a card matching `predicate` from `owner`'s revealed
    /// hand; `owner` discards it. Candidates are read from the current hand,
    /// one per definition, so resampled physical copies stay consistent.
    DiscardFromRevealedHand {
        player: PlayerId,
        owner: PlayerId,
        predicate: CardPredicate,
    },
    /// An opponent chooses `remaining` more of `owner`'s `revealed` library
    /// cards for the graveyard; the rest enter tapped, then `owner` shuffles.
    OpponentSplitsRevealed {
        player: PlayerId,
        owner: PlayerId,
        revealed: Vec<ObjectId>,
        remaining: usize,
    },
    /// "You may cast [a spell] (without paying its mana cost)" while an
    /// effect resolves (CR 608.2g): choose a candidate from `offer`, or
    /// decline. Timing and zone permissions don't apply.
    CastDuringResolution {
        player: PlayerId,
        offer: ResolutionCastOffer,
        /// The caster pays the spell's costs ("you may cast"); otherwise it
        /// is cast without paying its mana cost.
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        paid: bool,
        /// The once-each-turn option `(exact source, printed face, ability index)` casting
        /// spends ("Do this only once each turn"); declining keeps it.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(serialize_with = "crate::state::game_object::exact")]
        spends: Option<OnceEachTurnKey>,
    },
    /// `player` exiles a matching card from `owner`'s graveyard or a
    /// nontoken permanent `owner` controls (or declines, when `up_to`);
    /// `grantee` may then cast it as `grant` allows while controlling
    /// `source`.
    ExileLinked {
        player: PlayerId,
        owner: PlayerId,
        zone: ZoneType,
        predicate: CardPredicate,
        up_to: bool,
        grant: LinkedGrant,
        grantee: PlayerId,
        #[serde(serialize_with = "crate::state::game_object::exact")]
        source: Option<ObjectId>,
    },
    /// "Choose one —".
    Modal {
        player: PlayerId,
        modes: Vec<Vec<Effect>>,
    },
    /// Learn: retrieve a Lesson, discard then draw, or do neither.
    Learn {
        player: PlayerId,
    },
    Discard {
        player: PlayerId,
        remaining: usize,
    },
    /// `player` chooses another creature they control to keep, within
    /// `max_total_power` with those already `kept` (by both players), or
    /// stops; once the nonactive player stops, every other creature is
    /// sacrificed.
    KeepCreatures {
        player: PlayerId,
        max_total_power: i32,
        /// "Sacrifices N creatures of their choice": the player keeps all
        /// but this many and may not stop sooner.
        #[serde(skip_serializing_if = "Option::is_none")]
        sacrifice: Option<usize>,
        kept: Vec<PermanentId>,
    },
    /// `player` chooses a new target for slot `slot` of the stack object
    /// (CR 115.7), or keeps the current one when `may_keep` ("you may
    /// choose new targets for the copy"); later slots follow.
    Retarget {
        player: PlayerId,
        object: Target,
        slot: usize,
        may_keep: bool,
    },
    /// "Starting with you, each player chooses up to one [predicate]
    /// permanent from among permanents your opponents control. Exile those
    /// permanents." `player` chooses now; `controller` is "you". Once the
    /// last player has chosen, every chosen permanent is exiled together.
    ChooseToExile {
        player: PlayerId,
        controller: PlayerId,
        predicate: CardPredicate,
        chosen: Vec<PermanentId>,
    },
    /// All legend selections precede the simultaneous SBA batch (CR 704.5j).
    LegendRule {
        player: PlayerId,
        groups: Vec<Vec<ObjectId>>,
        keep: Vec<ObjectId>,
    },
}

impl Decision {
    pub fn player(&self) -> PlayerId {
        match self {
            Decision::Payment(payment) => payment.player,
            Decision::AssignCombatDamage { player, .. }
            | Decision::OrderReplacement { player, .. }
            | Decision::EnterAsCopy { player, .. }
            | Decision::ChoosePaymentX { player, .. }
            | Decision::NameCard { player, .. }
            | Decision::SearchNamed { player, .. }
            | Decision::ManaUpfront { player, .. }
            | Decision::ManaChoice { player, .. }
            | Decision::ManaSpend { player, .. }
            | Decision::ChooseObject { player, .. }
            | Decision::KeepCreatures { player, .. }
            | Decision::ChooseToExile { player, .. }
            | Decision::Retarget { player, .. }
            | Decision::DistributeCounters { player, .. } => *player,
            Decision::SelectCards { player, .. }
            | Decision::SearchDistinctNames { player, .. }
            | Decision::DiscardFromRevealedHand { player, .. }
            | Decision::CastDuringResolution { player, .. }
            | Decision::ExileLinked { player, .. }
            | Decision::OpponentSplitsRevealed { player, .. }
            | Decision::Scry { player, .. }
            | Decision::Surveil { player, .. }
            | Decision::OrderLibrary { player, .. }
            | Decision::LookAndSelect { player, .. }
            | Decision::Modal { player, .. }
            | Decision::Learn { player }
            | Decision::Discard { player, .. }
            | Decision::LegendRule { player, .. } => *player,
        }
    }

    /// Every library card this decision's chooser has looked at, including
    /// cards an `OrderLibrary` has already placed on the bottom.
    pub(crate) fn inspected_library_objects(&self) -> Option<(PlayerId, &[ObjectId])> {
        match self {
            Self::OrderLibrary {
                player, object_ids, ..
            } => Some((*player, object_ids)),
            _ => self.library_inspection(),
        }
    }

    /// Private inspection domains that survive library resampling. Search
    /// offers over an entire zone have a different membership authority.
    pub(crate) fn library_inspection(&self) -> Option<(PlayerId, &[ObjectId])> {
        match self {
            Self::Scry { player, looked, .. }
            | Self::Surveil { player, looked, .. }
            | Self::LookAndSelect { player, looked, .. } => Some((*player, looked)),
            Self::OrderLibrary {
                player,
                object_ids: objects,
                next,
                end,
            } => Some((
                *player,
                match end {
                    LibraryEnd::Top => objects,
                    LibraryEnd::Bottom => &objects[*next..],
                },
            )),
            Self::CastDuringResolution {
                player,
                offer: ResolutionCastOffer::LibraryTop { object },
                ..
            } => Some((*player, std::slice::from_ref(object))),
            _ => None,
        }
    }

    /// Exact top-of-library cards inspected by scry/look decisions, top first.
    /// These enter the deciding player's observation and stay pinned during
    /// `Game::determinize`. Search candidates are derived separately.
    pub fn revealed_objects(&self) -> &[ObjectId] {
        match self {
            Decision::ChoosePaymentX { .. }
            | Decision::NameCard { .. }
            | Decision::SearchNamed { .. }
            | Decision::ManaUpfront { .. }
            | Decision::Payment(_)
            | Decision::ManaChoice { .. }
            | Decision::ManaSpend { .. }
            | Decision::ChooseObject { .. }
            | Decision::KeepCreatures { .. }
            | Decision::ChooseToExile { .. }
            | Decision::Retarget { .. }
            | Decision::DistributeCounters { .. } => &[],
            Decision::Scry { looked, .. } => looked,
            Decision::OrderLibrary {
                object_ids: objects,
                next,
                end,
                ..
            } => match end {
                LibraryEnd::Top => objects,
                LibraryEnd::Bottom => &objects[*next..],
            },
            Decision::LookAndSelect { looked, .. } | Decision::Surveil { looked, .. } => looked,
            Decision::SearchDistinctNames { selected, .. } => selected,
            Decision::OpponentSplitsRevealed { revealed, .. } => revealed,
            Decision::CastDuringResolution {
                offer: ResolutionCastOffer::LibraryTop { object },
                ..
            } => std::slice::from_ref(object),
            Decision::AssignCombatDamage { .. }
            | Decision::OrderReplacement { .. }
            | Decision::EnterAsCopy { .. }
            | Decision::SelectCards { .. }
            | Decision::DiscardFromRevealedHand { .. }
            | Decision::CastDuringResolution { .. }
            | Decision::ExileLinked { .. }
            | Decision::Modal { .. }
            | Decision::Learn { .. }
            | Decision::Discard { .. }
            | Decision::LegendRule { .. } => &[],
        }
    }
}

/// `player` selects one of the objects; `decline` adds the option to choose none.
fn select_objects(
    player: PlayerId,
    objects: impl IntoIterator<Item = ObjectId>,
    decline: bool,
) -> Vec<Action> {
    objects
        .into_iter()
        .map(|object| Action::SelectCard { player, object })
        .chain(decline.then_some(Action::Decline { player }))
        .collect()
}

/// A resolution paused on a decision.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SuspendedResolution {
    pub frame: EffectFrame,
    pub decision: Decision,
}

/// A borrowed part of the retained caller chain, not a separate rules program.
pub(crate) enum ResolutionRef<'a> {
    Frame(&'a EffectFrame),
    Decision(&'a Decision),
}

impl SuspendedResolution {
    /// Walk the complete caller chain, outermost first. Projection and private
    /// inspection use this same order across both child scopes and mana costs.
    pub(crate) fn continuation(&self) -> Vec<ResolutionRef<'_>> {
        fn completion_chain<'a>(completion: &'a FrameCompletion, out: &mut Vec<ResolutionRef<'a>>) {
            match completion {
                FrameCompletion::Frame(parent) | FrameCompletion::Event { frame: parent, .. } => {
                    frame_chain(parent, out)
                }
                FrameCompletion::Decision(caller) => {
                    frame_chain(&caller.frame, out);
                    decision_chain(&caller.decision, out);
                }
                FrameCompletion::StateBased { next } => completion_chain(next, out),
                FrameCompletion::Paid { .. }
                | FrameCompletion::Activated { .. }
                | FrameCompletion::EndTurn
                | FrameCompletion::Priority
                | FrameCompletion::Continue
                | FrameCompletion::Spell { .. } => {}
            }
        }
        fn frame_chain<'a>(frame: &'a EffectFrame, out: &mut Vec<ResolutionRef<'a>>) {
            completion_chain(&frame.completion, out);
            out.push(ResolutionRef::Frame(frame));
        }
        fn decision_chain<'a>(decision: &'a Decision, out: &mut Vec<ResolutionRef<'a>>) {
            if let Decision::ManaSpend { completion, .. } = decision {
                if let SpendCompletion::Upfront(next) = completion.as_ref() {
                    decision_chain(next, out);
                }
            }
            if let Decision::ManaUpfront {
                resume: Some(parent),
                ..
            }
            | Decision::ManaChoice {
                resume: Some(parent),
                ..
            } = decision
            {
                decision_chain(parent, out);
            }
            out.push(ResolutionRef::Decision(decision));
        }
        let mut chain = Vec::new();
        frame_chain(&self.frame, &mut chain);
        decision_chain(&self.decision, &mut chain);
        chain
    }

    /// A child's chooser does not end the retained caller's inspection rights.
    pub(crate) fn decisions(&self) -> impl Iterator<Item = &Decision> {
        self.continuation()
            .into_iter()
            .filter_map(|part| match part {
                ResolutionRef::Decision(decision) => Some(decision),
                ResolutionRef::Frame(_) => None,
            })
    }
}

impl Game {
    /// Record that the exact source took its ability's once-each-turn option.
    pub(crate) fn spend_once_each_turn(&mut self, key: OnceEachTurnKey) {
        *self
            .state
            .turn
            .ability_triggers_this_turn
            .entry(key)
            .or_default() += 1;
    }

    /// The creatures a pending `Decision::KeepCreatures` may add: its
    /// player's, not yet kept, within the power bound with that player's
    /// kept creatures.
    pub(crate) fn keep_candidates(&self, decision: &Decision) -> Vec<PermanentId> {
        let Decision::KeepCreatures {
            player,
            max_total_power,
            sacrifice,
            kept,
        } = decision
        else {
            return Vec::new();
        };
        let creatures: Vec<_> = self
            .battlefield_permanents(*player)
            .into_iter()
            .filter(|id| self.characteristics(*id).is_creature())
            .collect();
        if let Some(count) = sacrifice {
            let keeps = creatures.len().saturating_sub(*count);
            let kept_own = creatures.iter().filter(|id| kept.contains(id)).count();
            return if kept_own < keeps {
                creatures
                    .into_iter()
                    .filter(|id| !kept.contains(id))
                    .collect()
            } else {
                Vec::new()
            };
        }
        let total: i32 = kept
            .iter()
            .filter(|id| {
                self.state.permanents[**id]
                    .as_ref()
                    .is_some_and(|permanent| permanent.controller == *player)
            })
            .map(|id| self.characteristics(*id).power)
            .sum();
        self.battlefield_permanents(*player)
            .into_iter()
            .filter(|id| {
                self.characteristics(*id).is_creature()
                    && !kept.contains(id)
                    && total + self.characteristics(*id).power <= *max_total_power
            })
            .collect()
    }

    /// Ask `player` to keep a creature, moving on to the nonactive player
    /// (and finally sacrificing the rest) when `player` has no candidate.
    pub(crate) fn keep_creatures(
        &mut self,
        player: PlayerId,
        max_total_power: i32,
        sacrifice: Option<usize>,
        kept: Vec<PermanentId>,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        let decision = Decision::KeepCreatures {
            player,
            max_total_power,
            sacrifice,
            kept,
        };
        if !self.keep_candidates(&decision).is_empty() {
            return Some(decision);
        }
        let Decision::KeepCreatures { kept, .. } = decision else {
            unreachable!()
        };
        self.stop_keeping_creatures(player, max_total_power, sacrifice, kept, frame)
    }

    /// The permanents a `Decision::ChooseToExile` may add: matching, controlled
    /// by an opponent of its controller, not yet chosen.
    fn exile_choice_candidates(
        &self,
        controller: PlayerId,
        predicate: &CardPredicate,
        chosen: &[PermanentId],
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(PlayerId(1 - controller.0))
            .into_iter()
            .filter(|id| !chosen.contains(id) && self.permanent_matches_predicate(*id, predicate))
            .collect()
    }

    /// Ask `player` to choose, skipping a player with nothing to choose;
    /// after the last player (the controller's opponent), exile the choices.
    pub(crate) fn choose_to_exile(
        &mut self,
        player: PlayerId,
        controller: PlayerId,
        predicate: CardPredicate,
        chosen: Vec<PermanentId>,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        if self
            .exile_choice_candidates(controller, &predicate, &chosen)
            .is_empty()
        {
            return self.next_exile_chooser(player, controller, predicate, chosen, frame);
        }
        Some(Decision::ChooseToExile {
            player,
            controller,
            predicate,
            chosen,
        })
    }

    fn next_exile_chooser(
        &mut self,
        player: PlayerId,
        controller: PlayerId,
        predicate: CardPredicate,
        chosen: Vec<PermanentId>,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        if player == controller {
            return self.choose_to_exile(
                PlayerId(1 - controller.0),
                controller,
                predicate,
                chosen,
                frame,
            );
        }
        frame.batch = Some(
            self.event_batch(
                EventBinding::None,
                chosen
                    .iter()
                    .filter_map(|id| self.permanent_object(*id))
                    .map(|object| {
                        self.zone_move_proposal(object, ZoneType::Exile, Default::default())
                    }),
            ),
        );
        None
    }

    /// `player` has stopped keeping creatures: the nonactive player chooses
    /// next, or every unkept creature is sacrificed.
    fn stop_keeping_creatures(
        &mut self,
        player: PlayerId,
        max_total_power: i32,
        sacrifice: Option<usize>,
        kept: Vec<PermanentId>,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        if player == self.active_player() {
            return self.keep_creatures(
                PlayerId(1 - player.0),
                max_total_power,
                sacrifice,
                kept,
                frame,
            );
        }
        let events = [self.active_player(), PlayerId(1 - self.active_player().0)]
            .into_iter()
            .flat_map(|player| self.battlefield_permanents(player))
            .filter(|id| self.characteristics(*id).is_creature() && !kept.contains(id))
            .filter_map(|id| self.permanent_object(id))
            .map(|target| crate::flow::proposed_event::ProposedEvent::Sacrifice { target });
        frame.batch = Some(self.event_batch(EventBinding::None, events));
        None
    }

    /// The permanents a pending `Decision::ChooseObject` may select: those
    /// `player` controls matching the predicate, less the excluded one.
    pub(crate) fn choice_candidates(&self, decision: &Decision) -> Vec<PermanentId> {
        let Decision::ChooseObject {
            player,
            predicate,
            excluded,
            ..
        } = decision
        else {
            return Vec::new();
        };
        self.battlefield_permanents(*player)
            .into_iter()
            .filter(|id| self.permanent_matches_predicate(*id, predicate))
            .filter(|id| Some(*id) != *excluded)
            .collect()
    }

    /// Temporary native permission to inspect another player's hidden zones.
    /// It ends with the search instruction and never depends on deck disclosure.
    pub(crate) fn searches_private_zones(&self, viewer: PlayerId, owner: PlayerId) -> bool {
        self.state
            .suspended_decision
            .as_ref()
            .is_some_and(|suspended| {
                suspended.decisions().any(|decision| {
                    matches!(decision, Decision::SearchNamed { player, owner: searched, .. }
                    if self.shares_information(viewer, *player) && *searched == owner)
                })
            })
    }

    /// Whether `viewer` (or a player sharing information with them) is
    /// inspecting `owner`'s library: a named search, or a pending selection
    /// from that library ("search target opponent's library"). The library
    /// selection grants no hand access, but its exact library cards fix the
    /// hidden partition, so sampled worlds keep `owner`'s zones as they are.
    pub(crate) fn inspects_library(&self, viewer: PlayerId, owner: PlayerId) -> bool {
        self.searches_private_zones(viewer, owner)
            || self
                .state
                .suspended_decision
                .as_ref()
                .is_some_and(|suspended| {
                    suspended.decisions().any(|decision| {
                        matches!(decision, Decision::SelectCards { player, from: Some(source), .. }
                        if self.shares_information(viewer, *player)
                            && source.owner == owner
                            && source.zone == ZoneType::Library)
                    })
                })
    }

    /// Search permission reveals all cards in the searched zones only to
    /// the chooser; eligibility by name restricts actions, not information.
    pub(crate) fn named_search_objects(
        &self,
        owner: PlayerId,
    ) -> impl Iterator<Item = ObjectId> + '_ {
        [ZoneType::Graveyard, ZoneType::Hand, ZoneType::Library]
            .into_iter()
            .flat_map(move |zone| self.state.zones.zone_objects(zone, owner).iter().copied())
    }

    pub(crate) fn named_search_cards(&self, owner: PlayerId) -> impl Iterator<Item = CardId> + '_ {
        self.named_search_objects(owner)
            .filter_map(|object| self.state.card_of(object))
    }

    fn object_focus(&self, ids: &[ObjectId]) -> Vec<PublicId> {
        ids.iter().map(|id| self.state[id].id).collect()
    }

    /// The current appearances of cards found in a zone, in their order.
    fn card_objects(&self, cards: impl IntoIterator<Item = CardId>) -> Vec<ObjectId> {
        cards
            .into_iter()
            .map(|card| self.state.object_of(card))
            .collect()
    }

    pub(crate) fn suspend_rule_decision(&mut self, decision: Decision) {
        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
            frame: EffectFrame {
                completion: FrameCompletion::Continue,
                ..EffectFrame::new(decision.player(), Vec::new())
            },
            decision,
        });
    }

    /// Finish the instruction's batch before exposing its next decision.
    /// Without a next decision, the frame runs the batch and its remaining effects.
    fn continue_resolution(&mut self, mut frame: EffectFrame, next: Option<Decision>) {
        match next {
            Some(decision) => {
                if let Some(batch) = frame.batch.take() {
                    let child = EffectFrame {
                        batch: Some(batch),
                        ..EffectFrame::new(frame.controller, Vec::new())
                    };
                    self.run_before_decision(child, SuspendedResolution { frame, decision });
                } else {
                    *self.state.suspended_decision_mut() =
                        Some(SuspendedResolution { frame, decision });
                }
            }
            None => self.run_frame(frame),
        }
    }

    /// Run cost effects before making the caller's next decision available.
    /// The caller has already committed its earlier costs; resumption restores
    /// its exact progress instead of invoking the payment entry point again.
    pub(crate) fn run_before_decision(
        &mut self,
        mut child: EffectFrame,
        caller: SuspendedResolution,
    ) {
        child.completion = FrameCompletion::Decision(Box::new(caller));
        self.run_frame(child);
    }

    /// Reach a stable board without granting priority. If an SBA needs a
    /// decision, its frame inherits the interrupted rule's return point.
    fn finish_sbas_before(&mut self, resume: FrameCompletion) -> bool {
        while !self.is_game_over() {
            if matches!(resume, FrameCompletion::EndTurn) {
                self.collect_waiting_triggers();
            }
            match self.perform_state_based_actions() {
                SbaProgress::Stable => return true,
                SbaProgress::Changed => continue,
                SbaProgress::Pending => {
                    self.state
                        .suspended_decision_mut()
                        .as_mut()
                        .expect("pending SBA")
                        .frame
                        .completion
                        .then(resume);
                    return false;
                }
            }
        }
        false
    }

    /// Execute a frame's remaining effects. If an effect suspends for a
    /// decision, park the frame; otherwise finalize the resolution.
    pub(crate) fn run_frame(&mut self, mut frame: EffectFrame) {
        while !self.is_game_over() {
            self.refresh_control();
            if let Some(mut batch) = frame.batch.take() {
                match self.advance_event_batch(&mut batch) {
                    BatchProgress::Choice { player, candidates } => {
                        frame.batch = Some(batch);
                        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
                            frame,
                            decision: Decision::OrderReplacement { player, candidates },
                        });
                        return;
                    }
                    BatchProgress::EntryCopy {
                        player,
                        object,
                        candidates,
                    } => {
                        frame.batch = Some(batch);
                        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
                            frame,
                            decision: Decision::EnterAsCopy {
                                player,
                                object_id: object,
                                candidates,
                            },
                        });
                        return;
                    }
                    BatchProgress::Complete { results, followups } => {
                        self.bind_event_results(&mut frame, &batch, &results);
                        // Cards returning as a linked exile ends are the
                        // instruction's next simultaneous event; its bound
                        // results are already its own.
                        if !followups.is_empty() {
                            frame.batch = Some(self.event_batch(EventBinding::None, followups));
                            continue;
                        }
                    }
                }
            }
            if frame.iteration.is_some() {
                frame = frame.next_iteration();
                continue;
            }
            let Some(effect) = frame.queue.pop_front() else {
                match frame.completion {
                    FrameCompletion::Event {
                        event,
                        frame: parent,
                    } => {
                        self.emit(event);
                        // Capture eligible sources at completion, before a
                        // caller's next instruction can remove them. Matching
                        // enqueues occurrences; it does not grant priority.
                        self.process_game_events();
                        frame = *parent;
                        continue;
                    }
                    FrameCompletion::Frame(parent) => {
                        frame = *parent;
                        continue;
                    }
                    FrameCompletion::Decision(caller) => {
                        *self.state.suspended_decision_mut() = Some(*caller);
                        return;
                    }
                    FrameCompletion::StateBased { next } => {
                        self.finish_state_based_batch();
                        frame.completion = *next;
                        continue;
                    }
                    FrameCompletion::EndTurn => {
                        if self.finish_sbas_before(FrameCompletion::EndTurn) {
                            self.end_step_for_cleanup();
                        }
                        return;
                    }
                    FrameCompletion::Paid { player, payment } => {
                        self.commit_spell_payment(player, payment);
                        if let Some(caller) = self.state.cast_caller_mut().take() {
                            let caller = *caller;
                            match &caller.decision {
                                Decision::CastDuringResolution {
                                    offer: ResolutionCastOffer::AmongExiled { .. },
                                    ..
                                } if !self
                                    .resolution_cast_candidates(&caller.decision)
                                    .is_empty() =>
                                {
                                    *self.state.suspended_decision_mut() = Some(caller);
                                    return;
                                }
                                Decision::CastDuringResolution {
                                    offer: ResolutionCastOffer::Cascade { exiled, .. },
                                    ..
                                } => {
                                    self.put_exiled_on_bottom_in_random_order(exiled.clone());
                                }
                                _ => {}
                            }
                            frame = caller.frame;
                            continue;
                        }
                    }
                    FrameCompletion::Activated {
                        mut ability,
                        source_after_cost,
                    } => {
                        // CR 400.7j: a cost that discarded the source finds
                        // the object it became.
                        if let Some(source) =
                            source_after_cost.and_then(|object| self.state.zones.successor(object))
                        {
                            ability.source_object = source;
                        }
                        self.emit_permanents_targeted(
                            &ability.targets,
                            Target::StackAbility(ability.id),
                            ability.controller,
                        );
                        self.push_to_stack(
                            crate::state::stack_object::StackObject::ActivatedAbility(ability),
                        );
                    }
                    FrameCompletion::Continue => return,
                    FrameCompletion::Spell { object, attach_to } => {
                        // A spell that exiled itself ("Exile Avatar's Wrath")
                        // has already left the stack (CR 608.2n); the entry
                        // batch always moves a card still on it.
                        if self.state.zones.zone_of(object) == Some(ZoneType::Stack) {
                            let index = self
                                .stack_index_of(Target::StackSpell(object))
                                .expect("resolving spell on stack");
                            let crate::state::stack_object::StackObject::Spell(spell) =
                                &self.state.stack_objects[index]
                            else {
                                unreachable!()
                            };
                            let profile = spell.profile;
                            let (to, entry) = if self.state[object].types.is_permanent() {
                                (
                                    ZoneType::Battlefield,
                                    crate::flow::proposed_event::ZoneEntry {
                                        controller: Some(frame.controller),
                                        transformed: profile == SpellProfile::BackFace,
                                        attached_to: attach_to,
                                        x: frame.x,
                                        // Spend effects that applied to
                                        // this spell as it was cast.
                                        counters: {
                                            let mut counters =
                                                crate::state::permanent::Counters::default();
                                            counters.change(
                                                crate::state::permanent::CounterKind::PlusOnePlusOne,
                                                i32::from(frame.spent.entry_counters()),
                                            );
                                            counters
                                        },
                                        ..Default::default()
                                    },
                                )
                            } else {
                                (
                                    if profile == SpellProfile::Omen {
                                        ZoneType::Library
                                    } else {
                                        ZoneType::Graveyard
                                    },
                                    crate::flow::proposed_event::ZoneEntry {
                                        resolving_adventure: (profile == SpellProfile::Adventure)
                                            .then_some(frame.controller),
                                        ..Default::default()
                                    },
                                )
                            };
                            let binding = if profile == SpellProfile::Omen {
                                EventBinding::Shuffle {
                                    player: self.state[object].owner,
                                }
                            } else {
                                EventBinding::None
                            };
                            frame.batch = Some(self.event_batch(
                                binding,
                                [self.zone_move_proposal(object, to, entry)],
                            ));
                            frame.completion = FrameCompletion::Spell { object, attach_to };
                            continue;
                        }
                        // CR 608.3a: the permanent is the object the spell
                        // became by resolving.
                        if let Some(permanent_id) = self
                            .state
                            .zones
                            .successor(object)
                            .and_then(|entered| self.state.permanent_of(entered))
                        {
                            // A keyword the spell gained from mana spent
                            // on it lasts the turn on the permanent.
                            let gained = frame.spent.keywords_gained();
                            if gained != crate::state::card::Keywords::default() {
                                let entered = self.state.permanents[permanent_id]
                                    .as_ref()
                                    .expect("entered")
                                    .object;
                                self.record_continuous(
                                    Some(entered),
                                    None,
                                    frame.controller,
                                    [crate::state::effect::Affected::Object(entered)],
                                    crate::state::effect::Modification::AddKeywords(gained),
                                    crate::state::effect::EffectDuration::EndOfTurn,
                                );
                            }
                        }
                        self.capture_resolution(&frame);
                        self.emit(GameEvent::SpellResolved { object_id: object });
                    }
                    FrameCompletion::Priority => self.capture_resolution(&frame),
                }
                self.restart_priority_round();
                return;
            };
            if let Some(decision) = self.execute_frame_effect(&effect, &mut frame) {
                *self.state.suspended_decision_mut() =
                    Some(SuspendedResolution { frame, decision });
                return;
            }
            // A chosen permanent belongs to the one instruction that ran.
            frame.chosen = None;
        }
    }

    /// Consume the instruction's results once its whole batch has committed.
    /// `results` pair with `batch.events` in order.
    fn bind_event_results(
        &mut self,
        frame: &mut EffectFrame,
        batch: &EventBatch,
        results: &[EventResult],
    ) {
        let moved = results.iter().filter_map(|result| result.moved);
        match batch.binding {
            EventBinding::None | EventBinding::Deaths | EventBinding::Combat => {}
            EventBinding::Destroyed => frame.moved = moved.collect(),
            EventBinding::LifeLost => {
                frame.life_lost = results.iter().fold(0_i32, |lost, result| {
                    lost.saturating_sub(result.life_delta.min(0))
                });
            }
            EventBinding::Bending { player, kind } => self.record_bending(player, kind),
            EventBinding::Shuffle { player } => self.shuffle_library(player),
            EventBinding::PlaceInLibrary { second_from_top } => {
                for object in moved {
                    if self.state.zones.zone_of(object) != Some(ZoneType::Library) {
                        continue;
                    }
                    if second_from_top {
                        let owner = self.state[object].owner;
                        let below_top = self
                            .state
                            .zones
                            .size(ZoneType::Library, owner)
                            .saturating_sub(2);
                        self.reposition_library_object(owner, object, below_top);
                    } else {
                        self.put_on_bottom_of_library(object);
                    }
                }
            }
            EventBinding::Created { attacking } => {
                frame.created = moved.collect();
                for &object in &frame.created {
                    self.finish_created_token(object, attacking);
                }
            }
            EventBinding::Damage { result: measured } => {
                // Each damage instruction measures its own excess: prevented
                // or redirected damage leaves none, never an earlier one's.
                frame.excess_damage = batch
                    .events
                    .iter()
                    .zip(results)
                    .filter(|(pending, _)| measured.is_none_or(|index| index == pending.ordinal))
                    .filter_map(|(_, result)| result.excess_damage)
                    .next_back()
                    .unwrap_or(0);
            }
            EventBinding::Moved { shuffle, .. } => {
                let is_discard = |event: &crate::flow::proposed_event::PendingEvent| {
                    matches!(
                        event.event,
                        crate::flow::proposed_event::ProposedEvent::ZoneMove {
                            from: Some(ZoneType::Hand),
                            to: ZoneType::Graveyard,
                            ..
                        }
                    )
                };
                if batch.events.iter().any(is_discard) {
                    frame.this_way = results
                        .iter()
                        .zip(&batch.events)
                        .filter(|(_, event)| is_discard(event))
                        .filter_map(|(result, _)| result.moved)
                        .collect();
                }
                frame.moved = moved.collect();
                let mut owners = Vec::new();
                for &object_id in &frame.moved {
                    let owner = self.state[object_id].owner;
                    match self.state.zones.zone_of(object_id) {
                        Some(ZoneType::Library) if shuffle => {
                            if !owners.contains(&owner) {
                                owners.push(owner);
                            }
                        }
                        Some(ZoneType::Hand) => self.invalidate_mana_cache(owner),
                        _ => {}
                    }
                }
                for owner in owners {
                    self.shuffle_library(owner);
                }
            }
            EventBinding::ThisWay => frame.this_way = moved.collect(),
        }
    }

    /// The action space for the pending mid-resolution decision, if any.
    pub(crate) fn suspended_decision_action_space(&self) -> Option<ActionSpace> {
        let _payments = self.payment_scan();
        let suspended = self.state.suspended_decision.as_ref()?;
        let source_focus = suspended
            .frame
            .source()
            .map(|object_id| {
                vec![
                    self.state
                        .objects
                        .provenance(object_id)
                        .expect("source provenance")
                        .0,
                ]
            })
            .unwrap_or_default();
        let space = match &suspended.decision {
            Decision::AssignCombatDamage {
                player,
                attacker,
                blocker,
                minimum,
                remaining,
                ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: (*minimum..=*remaining)
                    .map(|amount| Action::ChooseMode {
                        player: *player,
                        mode: amount as usize,
                    })
                    .collect(),
                focus: vec![self.state[*attacker].id, self.state[*blocker].id],
            },
            Decision::OrderReplacement { player, candidates } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: (0..candidates.len())
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
                focus: source_focus,
            },
            Decision::EnterAsCopy {
                player,
                object_id,
                candidates,
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::ChooseTarget,
                actions: candidates
                    .iter()
                    .map(|id| Action::ChooseTarget {
                        player: *player,
                        target: Target::Permanent(*id),
                    })
                    .chain([Action::Decline { player: *player }])
                    .collect(),
                focus: vec![self.state[*object_id].id],
            },
            Decision::ChoosePaymentX { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: (0..=u8::MAX)
                    .take_while(|x| {
                        self.can_pay_mana_cost(*player, &ManaCost::default().with_generic(*x))
                    })
                    .map(|x| Action::ChooseMode {
                        player: *player,
                        mode: usize::from(x),
                    })
                    .chain([Action::Decline { player: *player }])
                    .collect(),
                focus: source_focus,
            },
            Decision::SearchNamed {
                player,
                owner,
                name,
                remaining,
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(
                    *player,
                    self.card_objects(self.named_search_cards(*owner).filter(|card| {
                        self.state[self.state.object_of(*card)].name == *name && *remaining > 0
                    })),
                    true,
                ),
                focus: source_focus,
            },
            Decision::ManaUpfront {
                player,
                tap_remaining,
                ..
            } => {
                let (kind, actions) = if *tap_remaining > 0 {
                    (
                        ActionSpaceKind::LookAndSelect,
                        self.mana_tap_cost_actions(*player),
                    )
                } else {
                    let mut actions = self.mana_ability_actions(*player);
                    if self.offers_extra_payment_taps(*player) && self.can_pay_mana_upfront(*player)
                    {
                        actions.push(Action::PayCost { player: *player });
                    }
                    self.append_mana_floats(*player, &mut actions);
                    (ActionSpaceKind::PayOrNot, actions)
                };
                ActionSpace {
                    player: Some(*player),
                    kind,
                    actions,
                    focus: source_focus,
                }
            }
            Decision::Payment(payment) => {
                let player = payment.player;
                let (cost, help) = match &payment.owed {
                    CostComponent::Discard => {
                        let mut actions: Vec<_> = self
                            .discard_candidates(player, None)
                            .into_iter()
                            .map(|object| Action::SelectCard { player, object })
                            .collect();
                        actions.push(Action::Decline { player });
                        return Some(ActionSpace {
                            player: Some(player),
                            kind: ActionSpaceKind::Discard,
                            actions,
                            focus: source_focus,
                        });
                    }
                    // Paying life is one action under manual and automatic
                    // payment alike.
                    CostComponent::Life(life) => {
                        let mut actions = Vec::new();
                        if self.can_pay_life(player, *life) {
                            actions.push(Action::PayCost { player });
                        }
                        actions.push(Action::Decline { player });
                        return Some(ActionSpace {
                            player: Some(player),
                            kind: ActionSpaceKind::PayOrNot,
                            actions,
                            focus: source_focus,
                        });
                    }
                    CostComponent::Mana { cost, help } => (cost, *help),
                    _ => unreachable!("unexpected resolution payment component"),
                };
                let waterbend = help.waterbend_left(cost, &payment.receipt);
                if self.automatic_mana_payment(player) && payment.can_decline() {
                    let mut actions = Vec::new();
                    let pool = self.pooled_mana(player);
                    if self.payable_with_help(player, &pool, cost, help, waterbend, None) {
                        actions.push(Action::PayCost { player });
                    }
                    actions.push(Action::Decline { player });
                    return Some(ActionSpace {
                        player: Some(player),
                        kind: ActionSpaceKind::PayOrNot,
                        actions,
                        focus: source_focus,
                    });
                }
                let mut actions = self.mana_ability_actions(player);
                actions.extend(
                    self.help_taps(
                        player,
                        cost,
                        help,
                        waterbend,
                        &self.waterbend_candidates(player),
                        None,
                    )
                    .into_iter()
                    .map(|(permanent, _)| Action::WaterbendTap { player, permanent }),
                );
                if self.can_pay_mana_cost(player, cost) {
                    actions.push(Action::PayCost { player });
                }
                if payment.can_decline() {
                    actions.push(Action::Decline { player });
                }
                self.append_mana_floats(player, &mut actions);
                ActionSpace {
                    player: Some(player),
                    kind: if payment.waterbends() {
                        ActionSpaceKind::Waterbend
                    } else {
                        ActionSpaceKind::PayOrNot
                    },
                    actions,
                    focus: source_focus,
                }
            }
            Decision::ManaSpend { player, cost, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                focus: source_focus,
                actions: self
                    .mana_spend_choices(*player, cost)
                    .iter()
                    .enumerate()
                    .map(|(mode, _)| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
            },
            Decision::ManaChoice {
                player, choices, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                focus: source_focus,
                actions: (0..choices.len())
                    .filter(|mode| self.mana_choice_completes_payment(*player, &choices[*mode]))
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
            },
            Decision::ChooseObject {
                player, optional, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(
                    *player,
                    self.choice_candidates(&suspended.decision)
                        .into_iter()
                        .map(|id| self.state.permanents[id].as_ref().unwrap().object),
                    *optional,
                ),
                focus: source_focus,
            },
            Decision::Retarget {
                player, may_keep, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::ChooseTarget,
                actions: self
                    .retarget_candidates(&suspended.decision)
                    .into_iter()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target,
                    })
                    .chain(may_keep.then_some(Action::Decline { player: *player }))
                    .collect(),
                focus: source_focus,
            },
            Decision::ChooseToExile {
                player,
                controller,
                predicate,
                chosen,
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(
                    *player,
                    self.exile_choice_candidates(*controller, predicate, chosen)
                        .into_iter()
                        .map(|id| self.state.permanents[id].as_ref().unwrap().object),
                    true,
                ),
                focus: source_focus,
            },
            Decision::KeepCreatures {
                player, sacrifice, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(
                    *player,
                    self.keep_candidates(&suspended.decision)
                        .into_iter()
                        .map(|id| self.state.permanents[id].as_ref().unwrap().object),
                    sacrifice.is_none(),
                ),
                focus: source_focus,
            },
            Decision::DistributeCounters { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: self
                    .tapped_opponent_creatures(*player)
                    .into_iter()
                    .map(|id| Action::SelectCard {
                        player: *player,
                        object: self.state.permanents[id].as_ref().unwrap().object,
                    })
                    .collect(),
                focus: source_focus,
            },
            Decision::SearchDistinctNames { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(
                    *player,
                    self.card_objects(self.distinct_name_search_candidates(&suspended.decision)),
                    true,
                ),
                focus: source_focus,
            },
            Decision::DiscardFromRevealedHand { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Discard,
                actions: select_objects(
                    *player,
                    self.card_objects(self.revealed_hand_candidates(&suspended.decision)),
                    false,
                ),
                focus: source_focus,
            },
            Decision::ExileLinked { player, up_to, .. } => {
                let cards = self.card_objects(self.linked_exile_candidates(&suspended.decision));
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    focus: self.object_focus(&cards),
                    actions: select_objects(*player, cards, *up_to),
                }
            }
            Decision::CastDuringResolution { player, .. } => {
                let candidates = self.resolution_cast_candidates(&suspended.decision);
                let cards: Vec<_> = candidates
                    .iter()
                    .map(|(card, _)| self.state.object_of(*card))
                    .collect();
                let actions = candidates
                    .into_iter()
                    .map(|(card, profile)| {
                        let object = self.state.object_of(card);
                        if !profile.is_front() {
                            Action::CastSpell {
                                player: *player,
                                object,
                                profile,
                                route: None,
                            }
                        } else {
                            Action::SelectCard {
                                player: *player,
                                object,
                            }
                        }
                    })
                    .chain([Action::Decline { player: *player }])
                    .collect();
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    focus: self.object_focus(&cards),
                    actions,
                }
            }
            Decision::OpponentSplitsRevealed {
                player,
                revealed: object_ids,
                ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(*player, object_ids.iter().copied(), false),
                focus: self.object_focus(object_ids),
            },
            Decision::OrderLibrary {
                player,
                object_ids,
                next,
                ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: select_objects(*player, object_ids[*next..].iter().copied(), false),
                focus: self.object_focus(&object_ids[*next..]),
            },
            Decision::Discard { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Discard,
                actions: select_objects(
                    *player,
                    self.card_objects(self.state.zone_cards(ZoneType::Hand, *player)),
                    false,
                ),
                focus: source_focus,
            },

            Decision::SelectCards {
                player,
                candidates,
                selected,
                min,
                ..
            } => {
                let object_ids: Vec<_> = candidates
                    .iter()
                    .filter(|object| !selected.contains(object))
                    .filter(|object| self.state.live(**object).is_some())
                    .copied()
                    .collect();
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions: select_objects(
                        *player,
                        object_ids.iter().copied(),
                        selected.len() >= *min,
                    ),
                    focus: self.object_focus(&object_ids),
                }
            }
            Decision::Surveil {
                player,
                looked,
                graveyard,
            } => {
                let remaining: Vec<_> = looked
                    .iter()
                    .copied()
                    .filter(|object_id| !graveyard.contains(object_id))
                    .collect();
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions: select_objects(*player, remaining.iter().copied(), true),
                    focus: self.object_focus(&remaining),
                }
            }
            Decision::Scry {
                player, remaining, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Scry,
                actions: remaining
                    .iter()
                    .flat_map(|object_id| {
                        [false, true].map(|to_bottom| Action::ScryCard {
                            player: *player,
                            object: *object_id,
                            to_bottom,
                        })
                    })
                    .collect(),
                focus: self.object_focus(remaining),
            },
            Decision::LookAndSelect {
                player,
                looked,
                predicate,
                selected,
                min_select,
                ..
            } => {
                let mut actions: Vec<Action> = looked
                    .iter()
                    .filter(|object_id| self.object_matches_predicate(**object_id, predicate))
                    .map(|object_id| Action::SelectCard {
                        player: *player,
                        object: *object_id,
                    })
                    .collect();
                if *selected >= *min_select || actions.is_empty() {
                    actions.push(Action::Decline { player: *player });
                }
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions,
                    focus: self.object_focus(looked),
                }
            }
            Decision::NameCard { player, scope, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                focus: source_focus,
                actions: (0..scope.option_count())
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
            },
            Decision::Modal { player, modes } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: (0..modes.len())
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
                focus: source_focus,
            },
            Decision::LegendRule { player, groups, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LegendRule,
                actions: select_objects(*player, groups[0].iter().copied(), false),
                focus: self.object_focus(&groups[0]),
            },
            Decision::Learn { player } => {
                let mut actions: Vec<Action> = self
                    .state
                    .zone_cards(ZoneType::Hand, *player)
                    .map(|card| Action::LearnDiscard {
                        player: *player,
                        object: self.state.object_of(card),
                    })
                    .collect();
                actions.extend(
                    self.state.players[player.0]
                        .sideboard
                        .iter()
                        .filter(|card| self.eligible_lesson(*player, **card))
                        .map(|card| Action::LearnTakeLesson {
                            player: *player,
                            object: self.state.object_of(*card),
                        }),
                );
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Learn,
                    actions,
                    focus: source_focus,
                }
            }
        };
        Some(space)
    }

    /// Feed a player action into the pending mid-resolution decision.
    pub(crate) fn execute_decision_action(&mut self, action: &Action) -> Result<(), AgentError> {
        let Some(suspended) = &self.state.suspended_decision else {
            return Err(AgentError("no pending decision".to_string()));
        };

        if action.player() != suspended.decision.player() {
            return Err(AgentError("wrong player for pending decision".to_string()));
        }

        // Validate while the full continuation is still attached, so the
        // payment model sees every reserved source and remaining obligation.
        if let (
            Decision::ManaChoice {
                player, choices, ..
            },
            Action::ChooseMode { mode, .. },
        ) = (&suspended.decision, action)
        {
            let mana = choices
                .get(*mode)
                .ok_or_else(|| AgentError("illegal mana color".into()))?;
            if !self.mana_choice_completes_payment(*player, mana) {
                return Err(AgentError("mana color cannot complete the payment".into()));
            }
        }
        if let (
            Decision::ManaUpfront {
                player,
                tap_remaining: 0,
                ..
            },
            Action::PayCost { .. },
        ) = (&suspended.decision, action)
        {
            return self.pay_mana_upfront(*player);
        }
        if let (
            Decision::ManaUpfront { tap_remaining, .. },
            Action::SelectCard {
                player,
                object: object_id,
            },
        ) = (&suspended.decision, action)
        {
            if *tap_remaining > 0 {
                return self.pay_mana_tap_cost(*player, *object_id);
            }
        }
        if let (Decision::ManaSpend { .. }, Action::ChooseMode { mode, .. }) =
            (&suspended.decision, action)
        {
            return self.advance_mana_spend(Some(*mode));
        }
        let mut suspended = self.state.suspended_decision_mut().take().unwrap();
        match (&mut suspended.decision, action) {
            (Decision::EnterAsCopy { candidates, .. }, choice) => {
                let of = match choice {
                    Action::ChooseTarget {
                        target: Target::Permanent(id),
                        ..
                    } if candidates.contains(id) => Some(*id),
                    Action::Decline { .. } => None,
                    _ => {
                        *self.state.suspended_decision_mut() = Some(suspended);
                        return Err(AgentError("illegal creature to copy".into()));
                    }
                };
                let mut batch = suspended.frame.batch.take().expect("retained event batch");
                self.choose_entry_copy(&mut batch, of);
                suspended.frame.batch = Some(batch);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::AssignCombatDamage {
                    pass,
                    minimum,
                    remaining,
                    assigned,
                    ..
                },
                Action::ChooseMode { mode, .. },
            ) => {
                let Some(amount) = i32::try_from(*mode)
                    .ok()
                    .filter(|v| (*minimum..=*remaining).contains(v))
                else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal combat damage assignment".into()));
                };
                let mut amounts = assigned.clone();
                amounts.push(amount);
                self.assign_combat_damage(suspended.frame, *pass, amounts);
                Ok(())
            }
            (Decision::OrderReplacement { candidates, .. }, Action::ChooseMode { mode, .. }) => {
                let Some(candidate) = candidates.get(*mode) else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal replacement".into()));
                };
                self.choose_replacement(
                    suspended
                        .frame
                        .batch
                        .as_mut()
                        .expect("retained event batch"),
                    candidate,
                );
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::ChoosePaymentX { player, if_paid }, Action::ChooseMode { mode, .. }) => {
                let x = u8::try_from(*mode).ok();
                let Some(x) = x.filter(|x| {
                    self.can_pay_mana_cost(*player, &ManaCost::default().with_generic(*x))
                }) else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("unpayable X".into()));
                };
                suspended.frame.x = x;
                let payment = PaymentProgress::new(
                    *player,
                    CostComponent::mana(ManaCost::default().with_generic(x)),
                    std::mem::take(if_paid),
                    Vec::new(),
                );
                self.continue_resolution(suspended.frame, Some(Decision::Payment(payment)));
                Ok(())
            }
            (Decision::ChoosePaymentX { .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::SearchNamed {
                    owner,
                    name,
                    remaining,
                    ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                if *remaining == 0
                    || self.state[*object_id].owner != *owner
                    || self.state[*object_id].name != *name
                    || !matches!(
                        self.state.zones.zone_of(*object_id),
                        Some(ZoneType::Hand | ZoneType::Library | ZoneType::Graveyard)
                    )
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal named-search card".into()));
                }
                self.move_object(*object_id, ZoneType::Exile);
                *remaining -= 1;
                if *remaining == 0 {
                    self.shuffle_library(*owner);
                    self.run_frame(suspended.frame);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }
            (Decision::SearchNamed { owner, .. }, Action::Decline { .. }) => {
                self.shuffle_library(*owner);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Payment(payment), Action::SelectCard { object, .. })
                if matches!(payment.owed, CostComponent::Discard) =>
            {
                if let Err(error) =
                    self.pay_discard_cost(payment.player, None, *object, &mut payment.receipt)
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(error);
                }
                let Decision::Payment(payment) = suspended.decision else {
                    unreachable!()
                };
                self.finish_resolution_payment(payment, suspended.frame);
                Ok(())
            }
            (Decision::Payment(payment), Action::WaterbendTap { permanent, .. }) => {
                let tapped = match &mut payment.owed {
                    CostComponent::Mana { cost, help } => self.resolution_help_tap(
                        payment.player,
                        *permanent,
                        cost,
                        *help,
                        &mut payment.receipt,
                    ),
                    _ => Err(AgentError("no waterbend payment is pending".into())),
                };
                *self.state.suspended_decision_mut() = Some(suspended);
                tapped
            }
            (Decision::Payment(payment), Action::PayCost { .. }) => {
                let cost = match &payment.owed {
                    CostComponent::Life(life) => {
                        let (player, life) = (payment.player, *life);
                        if !self.can_pay_life(player, life) {
                            *self.state.suspended_decision_mut() = Some(suspended);
                            return Err(AgentError("cannot pay life cost".into()));
                        }
                        self.pay_life_cost(player, life, &mut payment.receipt);
                        let Decision::Payment(payment) = suspended.decision else {
                            unreachable!()
                        };
                        self.finish_resolution_payment(payment, suspended.frame);
                        return Ok(());
                    }
                    CostComponent::Mana { cost, .. } => cost.clone(),
                    _ => unreachable!("unexpected resolution payment component"),
                };
                if self.automatic_mana_payment(payment.player) && payment.can_decline() {
                    payment.committed = true;
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Ok(());
                }
                if let Err(error) = self.produce_mana(payment.player, &cost) {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(error);
                }
                let Decision::Payment(payment) = suspended.decision else {
                    unreachable!()
                };
                self.begin_mana_spend(
                    payment.player,
                    cost,
                    SpendCompletion::Resolution(payment),
                    suspended.frame,
                )
            }
            (Decision::ChooseObject { optional: true, .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Payment(payment), Action::Decline { .. }) if payment.can_decline() => {
                match payment.counters_on_decline {
                    Some(Target::StackSpell(object_id)) => {
                        self.counter_spell(object_id, suspended.frame.source(), ZoneType::Graveyard)
                    }
                    Some(Target::StackAbility(id)) => {
                        self.counter_ability(id, suspended.frame.source())
                    }
                    _ => {}
                }
                let effects = std::mem::take(&mut payment.if_declined);
                suspended.frame.prepend_effects(effects.into_iter());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::ManaChoice {
                    player,
                    choices,
                    production,
                    resume,
                },
                Action::ChooseMode { mode, .. },
            ) => {
                let mana = &choices[*mode];
                let player = *player;
                // A priority activation has no enclosing payment or suspended
                // caller. Each output choice also owns the fixed bonuses it adds.
                let before = (resume.is_none()
                    && self.pending_choice.is_none()
                    && self.mana_capture(player).is_some())
                .then(|| self.state.players[player.0].mana_pool.total());
                // Mana made for no nested cost pays toward an announced cost
                // being paid as it is produced.
                let start = resume
                    .is_none()
                    .then(|| self.state.players[player.0].mana_pool.unit_count());
                let origin = match production {
                    ManaProduction::Output { source, .. } => *source,
                    ManaProduction::Bonus { source, .. } => Some(*source),
                }
                .map(|source| self.mana_origin(source));
                self.state.mana_pool_mut(player).add(mana, origin);
                // The output is produced: its triggered mana abilities
                // trigger now (CR 106.12a) and resolve at once (CR 605.4a).
                let next = match production {
                    ManaProduction::Output {
                        source: Some(source),
                        tapped: Some(tapped),
                    } => {
                        let bonuses = self.production_bonuses(player, tapped);
                        self.resolve_mana_bonuses(player, *source, mana, bonuses, resume.take())
                    }
                    ManaProduction::Output { .. } => resume.take(),
                    ManaProduction::Bonus {
                        tapped,
                        produced,
                        pending,
                        ..
                    } => self.resolve_mana_bonuses(
                        player,
                        *tapped,
                        produced,
                        std::mem::take(pending),
                        resume.take(),
                    ),
                };
                if let Some(before) = before {
                    let produced = self.state.players[player.0].mana_pool.total() - before;
                    self.mana_capture(player).unwrap().float(produced);
                }
                if let Some(start) = start {
                    self.commit_produced_mana(player, start);
                }
                if let Some(next) = next {
                    suspended.decision = *next;
                    *self.state.suspended_decision_mut() = Some(suspended);
                    // Only an upfront payment advances here; a bonus choice
                    // stays suspended for the chooser.
                    self.advance_mana_upfront()?;
                } else {
                    // An activated mana ability retains its payer's priority;
                    // its empty rule frame is not a resolving stack object.
                    self.state.priority_mut().on_non_pass_action(player);
                    // The announced payment its output paid toward advances
                    // as a fixed tap's does.
                    if start.is_some()
                        && matches!(self.paying(), Some((payer, _, CostComponent::Mana { .. })) if payer == player)
                    {
                        let payment = self.pending_payment(player)?.clone();
                        return self.advance_payment(player, payment);
                    }
                }
                Ok(())
            }
            (
                Decision::ChooseObject { .. },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let selected = self
                    .choice_candidates(&suspended.decision)
                    .into_iter()
                    .find(|id| {
                        self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|p| p.object == *object_id)
                    })
                    .and_then(|id| self.permanent_object(id));
                let Some(selected) = selected else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal object selection".into()));
                };
                let Decision::ChooseObject { effect, .. } = suspended.decision else {
                    unreachable!()
                };
                // The instruction runs next, over the chosen permanent.
                suspended.frame.chosen = Some(selected);
                suspended.frame.queue.push_front(*effect);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::ChooseToExile {
                    player,
                    controller,
                    predicate,
                    chosen,
                },
                action @ (Action::SelectCard { .. } | Action::Decline { .. }),
            ) => {
                if let Action::SelectCard {
                    object: object_id, ..
                } = action
                {
                    let candidates = self.exile_choice_candidates(*controller, predicate, chosen);
                    let selected = self
                        .state
                        .permanent_of(*object_id)
                        .filter(|id| candidates.contains(id));
                    let Some(selected) = selected else {
                        *self.state.suspended_decision_mut() = Some(suspended);
                        return Err(AgentError("illegal permanent to choose".into()));
                    };
                    chosen.push(selected);
                }
                let next = self.next_exile_chooser(
                    *player,
                    *controller,
                    predicate.clone(),
                    std::mem::take(chosen),
                    &mut suspended.frame,
                );
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::KeepCreatures { .. },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let candidates = self.keep_candidates(&suspended.decision);
                let selected = self
                    .state
                    .permanent_of(*object_id)
                    .filter(|id| candidates.contains(id));
                let Some(selected) = selected else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal creature to keep".into()));
                };
                let Decision::KeepCreatures {
                    player,
                    max_total_power,
                    sacrifice,
                    kept,
                } = &mut suspended.decision
                else {
                    unreachable!()
                };
                kept.push(selected);
                let (player, max_total_power, sacrifice, kept) =
                    (*player, *max_total_power, *sacrifice, std::mem::take(kept));
                let next = self.keep_creatures(
                    player,
                    max_total_power,
                    sacrifice,
                    kept,
                    &mut suspended.frame,
                );
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::Retarget {
                    player,
                    object,
                    slot,
                    may_keep,
                },
                Action::ChooseTarget { target, .. },
            ) => {
                let (player, object, slot, may_keep) = (*player, *object, *slot, *may_keep);
                if !self
                    .retarget_candidates(&suspended.decision)
                    .contains(target)
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal new target".into()));
                }
                self.change_stack_target(object, slot, *target);
                let next = self
                    .next_retarget(player, object, slot + 1, may_keep)
                    .filter(|_| may_keep);
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::Retarget {
                    player,
                    object,
                    slot,
                    may_keep: true,
                },
                Action::Decline { .. },
            ) => {
                let next = self.next_retarget(*player, *object, *slot + 1, true);
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::KeepCreatures {
                    player,
                    max_total_power,
                    sacrifice: None,
                    kept,
                },
                Action::Decline { .. },
            ) => {
                let kept = std::mem::take(kept);
                let next = self.stop_keeping_creatures(
                    *player,
                    *max_total_power,
                    None,
                    kept,
                    &mut suspended.frame,
                );
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::DistributeCounters {
                    player,
                    kind,
                    remaining,
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let selected = self
                    .state
                    .permanent_of(*object_id)
                    .filter(|id| self.tapped_opponent_creatures(*player).contains(id));
                let Some(selected) = selected else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("illegal counter recipient".into()));
                };
                let batch = self.event_batch(
                    EventBinding::None,
                    [crate::flow::proposed_event::ProposedEvent::CounterChange {
                        provenance: crate::flow::proposed_event::CounterProvenance::new(
                            crate::flow::proposed_event::EventCause::Effect,
                        ),
                        target: self.permanent_object(selected).expect("chosen permanent"),
                        kind: *kind,
                        delta: 1,
                        by: Some(*player),
                    }],
                );
                let next = self.distribute_counters(*player, *kind, *remaining - 1);
                suspended.frame.batch = Some(batch);
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::Discard { player, remaining },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                if !self
                    .state
                    .zones
                    .contains(*object_id, ZoneType::Hand, *player)
                    || self.state.card_of(*object_id).is_none()
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("discard must be from hand".into()));
                }
                let discarded = self.move_object(*object_id, ZoneType::Graveyard);
                if self.state.live(discarded).is_some() {
                    suspended.frame.this_way.push(discarded);
                }
                *remaining -= 1;
                if *remaining == 0
                    || self
                        .state
                        .zone_cards(ZoneType::Hand, *player)
                        .next()
                        .is_none()
                {
                    self.run_frame(suspended.frame);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }

            (
                Decision::Scry {
                    player,
                    looked: _,
                    remaining,
                    top,
                    bottom,
                },
                Action::ScryCard {
                    object: object_id,
                    to_bottom,
                    ..
                },
            ) => {
                let Some(index) = remaining
                    .iter()
                    .position(|candidate| candidate == object_id)
                else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("card is not in the pending scry".into()));
                };
                remaining.remove(index);
                if *to_bottom {
                    bottom.push(*object_id);
                } else {
                    top.push(*object_id);
                }
                if remaining.is_empty() {
                    for object in bottom.iter().rev() {
                        self.reposition_library_object(*player, *object, 0);
                    }
                    self.put_on_top_of_library(*player, top);
                    self.remember_library_objects(*player, bottom);
                    self.run_frame(suspended.frame);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }
            (
                Decision::LookAndSelect {
                    player,
                    looked,
                    predicate,
                    selected,
                    reveal,
                    to_battlefield,
                    random_order,
                    max_select,
                    ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let selectable = looked.contains(object_id)
                    && self.object_matches_predicate(*object_id, predicate);
                if !selectable {
                    let err = AgentError("card is not selectable".to_string());
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(err);
                }
                let player = *player;
                looked.retain(|c| c != object_id);
                *selected += 1;
                suspended.frame.batch = if *to_battlefield {
                    Some(self.event_batch(
                        EventBinding::None,
                        [self.zone_move_proposal(
                            *object_id,
                            ZoneType::Battlefield,
                            Default::default(),
                        )],
                    ))
                } else {
                    if *reveal {
                        self.reveal_object_to_hand(*object_id, player, ZoneEntry::default());
                    } else {
                        self.move_object(*object_id, ZoneType::Hand);
                    }
                    None
                };
                self.invalidate_mana_cache(player);

                let more_selectable = looked
                    .iter()
                    .any(|c| self.object_matches_predicate(*c, predicate));
                let next = if *selected >= *max_select || !more_selectable {
                    self.library_order_decision(
                        player,
                        std::mem::take(looked),
                        LibraryEnd::Bottom,
                        *random_order,
                    )
                } else {
                    Some(suspended.decision)
                };
                self.continue_resolution(suspended.frame, next);
                Ok(())
            }
            (
                Decision::LookAndSelect {
                    player,
                    random_order,
                    looked,
                    selected,
                    min_select,
                    predicate,
                    ..
                },
                Action::Decline { .. },
            ) => {
                let any_selectable = looked
                    .iter()
                    .any(|c| self.object_matches_predicate(*c, predicate));
                if *selected < *min_select && any_selectable {
                    let err = AgentError("selection is not optional yet".to_string());
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(err);
                }
                let rest = std::mem::take(looked);
                let decision =
                    self.library_order_decision(*player, rest, LibraryEnd::Bottom, *random_order);
                self.continue_resolution(suspended.frame, decision);
                Ok(())
            }
            (
                Decision::OrderLibrary {
                    player,
                    object_ids,
                    next,
                    end,
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let Some(offset) = object_ids[*next..]
                    .iter()
                    .position(|candidate| candidate == object_id)
                else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError(
                        "card is not in the pending library order".into(),
                    ));
                };
                // Move the chosen card into the prefix without disturbing the
                // relative order of the remaining offers.
                object_ids[*next..=*next + offset].rotate_right(1);
                *next += 1;
                match end {
                    LibraryEnd::Bottom => self.put_on_bottom_of_library(*object_id),
                    LibraryEnd::Top => self.put_on_top_of_library(*player, object_ids),
                }
                self.remember_library_objects(*player, &object_ids[..*next]);
                if *next == object_ids.len() {
                    self.run_frame(suspended.frame);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }
            (
                Decision::SearchDistinctNames { .. },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                if !self
                    .distinct_name_search_candidates(&suspended.decision)
                    .iter()
                    .any(|card| self.state.object_of(*card) == *object_id)
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not a distinct-name search candidate".into()));
                }
                let Decision::SearchDistinctNames {
                    remaining,
                    selected,
                    ..
                } = &mut suspended.decision
                else {
                    unreachable!()
                };
                selected.push(*object_id);
                *remaining -= 1;
                if *remaining > 0 {
                    *self.state.suspended_decision_mut() = Some(suspended);
                } else {
                    self.finish_distinct_name_search(suspended);
                }
                Ok(())
            }
            (
                Decision::DiscardFromRevealedHand {
                    owner, predicate, ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                if !self
                    .state
                    .zones
                    .contains(*object_id, ZoneType::Hand, *owner)
                    || self.state.card_of(*object_id).is_none()
                    || !self.object_matches_predicate(*object_id, predicate)
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError(
                        "not a matching card in the revealed hand".into(),
                    ));
                }
                self.move_object(*object_id, ZoneType::Graveyard);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::CastDuringResolution { .. },
                Action::CastSpell {
                    object: object_id,
                    profile: SpellProfile::Adventure | SpellProfile::Omen | SpellProfile::BackFace,
                    route: None,
                    ..
                }
                | Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let profile = action.spell_profile();
                if !self
                    .resolution_cast_candidates(&suspended.decision)
                    .iter()
                    .any(|(card, inset)| {
                        self.state.object_of(*card) == *object_id && *inset == profile
                    })
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not a castable offered card".into()));
                }
                if let Decision::CastDuringResolution {
                    spends: Some(key), ..
                } = suspended.decision
                {
                    self.spend_once_each_turn(key);
                }
                let (player, object_id) = (suspended.decision.player(), *object_id);
                let cast = suspended
                    .decision
                    .cast_authority()
                    .and_then(|authority| self.cast_choices(player, object_id, profile, authority))
                    .expect("cast decision");
                if let Decision::CastDuringResolution {
                    offer: ResolutionCastOffer::AmongExiled { objects },
                    ..
                } = &mut suspended.decision
                {
                    objects.retain(|object| *object != object_id);
                }
                *self.state.cast_caller_mut() = Some(Box::new(suspended));
                self.begin_cast(player, object_id, cast)
            }
            (
                Decision::ExileLinked {
                    grant,
                    grantee,
                    source,
                    ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let (grant, grantee, source) = (*grant, *grantee, *source);
                if !self
                    .linked_exile_candidates(&suspended.decision)
                    .iter()
                    .any(|card| self.state.object_of(*card) == *object_id)
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not an offered card to exile".into()));
                }
                self.exile_linked(*object_id, grantee, source, grant);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::ExileLinked { up_to: true, .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::CastDuringResolution { player, offer, .. }, Action::Decline { .. }) => {
                match offer {
                    ResolutionCastOffer::Exiled {
                        object: object_id, ..
                    } => {
                        self.reveal_object_to_hand(*object_id, *player, ZoneEntry::default());
                    }
                    ResolutionCastOffer::Cascade { exiled, .. } => {
                        self.put_exiled_on_bottom_in_random_order(exiled.clone());
                    }
                    _ => {}
                }
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::SearchDistinctNames { .. }, Action::Decline { .. }) => {
                self.finish_distinct_name_search(suspended);
                Ok(())
            }
            (
                Decision::OpponentSplitsRevealed {
                    owner,
                    revealed,
                    remaining,
                    ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let Some(index) = revealed.iter().position(|c| c == object_id) else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not a revealed card".into()));
                };
                revealed.remove(index);
                self.move_object(*object_id, ZoneType::Graveyard);
                *remaining -= 1;
                if *remaining > 0 && !revealed.is_empty() {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Ok(());
                }
                let owner = *owner;
                let events = std::mem::take(revealed).into_iter().map(|object_id| {
                    self.zone_move_proposal(
                        object_id,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::ZoneEntry {
                            tapped: true,
                            ..Default::default()
                        },
                    )
                });
                suspended.frame.batch =
                    Some(self.event_batch(EventBinding::Shuffle { player: owner }, events));
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::Surveil {
                    looked, graveyard, ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) if looked.contains(object_id) && !graveyard.contains(object_id) => {
                graveyard.push(*object_id);
                if graveyard.len() == looked.len() {
                    self.finish_surveil_selection(suspended);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }
            (Decision::Surveil { .. }, Action::Decline { .. }) => {
                self.finish_surveil_selection(suspended);
                Ok(())
            }
            (
                Decision::SelectCards {
                    candidates,
                    selected,
                    min,
                    max,
                    ..
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                let choice = (self.state.live(*object_id).is_some()
                    && candidates.contains(object_id)
                    && !selected.contains(object_id))
                .then_some(*object_id);
                let Some(choice) = choice.filter(|_| selected.len() < *max) else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not an available selection candidate".into()));
                };
                selected.push(choice);
                if selected.len() == *max || selected.len() == candidates.len() {
                    debug_assert!(selected.len() >= *min);
                    self.finish_card_selection(suspended);
                } else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                }
                Ok(())
            }
            (Decision::SelectCards { selected, min, .. }, Action::Decline { .. })
                if selected.len() >= *min =>
            {
                self.finish_card_selection(suspended);
                Ok(())
            }
            (
                Decision::NameCard {
                    count,
                    names,
                    scope,
                    ..
                },
                Action::ChooseMode { mode, .. },
            ) => {
                let candidate = match scope.option(*mode) {
                    Some(NameOption::Group(child)) => {
                        // Group navigation narrows the pending prompt only.
                        *scope = child;
                        *self.state.suspended_decision_mut() = Some(suspended);
                        return Ok(());
                    }
                    Some(NameOption::Name(row)) => names.get(row),
                    None => None,
                };
                let Some(candidate) = candidate else {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("name option out of range".into()));
                };
                suspended.frame.queue.push_front(Effect::ExileNamed {
                    name: candidate.name.clone(),
                    count: *count,
                });
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Modal { modes, .. }, Action::ChooseMode { mode, .. }) => {
                if *mode >= modes.len() {
                    let err = AgentError("modal mode out of range".to_string());
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(err);
                }
                let effects = std::mem::take(&mut modes[*mode]);
                if effects.is_empty() {
                    // Stopping an "up to" counter choice completes its
                    // instruction; the next verb must choose independently.
                    suspended.frame.chosen = None;
                }
                suspended.frame.prepend_effects(effects.into_iter());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::LegendRule {
                    player,
                    groups,
                    keep,
                },
                Action::SelectCard {
                    object: object_id, ..
                },
            ) => {
                if !groups[0].contains(object_id) {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError("not a legend in this group".into()));
                }
                keep.push(*object_id);
                groups.remove(0);
                if let Some(group) = groups.first() {
                    let id = self
                        .state
                        .permanent_of(group[0])
                        .expect("legend on battlefield");
                    *player = self.state.permanents[id]
                        .as_ref()
                        .expect("legend")
                        .controller;
                    *self.state.suspended_decision_mut() = Some(suspended);
                } else {
                    let completion = suspended.frame.completion;
                    self.perform_state_based_actions_with_legends(keep);
                    if let Some(pending) = self.state.suspended_decision_mut().as_mut() {
                        pending.frame.completion.then(completion);
                    } else {
                        self.run_frame(EffectFrame {
                            completion,
                            ..EffectFrame::new(*player, Vec::new())
                        });
                    }
                }
                Ok(())
            }
            (
                Decision::Learn { player },
                Action::LearnDiscard {
                    object: object_id, ..
                },
            ) => {
                let player = *player;
                if !self
                    .state
                    .zones
                    .contains(*object_id, ZoneType::Hand, player)
                    || self.state.card_of(*object_id).is_none()
                {
                    let err = AgentError("card to discard is not in hand".to_string());
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(err);
                }
                self.move_object(*object_id, ZoneType::Graveyard);
                self.draw_cards(player, 1);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::Learn { player },
                Action::LearnTakeLesson {
                    object: object_id, ..
                },
            ) => {
                if !self
                    .state
                    .card_id(*object_id)
                    .is_some_and(|card| self.eligible_lesson(*player, card))
                {
                    *self.state.suspended_decision_mut() = Some(suspended);
                    return Err(AgentError(
                        "card is not an available owned sideboard Lesson".to_string(),
                    ));
                }
                self.reveal_object_to_hand(*object_id, *player, ZoneEntry::default());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Learn { .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            _ => {
                let err = AgentError("action does not match pending decision".to_string());
                *self.state.suspended_decision_mut() = Some(suspended);
                Err(err)
            }
        }
    }

    fn eligible_lesson(&self, player: PlayerId, card: CardId) -> bool {
        self.decision_controller(player) == player
            && self.state.players[player.0].sideboard.contains(&card)
            && self.state.cards[card].owner == player
            && self
                .state
                .zones
                .zone_of(self.state.object_of(card))
                .is_none()
            && self
                .state
                .objects
                .retained_base(self.state.object_of(card))
                .expect("sideboard printing")
                .1
                .subtypes
                .iter()
                .any(|subtype| subtype == "Lesson")
    }

    /// The top `count` objects of `player`'s library, including transient visitors.
    pub(crate) fn library_top_objects(&self, player: PlayerId, count: usize) -> Vec<ObjectId> {
        self.state
            .zones
            .zone_objects(ZoneType::Library, player)
            .iter()
            .rev()
            .take(count)
            .copied()
            .collect()
    }

    /// The top card of `player`'s library.
    pub(crate) fn library_top_card(&self, player: PlayerId) -> Option<CardId> {
        self.state
            .zones
            .top(ZoneType::Library, player)
            .and_then(|object| self.state.card_of(object))
    }

    /// The cards an `ExileLinked` decision may exile: matching graveyard
    /// cards, or the cards of matching nontoken permanents `owner` controls.
    pub(crate) fn linked_exile_candidates(&self, decision: &Decision) -> Vec<CardId> {
        let Decision::ExileLinked {
            owner,
            zone,
            predicate,
            ..
        } = decision
        else {
            return Vec::new();
        };
        match zone {
            ZoneType::Battlefield => self
                .battlefield_permanents(*owner)
                .into_iter()
                .filter(|id| self.permanent_matches_predicate(*id, predicate))
                .filter_map(|id| self.state.permanents[id].as_ref().map(|p| p.object))
                .filter_map(|object| self.state.card_of(object))
                .collect(),
            _ => self
                .state
                .zone_cards(ZoneType::Graveyard, *owner)
                .filter(|card| {
                    self.object_matches_predicate(self.state.object_of(*card), predicate)
                })
                .collect(),
        }
    }

    /// Playable card and profile pairs. Graveyard choices bind each public
    /// object. Existing free-hand offers group interchangeable copies by definition.
    pub(crate) fn resolution_cast_candidates(
        &self,
        decision: &Decision,
    ) -> Vec<(CardId, SpellProfile)> {
        let Decision::CastDuringResolution {
            player,
            offer,
            paid,
            ..
        } = decision
        else {
            return Vec::new();
        };
        // Casting from exile or a library is casting from outside the hand
        // (Drannith Magistrate).
        if !matches!(offer, ResolutionCastOffer::Hand { .. }) && self.casts_only_from_hand(*player)
        {
            return Vec::new();
        }
        let mut cards: Vec<CardId> = match offer {
            ResolutionCastOffer::Graveyard { .. } => self
                .state
                .zone_cards(ZoneType::Graveyard, *player)
                .collect(),
            ResolutionCastOffer::Hand { .. } => {
                self.state.zone_cards(ZoneType::Hand, *player).collect()
            }
            ResolutionCastOffer::AmongExiled { objects } => objects
                .iter()
                .filter(|object| self.state.zones.zone_of(**object) == Some(ZoneType::Exile))
                .filter_map(|object| self.state.card_of(*object))
                .collect(),
            ResolutionCastOffer::Exiled { object, .. }
            | ResolutionCastOffer::Cascade { object, .. } => self
                .state
                .card_of(*object)
                .filter(|_| self.state.zones.contains(*object, ZoneType::Exile, *player))
                .into_iter()
                .collect(),
            ResolutionCastOffer::ExiledWith { source, .. } => self
                .state
                .exile_links
                .iter()
                .filter(|link| {
                    link.source == *source
                        && self
                            .state
                            .zones
                            .contains(link.exiled_object, ZoneType::Exile, *player)
                })
                .filter_map(|link| self.state.card_of(link.exiled_object))
                .collect(),
            ResolutionCastOffer::LibraryTop { object } => self
                .library_top_card(*player)
                .filter(|card| self.state.object_of(*card) == *object)
                .into_iter()
                .collect(),
        };

        cards.sort_unstable_by_key(|card| {
            (self.state[self.state.object_of(*card)].definition_id, *card)
        });
        let mut candidates = Vec::new();
        for card in cards {
            let object = self.state.object_of(card);
            let definition = &self.state[object];
            for profile in definition.spell_profiles() {
                let cast = self
                    .cast_choices(
                        *player,
                        object,
                        profile,
                        CastAuthority::Resolution { offer, paid: *paid },
                    )
                    .expect("an offered card's own profile");
                let spell = cast.spell(definition);
                if !spell.types.is_castable() {
                    continue;
                }
                let eligible = match offer {
                    ResolutionCastOffer::Graveyard { predicate, .. }
                    | ResolutionCastOffer::Hand { predicate }
                    | ResolutionCastOffer::ExiledWith { predicate, .. } => {
                        cast.matches_spell(self, object, predicate)
                    }
                    ResolutionCastOffer::Exiled { below, .. } => {
                        i32::from(
                            spell
                                .mana_cost
                                .as_ref()
                                .map_or(0, |cost| cost.mana_value_at(0)),
                        ) < *below
                    }
                    ResolutionCastOffer::Cascade { below, .. } => {
                        spell
                            .mana_cost
                            .as_ref()
                            .map_or(0, |cost| cost.mana_value_at(0))
                            < *below
                    }
                    ResolutionCastOffer::AmongExiled { .. }
                    | ResolutionCastOffer::LibraryTop { .. } => true,
                };
                if eligible && self.castable_mana_value(*player, object, cast).is_some() {
                    candidates.push((card, profile));
                }
            }
        }
        if !matches!(
            offer,
            ResolutionCastOffer::Graveyard { .. } | ResolutionCastOffer::AmongExiled { .. }
        ) {
            let mut offered = HashSet::new();
            candidates.retain(|(card, profile)| {
                offered.insert((
                    self.state[self.state.object_of(*card)].definition_id,
                    *profile,
                ))
            });
        }
        candidates
    }

    /// One hand card per matching definition for `DiscardFromRevealedHand`.
    pub(crate) fn revealed_hand_objects(&self, decision: &Decision) -> Vec<ObjectId> {
        let Decision::DiscardFromRevealedHand {
            owner, predicate, ..
        } = decision
        else {
            return Vec::new();
        };
        let mut candidates: Vec<ObjectId> = self
            .state
            .zones
            .zone_objects(ZoneType::Hand, *owner)
            .iter()
            .copied()
            .filter(|object| self.object_matches_predicate(*object, predicate))
            .collect();
        candidates.sort_unstable_by_key(|object| (self.state[*object].definition_id, *object));
        candidates.dedup_by_key(|object| self.state[*object].definition_id);
        candidates
    }

    pub(crate) fn revealed_hand_candidates(&self, decision: &Decision) -> Vec<CardId> {
        self.revealed_hand_objects(decision)
            .into_iter()
            .filter_map(|object| self.state.card_of(object))
            .collect()
    }

    /// Library cards matching a `SearchDistinctNames` predicate whose names
    /// differ from every card it already selected.
    pub(crate) fn distinct_name_search_objects(&self, decision: &Decision) -> Vec<ObjectId> {
        let Decision::SearchDistinctNames {
            player,
            predicate,
            selected,
            ..
        } = decision
        else {
            return Vec::new();
        };
        let mut candidates: Vec<ObjectId> = self
            .state
            .zones
            .zone_objects(ZoneType::Library, *player)
            .iter()
            .copied()
            .filter(|object| {
                let card_ref = &self.state[*object];
                self.object_matches_predicate(*object, predicate)
                    && !selected
                        .iter()
                        .any(|chosen| self.state[*chosen].name == card_ref.name)
            })
            .collect();
        candidates.sort_unstable_by_key(|object| (self.state[*object].definition_id, *object));
        candidates
    }

    pub(crate) fn distinct_name_search_candidates(&self, decision: &Decision) -> Vec<CardId> {
        self.distinct_name_search_objects(decision)
            .into_iter()
            .filter_map(|object| self.state.card_of(object))
            .collect()
    }

    /// The revealed cards go to `end`: into hand, or to an opponent's split.
    /// With none revealed, or once in hand, the library is shuffled.
    fn finish_distinct_name_search(&mut self, mut suspended: SuspendedResolution) {
        let Decision::SearchDistinctNames {
            player,
            selected,
            end,
            ..
        } = &mut suspended.decision
        else {
            unreachable!()
        };
        let player = *player;
        match *end {
            DistinctSearchEnd::OpponentSplits { picks } if !selected.is_empty() => {
                suspended.decision = Decision::OpponentSplitsRevealed {
                    player: PlayerId(1 - player.0),
                    owner: player,
                    revealed: std::mem::take(selected),
                    remaining: picks,
                };
                *self.state.suspended_decision_mut() = Some(suspended);
            }
            _ => {
                for object in std::mem::take(selected) {
                    self.reveal_object_to_hand(object, player, ZoneEntry::default());
                }
                self.shuffle_library(player);
                self.run_frame(suspended.frame);
            }
        }
    }

    /// Random placement and owner-chosen placement share the same library
    /// move. A singleton is forced and needs no pending choice.
    fn library_order_decision(
        &mut self,
        player: PlayerId,
        mut objects: Vec<ObjectId>,
        end: LibraryEnd,
        random_order: bool,
    ) -> Option<Decision> {
        if !random_order && objects.len() > 1 {
            return Some(Decision::OrderLibrary {
                player,
                object_ids: objects,
                next: 0,
                end,
            });
        }
        if random_order {
            objects.shuffle(self.state.rng_mut());
        }
        match end {
            LibraryEnd::Bottom => {
                for object in &objects {
                    self.put_on_bottom_of_library(*object);
                }
                if !random_order {
                    self.remember_library_objects(player, &objects);
                }
            }
            LibraryEnd::Top => self.put_on_top_of_library(player, &objects),
        }
        if random_order {
            self.state.known_library_mut(player).forget_order();
        }
        None
    }

    fn finish_card_selection(&mut self, suspended: SuspendedResolution) {
        let Decision::SelectCards {
            selected,
            from,
            effects,
            if_declined,
            ..
        } = suspended.decision
        else {
            unreachable!("card selection");
        };
        let effects = if selected.is_empty() {
            if_declined
        } else {
            effects
        };
        let mut frame = suspended.frame;
        frame.select_scope(selected, from, effects);
        self.run_frame(frame);
    }

    /// Commit the chosen graveyard cards as one batch, then order the kept
    /// cards. The frame's completion announces Surveilled only after ordering.
    fn finish_surveil_selection(&mut self, mut suspended: SuspendedResolution) {
        let Decision::Surveil {
            player,
            looked,
            graveyard,
        } = &mut suspended.decision
        else {
            unreachable!("surveil selection");
        };
        // A private selection can change any position in the inspected prefix.
        // The public suffix still shifts predictably with each actual departure.
        if !graveyard.is_empty() {
            self.state
                .known_library_mut(*player)
                .public
                .retain(|position, _| *position >= looked.len());
        }
        let events = graveyard.iter().map(|object| {
            self.zone_move_proposal(
                *object,
                ZoneType::Graveyard,
                ZoneEntry {
                    inspected_library_departure: true,
                    ..Default::default()
                },
            )
        });
        let batch = self.event_batch(EventBinding::None, events);
        looked.retain(|object| !graveyard.contains(object));
        // A single kept card is already in order. The batch shifts its known
        // position when the selected cards actually leave the library.
        let next = (looked.len() > 1).then(|| Decision::OrderLibrary {
            player: *player,
            object_ids: std::mem::take(looked),
            next: 0,
            end: LibraryEnd::Top,
        });
        suspended.frame.batch = Some(batch);
        self.continue_resolution(suspended.frame, next);
    }

    /// Reorder inspected library cards, first to draw first. Their exact
    /// objects survive: placement within a zone emits no zone-change events.
    fn put_on_top_of_library(&mut self, player: PlayerId, objects: &[ObjectId]) {
        let knowledge = self.state.known_library_mut(player);
        knowledge
            .public
            .retain(|position, _| *position >= objects.len());
        for object in objects.iter().rev() {
            self.state
                .zones
                .reposition(ZoneType::Library, player, *object, usize::MAX);
        }
        self.remember_library_objects(player, objects);
        self.assert_library_knowledge();
    }

    /// Put each of `cards` still in exile on the bottom of its owner's
    /// library, in a random order (cascade, CR 702.85a).
    pub(crate) fn put_exiled_on_bottom_in_random_order(&mut self, mut objects: Vec<ObjectId>) {
        objects.retain(|object| self.state.zones.zone_of(*object) == Some(ZoneType::Exile));
        objects.shuffle(self.state.rng_mut());
        for object in objects {
            let object = self.move_object(object, ZoneType::Library);
            self.put_on_bottom_of_library(object);
        }
    }

    /// Move a library card to the bottom of its owner's library. Not a zone
    /// change (CR 701.26b-style reorder), so no events fire.
    pub(crate) fn put_on_bottom_of_library(&mut self, object: ObjectId) {
        self.reorder_in_library(object, LibraryEnd::Bottom);
    }

    /// Put a card already in its owner's library at one end: a reorder
    /// within one zone, so no zone change (CR 400.7) and no zone-move
    /// inverse applies. Index 0 is the bottom; the last card is the top.
    pub(crate) fn reorder_in_library(&mut self, object: ObjectId, end: LibraryEnd) {
        let owner = self.state[object].owner;
        let index = match end {
            LibraryEnd::Bottom => 0,
            LibraryEnd::Top => usize::MAX,
        };
        self.reposition_library_object(owner, object, index);
    }

    /// A card the resolution put on top of its owner's library: the
    /// resolving player saw it, and a revealed one (Campus Guide) is public.
    pub(crate) fn remember_placed_top(&mut self, object: ObjectId, frame: &EffectFrame) {
        let owner = self.state[object].owner;
        let definition = self.state[object].definition_id;
        let public = frame.selected_revealed && frame.selected.contains(&object);
        let knowledge = self.state.known_library_mut(owner);
        if public {
            knowledge.public.insert(0, definition);
            LibraryKnowledge::bound(&mut knowledge.public);
        }
        if public || frame.controller == owner {
            knowledge.own.insert(0, definition);
            LibraryKnowledge::bound(&mut knowledge.own);
        }
        self.assert_library_knowledge();
    }

    /// Reveal cards from the top of `player`'s library until one matches
    /// `predicate`; it goes to their hand and the cards revealed before it to
    /// the bottom in a random order. The cards never leave the library while
    /// revealed, so the match leaves a public position and a single passed
    /// card keeps every fact below it and is publicly known on the bottom.
    pub(crate) fn reveal_until_to_hand(&mut self, player: PlayerId, predicate: &CardPredicate) {
        let from_top: Vec<ObjectId> = self.library_from_top(player).collect();
        let hit = from_top
            .iter()
            .position(|object| self.object_matches_predicate(*object, predicate));
        let passed = from_top[..hit.unwrap_or(from_top.len())].to_vec();
        for object in &passed {
            self.emit(GameEvent::CardRevealed {
                owner: player,
                definition: self.state[*object].definition_id,
            });
        }
        if let Some(index) = hit {
            self.reveal_object_to_hand(
                from_top[index],
                player,
                ZoneEntry {
                    inspected_library_departure: true,
                    ..ZoneEntry::default()
                },
            );
        }
        match passed.as_slice() {
            [] => {}
            [top] => {
                let definition = self.state[*top].definition_id;
                let knowledge = self.state.known_library_mut(player);
                for view in [&mut knowledge.public, &mut knowledge.own] {
                    view.insert(0, definition);
                    LibraryKnowledge::bound(view);
                }
                self.place_entrant_on_bottom(*top);
            }
            _ => {
                let mut rest = passed;
                rest.shuffle(self.state.rng_mut());
                for object in rest {
                    self.reposition_library_object(player, object, 0);
                }
            }
        }
    }

    /// Put the top card of its owner's library on the bottom ("on the bottom
    /// of its owner's library"), as a card that has just entered on top. Its
    /// position is public, so this is no private permutation: facts about the
    /// cards above it keep their places from the top.
    pub(crate) fn place_entrant_on_bottom(&mut self, object: ObjectId) {
        let owner = self.state[object].owner;
        let bottom = self.state.zones.size(ZoneType::Library, owner) - 1;
        self.state.known_library_mut(owner).move_top_to(bottom);
        self.state
            .zones
            .reposition(ZoneType::Library, owner, object, 0);
        self.assert_library_knowledge();
    }

    /// Arbitrary reorders invalidate positional facts at the mutation boundary.
    pub(crate) fn reposition_library_object(
        &mut self,
        owner: PlayerId,
        object: ObjectId,
        index: usize,
    ) {
        self.state.library_order_pins_mut(owner).insert(object);
        self.state.known_library_mut(owner).forget_order();
        self.state
            .zones
            .reposition(ZoneType::Library, owner, object, index);
    }
}

#[cfg(test)]
mod learn_tests {
    use super::*;
    use crate::{decision::Command, search_state::witness, state::player::PlayerConfig};
    use std::collections::BTreeMap;

    fn counts(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
        entries
            .iter()
            .map(|(name, count)| (name.to_string(), *count))
            .collect()
    }

    fn game(sideboard: &[(&str, usize)]) -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new("learner", counts(&[("Island", 40)]))
                    .with_sideboard(counts(sideboard)),
                PlayerConfig::new("opponent", counts(&[("Plains", 40)]))
                    .with_sideboard(counts(&[("Fancy Footwork", 1)])),
            ],
            17,
            false,
        );
        game.keep_opening_hands();
        game
    }

    #[test]
    fn card_choices_exclude_transient_tokens_without_removing_them() {
        let mut game = game(&[]);
        let player = PlayerId(0);
        let permanent = game.create_token("token.spirit", player, false).unwrap();
        let token = game.state.permanents[permanent].as_ref().unwrap().object;
        let token = game.move_object(token, ZoneType::Hand);
        learn(&mut game);
        assert!(game.action_space().unwrap().actions.iter().all(|action| {
            !matches!(action, Action::LearnDiscard { object, .. } if *object == token)
        }));
        assert!(game
            .execute_decision_action(&Action::LearnDiscard {
                player,
                object: token
            })
            .is_err());
        assert!(game.state.zones.contains(token, ZoneType::Hand, player));
        assert!(game.state.suspended_decision.is_some());
        let token = game.move_object(token, ZoneType::Graveyard);
        let card = game.state.players[player.0].deck[0];
        game.move_object(game.state.object_of(card), ZoneType::Graveyard);
        let spec = crate::state::ability::TargetSpec::GraveyardCard {
            predicate: CardPredicate::default(),
        };
        let token_target = Target::GraveyardCard(game.state.live(token).unwrap());
        assert!(!game.test_target_is_legal(token_target, &spec, player));
        assert_eq!(
            game.legal_targets_for_spec(&spec, player),
            vec![Target::GraveyardCard(
                game.state.live(game.state.object_of(card)).unwrap()
            ),]
        );
    }

    fn learn(game: &mut Game) {
        game.run_frame(EffectFrame::new(PlayerId(0), vec![Effect::Learn]));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
    }

    fn take(game: &mut Game, object: ObjectId) {
        game.execute_decision_action(&Action::LearnTakeLesson {
            player: PlayerId(0),
            object,
        })
        .unwrap();
    }

    #[test]
    fn learn_retrieves_each_supported_lesson_without_drawing_or_allocating() {
        for name in [
            "Firebending Lesson",
            "It'll Quench Ya!",
            "Accumulate Wisdom",
            "Yip Yip!",
            "Fancy Footwork",
        ] {
            let mut game = game(&[(name, 1)]);
            let card = game.state.players[0].sideboard[0];
            assert!(game.state.live(game.state.object_of(card)).is_none());
            let deck = game.state.players[0].deck.clone();
            assert!(!deck.contains(&card));
            learn(&mut game);
            let before = witness(&game);
            let library = game
                .state
                .zones
                .zone_objects(ZoneType::Library, PlayerId(0))
                .to_vec();
            let hand_count = game.state.zones.size(ZoneType::Hand, PlayerId(0));
            let card_count = game.state.objects.len();
            let event_count = game.state.events.len();
            let lesson = game.state.object_of(card);
            take(&mut game, lesson);
            assert_eq!(game.state.objects.len(), card_count);
            assert_eq!(
                game.state.zones.size(ZoneType::Hand, PlayerId(0)),
                hand_count + 1
            );
            assert!(game.state.zones.contains(
                game.state.object_of(card),
                ZoneType::Hand,
                PlayerId(0)
            ));
            assert_eq!(game.state[game.state.object_of(card)].name, name);
            assert!(game.state.live(game.state.object_of(card)).is_some());
            assert_eq!(game.state.players[0].deck, deck);
            assert_eq!(
                game.state
                    .zones
                    .zone_objects(ZoneType::Library, PlayerId(0))
                    .to_vec(),
                library
            );
            assert_eq!(
                witness(&game).diagnostics.rng_probe,
                before.diagnostics.rng_probe
            );
            assert!(game.state.suspended_decision.is_none());
            let events: Vec<_> = game.state.events.iter().skip(event_count).collect();
            assert!(events.iter().any(|event| matches!(event, GameEvent::CardMoved { object_id: moved, from: None, to: ZoneType::Hand, .. } if *moved == game.state.object_of(card))));
            assert!(!events
                .iter()
                .any(|event| matches!(event, GameEvent::CardDrawn { .. })));
        }
    }

    #[test]
    fn learn_consumes_each_physical_copy_once_in_every_destination() {
        for destination in [ZoneType::Hand, ZoneType::Graveyard, ZoneType::Exile] {
            let mut game = game(&[("Firebending Lesson", 2)]);
            let roster = game.state.players[0].sideboard.clone();
            learn(&mut game);
            let lesson = game.state.object_of(roster[0]);
            take(&mut game, lesson);
            if destination != ZoneType::Hand {
                game.move_object(game.state.object_of(roster[0]), destination);
            }
            learn(&mut game);
            let before = witness(&game);
            assert!(game
                .execute_decision_action(&Action::LearnTakeLesson {
                    player: PlayerId(0),
                    object: game.state.object_of(roster[0])
                })
                .is_err());
            assert_eq!(witness(&game), before);
            let lesson = game.state.object_of(roster[1]);
            take(&mut game, lesson);
            learn(&mut game);
            assert!(!game
                .action_space()
                .unwrap()
                .actions
                .iter()
                .any(|action| matches!(action, Action::LearnTakeLesson { .. })));
            assert_eq!(game.state.players[0].sideboard, roster);
        }
    }

    #[test]
    fn learn_rejects_wrong_owner_nonlesson_unowned_and_wrong_actor_atomically() {
        let mut game = game(&[("Firebending Lesson", 1), ("Island", 1)]);
        learn(&mut game);
        let opponent_card = game.state.players[1].sideboard[0];
        let nonlesson = *game.state.players[0]
            .sideboard
            .iter()
            .find(|&&object| {
                game.state.source_base(game.state.object_of(object)).1.name == "Island"
            })
            .unwrap();
        let lesson = *game.state.players[0]
            .sideboard
            .iter()
            .find(|&&object| {
                game.state.source_base(game.state.object_of(object)).1.name == "Firebending Lesson"
            })
            .unwrap();
        for action in [
            Action::LearnTakeLesson {
                player: PlayerId(0),
                object: game.state.object_of(opponent_card),
            },
            Action::LearnTakeLesson {
                player: PlayerId(0),
                object: game.state.object_of(nonlesson),
            },
            Action::LearnTakeLesson {
                player: PlayerId(0),
                object: {
                    let mut zones = crate::state::zone::ZoneManager::default();
                    for _ in 0..game.state.objects.len() {
                        zones.allocate();
                    }
                    zones.allocate()
                },
            },
            Action::LearnTakeLesson {
                player: PlayerId(1),
                object: game.state.object_of(lesson),
            },
            Action::LearnDiscard {
                player: PlayerId(0),
                object: game.state.object_of(lesson),
            },
        ] {
            let before = witness(&game);
            assert!(game.execute_decision_action(&action).is_err());
            assert_eq!(witness(&game), before);
        }
    }

    #[test]
    fn learn_empty_hand_and_empty_sideboard_have_exact_choices() {
        for sideboard in [vec![], vec![("Firebending Lesson", 1)], vec![("Island", 1)]] {
            let mut game = game(&sideboard);
            let hand = game
                .state
                .zones
                .zone_objects(ZoneType::Hand, PlayerId(0))
                .to_vec();
            for object in hand {
                game.move_object(object, ZoneType::Graveyard);
            }
            learn(&mut game);
            let actions = &game.action_space().unwrap().actions;
            assert!(!actions
                .iter()
                .any(|action| matches!(action, Action::LearnDiscard { .. })));
            assert_eq!(
                actions.len(),
                if sideboard
                    .first()
                    .is_some_and(|(name, _)| *name == "Firebending Lesson")
                {
                    2
                } else {
                    1
                }
            );
            let before = game.state.deterministic_hash();
            let zones = game.state.zones.clone();
            game.execute_decision_action(&Action::Decline {
                player: PlayerId(0),
            })
            .unwrap();
            assert_eq!(
                serde_json::to_value(&game.state.zones).unwrap(),
                serde_json::to_value(&zones).unwrap()
            );
            assert_ne!(
                game.state.deterministic_hash(),
                before,
                "suspended continuation completes"
            );
        }
    }

    #[test]
    fn learn_sideboard_projection_is_private_complete_and_consumed_once() {
        use crate::agent::observation::Observation;

        let mut game = game(&[("Firebending Lesson", 40), ("Island", 1)]);
        let initial = Observation::for_player(&game, PlayerId(0));
        assert_eq!(initial.agent_sideboard.len(), 41);
        assert_eq!(
            initial
                .agent
                .sideboard_counts
                .as_ref()
                .unwrap()
                .values()
                .sum::<u32>(),
            41
        );
        assert_eq!(
            initial.agent.sideboard_counts,
            initial.agent.remaining_sideboard_counts
        );
        assert!(initial.agent_sideboard.iter().all(|outside| {
            initial
                .agent_cards
                .iter()
                .all(|object| object.id != outside.candidate_id)
        }));
        let other = Observation::for_player(&game, PlayerId(1));
        assert_eq!(other.agent_sideboard.len(), 1);
        assert_eq!(other.agent_sideboard[0].name, "Fancy Footwork");
        assert_eq!(
            other.opponent.sideboard_counts,
            initial.agent.sideboard_counts
        );
        assert!(other.action_space.actions.is_empty());

        learn(&mut game);
        let deciding = Observation::for_player(&game, PlayerId(0));
        let retrievals: Vec<_> = deciding
            .action_space
            .actions
            .iter()
            .filter(|action| {
                action.action_type == crate::agent::action::ActionType::LearnTakeLesson
            })
            .collect();
        assert_eq!(retrievals.len(), 40);
        for action in &retrievals {
            assert_eq!(action.focus.len(), 1);
            let outside = deciding
                .agent_sideboard
                .iter()
                .find(|object| object.candidate_id == action.focus[0])
                .unwrap();
            assert_eq!(outside.name, "Firebending Lesson");
        }
        let last = retrievals.last().unwrap().focus[0];
        let index = deciding
            .action_space
            .actions
            .iter()
            .rposition(|action| {
                action.action_type == crate::agent::action::ActionType::LearnTakeLesson
            })
            .unwrap();
        assert!(index > 32);
        let before = witness(&game);
        let mut branch = game.clone();
        let frame = branch.semantic_decision_frame().unwrap();
        branch
            .execute_semantic_command(&Command {
                command_id: "outside-projection".into(),
                expected_revision: frame.revision,
                offer_id: frame.offers[index].id.0,
                answers: vec![],
                object_preconditions: vec![],
            })
            .unwrap();
        let after = Observation::for_player(&branch, PlayerId(0));
        assert_eq!(after.agent_sideboard.len(), 40);
        assert!(after
            .agent_sideboard
            .iter()
            .all(|object| object.candidate_id != last));
        assert!(after
            .agent_cards
            .iter()
            .any(|object| object.id == last && object.zone == ZoneType::Hand));
        assert_eq!(after.agent.sideboard_counts, initial.agent.sideboard_counts);
        assert_eq!(
            after
                .agent
                .remaining_sideboard_counts
                .as_ref()
                .unwrap()
                .values()
                .sum::<u32>(),
            40
        );
        let other_after = Observation::for_player(&branch, PlayerId(1));
        assert_eq!(
            other_after.opponent.remaining_sideboard_counts,
            after.agent.remaining_sideboard_counts
        );
        assert_eq!(other_after.opponent.known_hand.values().sum::<u32>(), 1);
        assert_eq!(other_after.agent_sideboard, other.agent_sideboard);
        assert_eq!(witness(&game), before);
    }

    #[test]
    fn learn_complete_offers_execute_beyond_thirty_two_and_reject_stale_reuse() {
        let mut game = game(&[("Firebending Lesson", 40)]);
        learn(&mut game);
        let source = game.clone();
        let before = witness(&source);
        let frame = game.semantic_decision_frame().unwrap();
        assert_eq!(
            frame.offers.len(),
            game.action_space().unwrap().actions.len()
        );
        assert!(frame.offers.len() > 40);
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .rposition(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        assert!(index > 32);
        let command = Command {
            command_id: "take-last-lesson".into(),
            expected_revision: frame.revision,
            offer_id: frame.offers[index].id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        game.execute_semantic_command(&command).unwrap();
        let committed = witness(&game);
        assert!(game.execute_semantic_command(&command).is_err());
        assert_eq!(witness(&game), committed);
        assert_eq!(witness(&source), before);
    }

    #[test]
    fn learn_retrieval_forks_and_rolls_back_outside_inventory() {
        let mut source = game(&[("Firebending Lesson", 2)]);
        learn(&mut source);
        let source_witness = witness(&source);
        let mut branch = source.clone();
        let sibling = source.clone();
        let mark = branch.clone();
        let action_index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        branch.step(action_index).unwrap();
        let retrieved = witness(&branch);
        let nested = branch.clone();
        branch.step(0).unwrap();
        branch = nested;
        assert_eq!(witness(&branch), retrieved);
        branch = mark;
        assert_eq!(witness(&branch), source_witness);
        assert_eq!(witness(&source), source_witness);
        assert_eq!(witness(&sibling), source_witness);
    }

    #[test]
    fn learn_reveal_is_ordered_public_definition_and_known_hand_is_not_a_hand_leak() {
        use crate::agent::observation::{EventEntityKind, EventType, Observation};
        let mut game = game(&[("Firebending Lesson", 1)]);
        let card = game.state.players[0].sideboard[0];
        let definition = game.state.cards[card].definition;
        learn(&mut game);
        let before = game.semantic_observation(PlayerId(1)).unwrap();
        let event_start = game.state.events.len();
        let lesson = game.state.object_of(card);
        take(&mut game, lesson);
        let events: Vec<_> = game.state.events.iter().skip(event_start).collect();
        assert!(
            matches!(events[0], GameEvent::CardRevealed { owner: PlayerId(0), definition: d } if *d == definition)
        );
        assert!(matches!(
            events[1],
            GameEvent::CardMoved {
                from: None,
                to: ZoneType::Hand,
                ..
            }
        ));
        let reveal = Observation::event_data(events[0]);
        assert_eq!(reveal[0].event_type, EventType::CardRevealed as i32);
        assert_eq!(reveal[0].source_kind, EventEntityKind::Definition as i32);
        assert_eq!(reveal[0].source_id, definition.0 as i32);
        assert_eq!(reveal[0].source_incarnation, -1);
        let owner = Observation::for_player(&game, PlayerId(0));
        let other = Observation::for_player(&game, PlayerId(1));
        assert_eq!(owner.agent.known_hand, other.opponent.known_hand);
        assert_eq!(other.opponent.known_hand.get(&definition.0), Some(&1));
        assert!(!other
            .opponent_cards
            .iter()
            .any(|card| card.zone == ZoneType::Hand));
        assert_ne!(
            before.identity.viewer_state_hash,
            game.semantic_observation(PlayerId(1))
                .unwrap()
                .identity
                .viewer_state_hash
        );
        game.draw_cards(PlayerId(0), 1);
        assert_eq!(game.state.players[0].known_hand.get(&definition), Some(&1));
        // Rebuilding a viewer projection needs no event queue or client memory.
        game.take_observation_events();
        assert_eq!(
            Observation::for_player(&game, PlayerId(1))
                .opponent
                .known_hand
                .get(&definition.0),
            Some(&1)
        );
        let with_knowledge = game.state.deterministic_hash();
        let mut without = game.clone();
        without.state.players[0].known_hand.clear();
        assert_ne!(without.state.deterministic_hash(), with_knowledge);
    }

    fn small_known_hand_game() -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new(
                    "learner",
                    counts(&[("Firebending Lesson", 2), ("Island", 2)]),
                )
                .with_sideboard(counts(&[("Firebending Lesson", 2)])),
                PlayerConfig::new("opponent", counts(&[("Plains", 40)])),
            ],
            17,
            false,
        );
        game.keep_opening_hands();
        // One unknown hand slot, a residual pool of two Lessons and two Islands.
        let deck = game.state.players[0].deck.clone();
        for card in deck {
            game.move_object(game.state.object_of(card), ZoneType::Library);
        }
        let island = game.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|card| game.state[game.state.object_of(*card)].name == "Island")
            .unwrap();
        game.move_object(game.state.object_of(island), ZoneType::Hand);
        learn(&mut game);
        let card = game.state.players[0].sideboard[0];
        let lesson = game.state.object_of(card);
        take(&mut game, lesson);
        game
    }

    #[test]
    fn hidden_hand_constraints_match_exact_domain_without_truth() {
        use crate::possible_worlds::{HiddenHandConstraints, PossibleWorldSpace};
        let mut game = small_known_hand_game();
        let before = HiddenHandConstraints::for_viewer(&game, PlayerId(1));
        let exact = PossibleWorldSpace::for_viewer(&game, PlayerId(1)).projection();
        assert_eq!(before.pool, exact.pool);
        assert_eq!(before.known_hand, exact.known_hand);
        assert_eq!(before.hand_size, exact.hand_size);
        game.determinize(PlayerId(1), 781).unwrap();
        let after = HiddenHandConstraints::for_viewer(&game, PlayerId(1));
        assert_eq!(before, after);
    }

    #[test]
    fn learn_possible_worlds_use_residual_copy_weights_and_reject_missing_knowledge() {
        use crate::possible_worlds::{PossibleWorldSpace, WorldQuery};
        let game = small_known_hand_game();
        let space = PossibleWorldSpace::for_viewer(&game, PlayerId(1));
        assert_eq!(
            space.projection().known_hand,
            BTreeMap::from([("Firebending Lesson".to_string(), 1)])
        );
        assert_eq!(space.pool().values().sum::<u32>(), 5); // unchosen sideboard excluded
        assert_eq!(space.hand_size(), 2);
        assert_eq!(space.total_weight(), 4); // C(4 residual copies, 1 unknown slot)
        assert_eq!(space.worlds().len(), 2);
        for world in space.worlds() {
            assert_eq!(world.weight, 2); // no extra multiplicity for the known Lesson
            assert!(world.hand["Firebending Lesson"] >= 1);
            let branch = space.materialize(&game, world, 42).unwrap();
            assert_eq!(
                branch.state.players[0].known_hand,
                game.state.players[0].known_hand
            );
            assert_eq!(
                crate::possible_worlds::viewer_observation(&branch, PlayerId(1)),
                crate::possible_worlds::viewer_observation(&game, PlayerId(1))
            );
        }
        let excludes = WorldQuery::Lacks {
            card: "Firebending Lesson".to_string(),
            fewer_than: 1,
        };
        assert_eq!(space.support_receipt(&excludes).total_weight, 0);
        assert!(space.condition(&excludes).is_err());
        let mut forged = space.worlds()[0].clone();
        forged.hand = BTreeMap::from([("Island".to_string(), 2)]);
        assert!(space.materialize(&game, &forged, 42).is_err());
        let mut stale = game.clone();
        stale.state.players[0].known_hand.clear();
        assert!(space.materialize(&stale, &space.worlds()[0], 42).is_err());
    }

    #[test]
    fn learn_determinization_preserves_known_counts_refreshes_choices_and_ignores_copy_identity() {
        use crate::possible_worlds::PossibleWorldSpace;
        let mut source = small_known_hand_game();
        learn(&mut source);
        let before = witness(&source);
        let space = PossibleWorldSpace::for_viewer(&source, PlayerId(1));
        for index in 0..space.worlds().len() {
            let mut branch = space
                .materialize_index(
                    &source,
                    index,
                    42,
                    crate::possible_worlds::MaterializeMode::RefreshOpponentCommitment,
                )
                .unwrap();
            let hand = branch.state.zones.zone_objects(ZoneType::Hand, PlayerId(0));
            for action in &branch.action_space().unwrap().actions {
                if let Action::LearnDiscard { object, .. } = action {
                    assert!(hand.contains(*object));
                }
            }
            let action_index = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
                .unwrap();
            branch.step(action_index).unwrap();
            assert_eq!(branch.state.players[0].known_hand.values().sum::<u32>(), 2);
        }
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..32 {
            let mut branch = source.clone();
            branch.determinize(PlayerId(1), seed).unwrap();
            let hand = branch.state.zones.zone_objects(ZoneType::Hand, PlayerId(0));
            let lesson_count = hand
                .iter()
                .filter(|object| branch.state[**object].name == "Firebending Lesson")
                .count();
            assert!(lesson_count >= 1);
            seen.insert(lesson_count);
            let actions = &branch.action_space().unwrap().actions;
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| matches!(action, Action::LearnDiscard { .. }))
                    .count(),
                hand.len()
            );
            for action in actions {
                if let Action::LearnDiscard { object, .. } = action {
                    assert!(hand.contains(*object));
                }
            }
            assert_eq!(
                branch.state.players[0].known_hand,
                source.state.players[0].known_hand
            );
            assert_eq!(
                PossibleWorldSpace::for_viewer(&branch, PlayerId(1)).worlds(),
                space.worlds()
            );
            let index = actions
                .iter()
                .position(|action| matches!(action, Action::LearnDiscard { .. }))
                .unwrap();
            branch.step(index).unwrap();
        }
        assert_eq!(seen, std::collections::BTreeSet::from([1, 2]));
        assert_eq!(witness(&source), before);
        let own_choices = source.semantic_decision_frame().unwrap();
        let mut own = source.clone();
        own.determinize(PlayerId(0), 42).unwrap();
        assert_eq!(
            serde_json::to_value(own.semantic_decision_frame().unwrap()).unwrap(),
            serde_json::to_value(own_choices).unwrap()
        );

        let mut swapped = source.clone();
        let retrieved = source.state.players[0].sideboard[0];
        let duplicate = source.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|object| {
                source.state[source.state.object_of(*object)].name == "Firebending Lesson"
            })
            .unwrap();
        // An identical main-deck physical copy can witness the known card.
        let (retrieved_object, duplicate_object) = (
            swapped.state.object_of(retrieved),
            swapped.state.object_of(duplicate),
        );
        swapped
            .state
            .place_object(retrieved_object, PlayerId(0), ZoneType::Library);
        swapped
            .state
            .place_object(duplicate_object, PlayerId(0), ZoneType::Hand);
        assert_eq!(
            crate::possible_worlds::viewer_observation(&swapped, PlayerId(1)),
            crate::possible_worlds::viewer_observation(&source, PlayerId(1))
        );
        assert_eq!(
            PossibleWorldSpace::for_viewer(&swapped, PlayerId(1)).projection(),
            space.projection()
        );
        let swapped_branch = space.materialize(&swapped, &space.worlds()[0], 42).unwrap();
        let source_branch = space.materialize(&source, &space.worlds()[0], 42).unwrap();
        assert_eq!(
            serde_json::to_value(&swapped_branch.state.zones).unwrap(),
            serde_json::to_value(&source_branch.state.zones).unwrap()
        );
    }

    #[test]
    fn learn_public_duplicate_departure_decrements_known_minimum_without_copy_identity() {
        let mut game = small_known_hand_game();
        let retrieved = game.state.players[0].sideboard[0];
        let definition = game.state[game.state.object_of(retrieved)].definition_id;
        let duplicate = game.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|object| game.state[game.state.object_of(*object)].definition_id == definition)
            .unwrap();
        game.move_object(game.state.object_of(duplicate), ZoneType::Hand);
        assert_eq!(game.state.players[0].known_hand.get(&definition), Some(&1));
        game.move_object(game.state.object_of(duplicate), ZoneType::Graveyard);
        assert!(game.state.zones.contains(
            game.state.object_of(retrieved),
            ZoneType::Hand,
            PlayerId(0)
        ));
        assert!(game.state.players[0].known_hand.is_empty());
    }

    #[test]
    fn learn_nested_knowledge_and_search_fork_preserve_retained_roots() {
        let mut source = small_known_hand_game();
        learn(&mut source);
        source
            .state
            .suspended_decision
            .as_mut()
            .unwrap()
            .frame
            .queue
            .push_back(Effect::Learn);
        let before = witness(&source);
        let mut branch = source.clone();
        let sibling = source.clone();
        let outer = branch.clone();
        let card = branch.state.players[0].sideboard[0];
        let index = branch.action_space().unwrap().actions.iter().position(|action|
            matches!(action, Action::LearnDiscard { object: chosen, .. } if *chosen == branch.state.object_of(card))).unwrap();
        branch.step(index).unwrap();
        assert!(branch.state.players[0].known_hand.is_empty());
        let departed = witness(&branch);
        let inner = branch.clone();
        let index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        branch.step(index).unwrap();
        assert_eq!(branch.state.players[0].known_hand.values().sum::<u32>(), 1);
        branch.determinize(PlayerId(1), 22).unwrap();
        branch = inner;
        assert!(witness(&branch) == departed, "nested restore");
        branch = outer;
        assert!(witness(&branch) == before, "outer restore");
        assert!(witness(&source) == before);
        assert!(witness(&sibling) == before);
    }
}

#[cfg(test)]
mod payment_tests {
    use super::*;
    use crate::state::ability::{EffectValue, PaymentCost};
    use crate::state::payment::{How, Toward};
    use crate::{
        search_state::witness,
        state::{
            mana::{Mana, ManaCost},
            player::PlayerConfig,
        },
    };
    use std::collections::BTreeMap;

    fn payment_game(cost: PaymentCost) -> Game {
        payment_game_with_deck(cost, BTreeMap::from([("Island".into(), 40)]))
    }

    fn payment_game_with_deck(cost: PaymentCost, deck: BTreeMap<String, usize>) -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new("payer", deck),
                PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
            ],
            165,
            false,
        );
        game.keep_opening_hands();
        game.run_frame(EffectFrame::new(
            PlayerId(0),
            vec![
                Effect::OptionalPayment {
                    cost,
                    if_paid: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(2),
                    }],
                    if_declined: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
                Effect::DrawCards {
                    count: EffectValue::Fixed(3),
                },
            ],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        game
    }

    #[test]
    fn either_method_runs_exactly_one_selected_continuation_before_the_suffix() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{0}")),
            PaymentCost::Waterbend { generic: 0 },
        ] {
            for paid in [false, true] {
                let mut game = payment_game(cost.clone());
                let before = game.state.zones.size(ZoneType::Hand, PlayerId(0));
                let action = if paid {
                    Action::PayCost {
                        player: PlayerId(0),
                    }
                } else {
                    Action::Decline {
                        player: PlayerId(0),
                    }
                };
                game.execute_decision_action(&action).unwrap();
                assert!(game.state.suspended_decision.is_none());
                assert_eq!(
                    game.state.zones.size(ZoneType::Hand, PlayerId(0)),
                    before + if paid { 5 } else { 4 }
                );
                assert!(game.execute_decision_action(&action).is_err());
            }
        }
    }

    #[test]
    fn failed_payment_keeps_the_pending_obligation_and_continuations() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{2}")),
            PaymentCost::Waterbend { generic: 2 },
        ] {
            let mut game = payment_game(cost);
            let before = witness(&game);
            assert!(game
                .execute_decision_action(&Action::PayCost {
                    player: PlayerId(0)
                })
                .is_err());
            assert_eq!(witness(&game), before);
            let actions = &game.suspended_decision_action_space().unwrap().actions;
            assert_eq!(
                actions,
                &[Action::Decline {
                    player: PlayerId(0)
                }]
            );
        }
    }

    #[test]
    fn shared_payment_and_both_continuations_restore_across_nested_command_forks() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{0}")),
            PaymentCost::Waterbend { generic: 0 },
        ] {
            let game = payment_game(cost);
            let root = witness(&game);
            for index in 0..game.action_space().unwrap().actions.len() {
                let mut branch = game.clone();
                let outer = branch.clone();
                branch.step(index).unwrap();
                let after_payment = witness(&branch);
                let inner = branch.clone();
                branch.step(0).unwrap();
                branch = inner;
                assert_eq!(witness(&branch), after_payment);
                branch = outer;
                assert_eq!(witness(&branch), root);
                assert_eq!(witness(&game), root);
            }
        }
    }

    fn apply_payment_action(game: &mut Game, predicate: impl Fn(&Action) -> bool) {
        let action_index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(predicate)
            .expect("payment action is legal");
        game.step(action_index).unwrap();
    }

    #[test]
    fn resolution_payment_tap_retains_the_offer_and_pays_for_the_selected_effects() {
        let player = PlayerId(0);
        let mut game = payment_game(PaymentCost::Mana(ManaCost::parse("U")));
        let island = game
            .scenario_force_battlefield(player, "Island", true)
            .unwrap();
        let ordinary = game.suspended_decision_action_space().unwrap().actions;
        game.mana_floats = [ManaFloats::All, ManaFloats::Off];
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let actions = &game.action_space().unwrap().actions;
        assert_eq!(&actions[..ordinary.len()], ordinary);
        assert_eq!(
            actions[ordinary.len()..],
            [Action::TapForMana {
                player,
                permanent: island
            }]
        );
        let before = game.state.zones.size(ZoneType::Hand, player);
        game.step(ordinary.len()).unwrap();
        assert!(game.state.permanents[island].as_ref().unwrap().tapped);
        assert_eq!(game.state.players[0].mana_pool, Mana::parse("U"));
        assert_eq!(game.action_space().unwrap().actions, ordinary);
        apply_payment_action(&mut game, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
        assert_eq!(game.state.zones.size(ZoneType::Hand, player), before + 5);
    }

    #[test]
    fn auto_payment_keeps_optional_acceptance_and_funds_the_nested_color() {
        let player = PlayerId(0);
        for accept in [false, true] {
            let mut game = payment_game_with_deck(
                PaymentCost::Mana(ManaCost::parse("G")),
                BTreeMap::from([("Island".into(), 40), ("Prophetic Prism".into(), 1)]),
            );
            let prism = game
                .scenario_force_battlefield(player, "Prophetic Prism", true)
                .unwrap();
            let island = game
                .scenario_force_battlefield(player, "Island", true)
                .unwrap();
            game.mana_payment[0] = crate::state::player::ManaPayment::Auto;
            game.publish_action_space(game.suspended_decision_action_space().unwrap());
            assert_eq!(
                game.action_space().unwrap().actions,
                [Action::PayCost { player }, Action::Decline { player }]
            );
            let before = game.state.zones.size(ZoneType::Hand, player);
            game.step(usize::from(!accept)).unwrap();
            assert_eq!(
                game.state.zones.size(ZoneType::Hand, player),
                before + if accept { 5 } else { 4 }
            );
            for id in [prism, island] {
                assert_eq!(game.state.permanents[id].as_ref().unwrap().tapped, accept);
            }
            assert_eq!(game.state.players[0].mana_pool.total(), 0);
        }
    }

    #[test]
    fn nested_mana_activation_allows_surplus_before_paying_its_fee() {
        let player = PlayerId(0);
        let mut game = payment_game_with_deck(
            PaymentCost::Mana(ManaCost::parse("G")),
            BTreeMap::from([("Island".into(), 40), ("Prophetic Prism".into(), 1)]),
        );
        let prism = game
            .scenario_force_battlefield(player, "Prophetic Prism", true)
            .unwrap();
        for _ in 0..2 {
            game.scenario_force_battlefield(player, "Island", true)
                .unwrap();
        }
        game.mana_floats = [ManaFloats::All, ManaFloats::Off];
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        apply_payment_action(
            &mut game,
            |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism),
        );
        apply_payment_action(&mut game, |a| matches!(a, Action::TapForMana { .. }));
        apply_payment_action(&mut game, |a| matches!(a, Action::PayCost { .. }));
        assert!(game.state.permanents[prism].as_ref().unwrap().tapped);
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
        // WUBRG order: select green, then spend it on the original effect.
        apply_payment_action(&mut game, |a| {
            matches!(a, Action::ChooseMode { mode: 4, .. })
        });
        assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
        apply_payment_action(&mut game, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
    }

    #[test]
    fn partial_waterbend_and_paid_mana_choice_restore_across_nested_command_forks() {
        let player = PlayerId(0);
        let mut game = payment_game_with_deck(
            PaymentCost::Waterbend { generic: 2 },
            BTreeMap::from([
                ("Island".into(), 40),
                ("Hermitic Herbalist".into(), 1),
                ("Prophetic Prism".into(), 1),
            ]),
        );
        let mut sources = Vec::new();
        for name in ["Hermitic Herbalist", "Prophetic Prism"] {
            let card = game.state.players[0]
                .deck
                .iter()
                .copied()
                .find(|card| game.state[game.state.object_of(*card)].name == name)
                .unwrap();
            game.move_object(game.state.object_of(card), ZoneType::Battlefield);
            sources.push(game.state.live(game.state.object_of(card)).unwrap());
        }
        let waterbend = game.lookup_current_permanent(sources[0]).unwrap();
        let prism = game.lookup_current_permanent(sources[1]).unwrap();
        game.state.players[0].mana_pool = Mana::parse("1").into();
        game.invalidate_mana_cache(player);
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let root = witness(&game);
        let before_hand = game.state.zones.size(ZoneType::Hand, player);

        // Keep a divergent sibling alive throughout all nested restores.
        let mut sibling = game.clone();
        apply_payment_action(&mut sibling, |a| matches!(a, Action::Decline { .. }));
        let sibling_witness = witness(&sibling);
        assert_eq!(
            sibling.state.zones.size(ZoneType::Hand, player),
            before_hand + 4
        );

        let mut branch = game.clone();
        let outer = branch.clone();
        apply_payment_action(
            &mut branch,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == waterbend),
        );
        let partial = witness(&branch);
        let partial_resolution = branch.state.suspended_decision.clone().unwrap();
        let Decision::Payment(payment) = &partial_resolution.decision else {
            panic!("partial waterbend must retain its payment");
        };
        assert_eq!(
            payment.owed,
            CostComponent::Mana {
                cost: ManaCost::parse("{1}"),
                help: crate::flow::payment::Help {
                    waterbend: 2,
                    ..Default::default()
                },
            }
        );
        assert_eq!(
            payment
                .receipt
                .contributed(How::Tapped, Toward::Waterbend)
                .collect::<Vec<_>>(),
            vec![sources[0]]
        );
        assert!(!payment.can_decline());
        assert!(!branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));

        let inner = branch.clone();
        let activate_prism = |a: &Action| matches!(a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == prism);
        apply_payment_action(&mut branch, activate_prism);
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
        assert!(branch.state.permanents[prism].as_ref().unwrap().tapped);
        let modal_resolution = branch.state.suspended_decision.clone().unwrap();
        let Decision::ManaChoice {
            resume: Some(resume),
            ..
        } = &modal_resolution.decision
        else {
            panic!("Prism must suspend the committed payment for its immediate color choice");
        };
        assert_eq!(**resume, partial_resolution.decision);
        assert_eq!(modal_resolution.frame, partial_resolution.frame);
        assert_eq!(branch.action_space().unwrap().kind, ActionSpaceKind::Modal);
        let modal = witness(&branch);

        let color_mark = branch.clone();
        let choose_blue = |a: &Action| matches!(a, Action::ChooseMode { mode: 1, .. });
        apply_payment_action(&mut branch, choose_blue);
        assert_eq!(
            branch.state.suspended_decision,
            Some(partial_resolution.clone())
        );
        assert_eq!(branch.state.players[0].mana_pool, Mana::parse("U"));
        for source in [waterbend, prism] {
            assert!(branch.state.permanents[source].as_ref().unwrap().tapped);
        }
        assert!(!branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));
        let resumed = witness(&branch);
        apply_payment_action(&mut branch, |a| matches!(a, Action::PayCost { .. }));
        assert!(branch.state.suspended_decision.is_none());
        assert_eq!(
            branch.state.zones.size(ZoneType::Hand, player),
            before_hand + 5
        );
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
        let completed = witness(&branch);

        branch = color_mark;
        assert_eq!(witness(&branch), modal);
        assert_eq!(branch.state.suspended_decision, Some(modal_resolution));
        apply_payment_action(&mut branch, choose_blue);
        assert_eq!(witness(&branch), resumed);
        apply_payment_action(&mut branch, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(witness(&branch), completed);

        branch = inner;
        assert_eq!(witness(&branch), partial);
        assert_eq!(branch.state.suspended_decision, Some(partial_resolution));
        assert!(!branch.state.permanents[prism].as_ref().unwrap().tapped);
        assert_eq!(branch.state.players[0].mana_pool, Mana::parse("1"));
        apply_payment_action(&mut branch, activate_prism);
        assert_eq!(witness(&branch), modal);
        branch = outer;
        assert_eq!(witness(&branch), root);
        assert_eq!(witness(&game), root);
        assert_eq!(witness(&sibling), sibling_witness);
    }
}

#[cfg(test)]
mod source_scope_tests {
    use super::EffectFrame;
    use crate::{
        flow::game::Game,
        state::{
            game_object::{AbilitySource, PlayerId},
            player::PlayerConfig,
        },
    };
    use std::collections::BTreeMap;

    #[test]
    fn child_scope_inherits_one_source_and_both_v16_projections() {
        let deck = BTreeMap::from([("Mountain".into(), 40)]);
        let game = Game::new(
            vec![
                PlayerConfig::new("a", deck.clone()),
                PlayerConfig::new("b", deck),
            ],
            341,
            false,
        );
        let object = game.state.latest_objects().next().unwrap().0;
        let mut parent = EffectFrame::new(PlayerId(0), vec![]);
        parent.bind_source(Some(object), Some(object));
        let child = parent.child_scope(vec![]);
        assert_eq!(
            child.source_identity,
            Some(AbilitySource::new(object, Some(object)))
        );
        let bytes = serde_json::to_string(&child).unwrap();
        assert!(bytes.contains(&format!(
            "\"source\":{},\"source_ref\":{{\"entity\":{},\"incarnation\":{}}}",
            object.slot(),
            object.slot(),
            object.generation()
        )));
        let absent = serde_json::to_string(&EffectFrame::new(PlayerId(0), vec![])).unwrap();
        assert!(absent.contains("\"source\":null,\"source_ref\":null"));
    }
}
