// cast_permission.rs
// Permissions to cast exact cards outside their owner's hand, and the
// prohibitions that override them.
//
// `CastPermissions` lives on `GameState`. A grant names the exact zone
// object (CR 400.7): once that card changes zones it is a new object and the
// grant is stale, so lookups compare the current incarnation instead of
// pruning on every zone move. Airbend (CR 701.65) is the first producer.
// A `HandOnly` restriction ("can't cast spells from anywhere other than
// their hands") overrides every non-hand casting permission, including
// graveyard flashback, until the start of its controller's next turn.
// Foretell (CR 702.143) exiles a card face down with a grant marked
// `foretold`: the grant is also the authority for that card being hidden
// from every player but its owner.

use crate::{
    agent::action::AgentError,
    flow::{
        decision::{EffectFrame, SpendCompletion},
        game::Game,
        proposed_event::ZoneEntry,
    },
    state::{
        ability::{LinkedGrant, PlayWindow},
        game_object::{CardId, ObjectId, PlayerId},
        mana::ManaCost,
        zone::ZoneType,
    },
};

/// The rule chosen to play one exact object. A grant keeps its key when other grants expire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PlayRoute {
    Hand,
    LibraryTop,
    PrintedFlashback,
    GrantedFlashback,
    Grant(usize),
    GraveyardType {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        source: ObjectId,
        /// Public join retained if paying the cost removes the source.
        #[serde(skip)]
        permanent_id: crate::state::game_object::PublicId,
        kind: crate::state::card::CardType,
    },
}

/// Permission for one player to cast or play a card in one exact zone appearance.
/// The granting effect determines the eligible object and the duration.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PlayGrant {
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub key: usize,
    /// Printed source of a resolved independent grant, even after it leaves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<crate::state::card::CardDefId>,
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub object: ObjectId,
    pub player: PlayerId,
    /// Alternative cost; `None` pays the card's own mana cost.
    pub cost: Option<ManaCost>,
    /// "this turn": the grant ends in the cleanup step.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub this_turn: bool,
    /// "Until the end of your next turn": the turn number it was granted on;
    /// it ends in the cleanup step of `player`'s first turn after that one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub through_next_turn: Option<u32>,
    /// Exact battlefield source. Preparation follows its controller. Other
    /// source-conditioned grants require the recorded player to control it.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub source: Option<ObjectId>,
    /// Usable only during `player`'s turn.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub own_turn_only: bool,
    /// The cost's generic may be paid by waterbending.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub waterbend: bool,
    /// Castable as though it had flash.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flash: bool,
    /// Mana of any type can be spent: colored pips may be paid by any mana.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub any_mana: bool,
    /// Usable only on a turn after this one ("cast it on a later turn").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_turn: Option<u32>,
    /// Foretold: the card lies face down and is cast for its foretell cost.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub foretold: bool,
    /// CR 715.3d: only the permanent spell may be cast with this grant.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub on_adventure: bool,
    /// This exact exile copy survives while linked to its battlefield source.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub prepared: bool,
}

#[deprecated(note = "use PlayGrant; ExileCastGrant will be removed in 0.4.0")]
pub type ExileCastGrant = PlayGrant;

impl PlayGrant {
    /// A grant with no further terms: cast for the card's mana cost, any time
    /// its timing allows.
    pub fn new(object: ObjectId, player: PlayerId) -> Self {
        Self {
            key: 0,
            origin: None,
            object,
            player,
            cost: None,
            this_turn: false,
            through_next_turn: None,
            source: None,
            own_turn_only: false,
            waterbend: false,
            flash: false,
            any_mana: false,
            after_turn: None,
            foretold: false,
            on_adventure: false,
            prepared: false,
        }
    }
}

/// `player` casts spells only from hand until `until_turn_of` begins a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct HandOnly {
    pub player: PlayerId,
    pub until_turn_of: PlayerId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CastPermissions {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(rename = "exile_grants")] // Retained state digest field.
    pub grants: Vec<PlayGrant>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hand_only: Vec<HandOnly>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spent_types: Vec<(PlayerId, PlayRoute)>,
}

impl CastPermissions {
    /// Keep one copy of identical terms and give distinct terms a stable
    /// key within this exact appearance. Cleanup may compact the grant list.
    pub fn grant(&mut self, mut grant: PlayGrant) {
        let mut next = 0;
        for existing in self
            .grants
            .iter()
            .filter(|other| other.object == grant.object)
        {
            grant.key = existing.key;
            if *existing == grant {
                return;
            }
            next = next.max(existing.key + 1);
        }
        grant.key = next;
        self.grants.push(grant);
    }

    pub fn is_empty(&self) -> bool {
        self.grants.is_empty() && self.hand_only.is_empty() && self.spent_types.is_empty()
    }
}

/// Airbend's alternative cost (CR 701.65a).
const AIRBEND_COST: &str = "{2}";

/// The cost of the foretell special action (CR 702.143a).
pub(crate) const FORETELL_ACTION_COST: &str = "{2}";

impl Game {
    /// Airbend a permanent or spell. Only a card that reaches exile gets the grant.
    pub(crate) fn airbend(&mut self, object_id: ObjectId) {
        let owner = self.state[object_id].owner;
        self.exile_with_play_grant(object_id, |object| PlayGrant {
            cost: Some(ManaCost::parse(AIRBEND_COST)),
            ..PlayGrant::new(object, owner)
        });
    }

    /// "Exile the top `count` cards of your library. You may play them
    /// [this turn | until the end of your next turn]."
    pub(crate) fn exile_top_cards_to_play(
        &mut self,
        player: PlayerId,
        count: usize,
        until: PlayWindow,
    ) {
        let turn = self.state.turn.turn_number;
        for _ in 0..count {
            let Some(object_id) = self.state.zones.top(ZoneType::Library, player) else {
                return;
            };
            self.exile_with_play_grant(object_id, |object| PlayGrant {
                this_turn: until == PlayWindow::ThisTurn,
                through_next_turn: (until == PlayWindow::ThroughYourNextTurn).then_some(turn),
                ..PlayGrant::new(object, player)
            });
        }
    }

    /// Whether `player` may foretell `card` now: a card with foretell in
    /// their hand, during their turn, with {2} available (CR 702.143a).
    pub(crate) fn can_foretell(&self, player: PlayerId, card: CardId) -> bool {
        self.active_player() == player
            && self.state[self.state.object_of(card)].foretell().is_some()
            && self
                .state
                .zones
                .contains(self.state.object_of(card), ZoneType::Hand, player)
            && self.can_pay_mana_cost(player, &ManaCost::parse(FORETELL_ACTION_COST))
    }

    /// The foretell special action: pay {2} and exile `card` face down
    /// without using the stack. Its owner may cast it on a later turn for
    /// its foretell cost.
    pub(crate) fn foretell(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        if !self.can_foretell(player, card) {
            return Err(AgentError("card cannot be foretold now".into()));
        }
        let cost = ManaCost::parse(FORETELL_ACTION_COST);
        self.produce_mana(player, &cost)?;
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Foretell(card),
            EffectFrame::new(player, Vec::new()),
        )
    }

    pub(crate) fn complete_foretell(&mut self, player: PlayerId, card: CardId) {
        let turn = self.state.turn.turn_number;
        let exiled = self.move_object_with_entry(
            self.state.object_of(card),
            ZoneType::Exile,
            ZoneEntry {
                face_down: true,
                ..ZoneEntry::default()
            },
        );
        self.grant_if_exiled(exiled, |object| PlayGrant {
            after_turn: Some(turn),
            foretold: true,
            ..PlayGrant::new(object, player)
        });
        self.restart_priority_round();
    }

    /// Whether `card` is a foretold card lying face down in exile, hidden
    /// from everyone but its owner.
    pub(crate) fn is_foretold(&self, object: ObjectId) -> bool {
        self.grants_naming(object).any(|grant| grant.foretold)
    }

    /// Determinization: refill `player`'s face-down foretold slots from the
    /// ordered unseen `pool`, taking its first foretell-bearing cards out of
    /// it. A chosen card already foretold keeps its own slot, so no card
    /// fills two slots; each vacated slot takes one incoming card, keeping
    /// its cast grant, and its old card is left in `pool` for the caller to
    /// assign to the hand or library. Returns false, changing nothing, when
    /// the pool holds too few foretell-bearing cards.
    pub(crate) fn fill_foretold_slots(
        &mut self,
        player: PlayerId,
        pool: &mut Vec<ObjectId>,
    ) -> bool {
        let slots: Vec<ObjectId> = self
            .foretold_cards(player)
            .map(|card| self.state.object_of(card))
            .collect();
        let bearers = pool
            .iter()
            .filter(|object_id| self.state[**object_id].foretell().is_some())
            .count();
        if bearers < slots.len() {
            return false;
        }
        let mut chosen = Vec::with_capacity(slots.len());
        pool.retain(|object_id| {
            let take = chosen.len() < slots.len() && self.state[*object_id].foretell().is_some();
            if take {
                chosen.push(*object_id);
            }
            !take
        });
        let mut incoming = chosen.iter().filter(|object_id| !slots.contains(object_id));
        for &old in slots.iter().filter(|old| !chosen.contains(old)) {
            let new = *incoming.next().expect("one incoming card per vacated slot");
            let from = self.state.live(old).expect("a foretold card is exiled");
            let to = self.state.live(new).expect("an unseen card is hidden");
            self.state.zones.replace_face_down_exile(player, old, new);
            for grant in &mut self.state.cast_permissions.grants {
                if grant.object == from {
                    grant.object = to;
                }
            }
        }
        true
    }

    /// `player`'s face-down foretold cards in exile.
    pub(crate) fn foretold_cards(&self, player: PlayerId) -> impl Iterator<Item = CardId> + '_ {
        self.state
            .zone_cards(ZoneType::Exile, player)
            .filter(|card| self.is_foretold(self.state.object_of(*card)))
    }

    /// Exile `card` linked to `source`: while `player` controls `source`,
    /// they may cast it during their turn as `linked` allows (Hama, Azula).
    /// Without a source the card is only exiled.
    pub(crate) fn exile_linked(
        &mut self,
        object_id: ObjectId,
        player: PlayerId,
        source: Option<ObjectId>,
        linked: LinkedGrant,
    ) {
        let Some(source) = source else {
            self.move_object(object_id, ZoneType::Exile);
            return;
        };
        let mana_value = self.object_mana_value(object_id);
        self.exile_with_play_grant(object_id, |object| {
            let base = PlayGrant {
                source: Some(source),
                own_turn_only: true,
                ..PlayGrant::new(object, player)
            };
            match linked {
                LinkedGrant::WaterbendManaValue => PlayGrant {
                    cost: Some(ManaCost::default().with_generic(mana_value)),
                    waterbend: true,
                    ..base
                },
                LinkedGrant::FlashAnyMana => PlayGrant {
                    flash: true,
                    any_mana: true,
                    ..base
                },
            }
        });
    }

    /// Exile `card` and add the play grant `grant` builds for the exiled
    /// object.
    fn exile_with_play_grant(
        &mut self,
        object: ObjectId,
        grant: impl FnOnce(ObjectId) -> PlayGrant,
    ) {
        let exiled = self.move_object(object, ZoneType::Exile);
        self.grant_if_exiled(exiled, grant);
    }

    /// Add the grant `grant` builds for `card`'s exiled object. A token, or a
    /// card a replacement kept out of exile, receives no grant.
    fn grant_if_exiled(&mut self, object_id: ObjectId, grant: impl FnOnce(ObjectId) -> PlayGrant) {
        if self.state.zones.zone_of(object_id) != Some(ZoneType::Exile)
            || !self.state[object_id].is_card()
        {
            return;
        }
        self.state.cast_permissions.grant(grant(object_id));
    }

    /// "This turn" grants end in the cleanup step (CR 514.2), and "until the
    /// end of your next turn" ones in that of the permitted player's next
    /// turn.
    pub(crate) fn expire_turn_cast_grants(&mut self) {
        self.state.cast_permissions.spent_types.clear();
        let (active, turn) = (self.active_player(), self.state.turn.turn_number);
        self.state.cast_permissions.grants.retain(|grant| {
            !grant.this_turn
                && !grant
                    .through_next_turn
                    .is_some_and(|granted| grant.player == active && turn > granted)
        });
    }

    /// "Until your next turn, your opponents can't cast spells from anywhere
    /// other than their hands."
    pub(crate) fn restrict_opponent_to_hand(&mut self, controller: PlayerId) {
        self.state.cast_permissions.hand_only.push(HandOnly {
            player: PlayerId(1 - controller.0),
            until_turn_of: controller,
        });
    }

    /// Restrictions lasting "until your next turn" end as that turn begins.
    pub(crate) fn expire_cast_restrictions(&mut self, active: PlayerId) {
        self.state
            .cast_permissions
            .hand_only
            .retain(|restriction| restriction.until_turn_of != active);
    }

    /// The grants naming this exact card in its current zone appearance (CR 400.7).
    fn grants_naming(&self, object_id: ObjectId) -> impl Iterator<Item = &PlayGrant> {
        self.state
            .cast_permissions
            .grants
            .iter()
            .filter(move |grant| {
                grant.object == object_id && self.state.zones.zone_of(object_id).is_some()
            })
    }

    pub(super) fn grant_by_key(&self, object: ObjectId, key: usize) -> Option<&PlayGrant> {
        self.grants_naming(object).find(|grant| grant.key == key)
    }

    /// The battlefield appearance keeping this particular spell copy prepared.
    pub(crate) fn preparation_source(&self, object: ObjectId) -> Option<ObjectId> {
        if self.state[object].kind() != crate::state::game_object::ObjectKind::SpellCopy
            || self.state.zones.zone_of(object) != Some(ZoneType::Exile)
        {
            return None;
        }
        let (grant, _) = self
            .play_grants_on(object)
            .find(|(grant, _)| grant.prepared)?;
        grant.source
    }

    /// Resolve the caster independently of timing and payment. Preparation follows
    /// its source, while a fixed-player grant may require continued control.
    fn grant_player(&self, grant: &PlayGrant) -> Option<PlayerId> {
        let Some(source) = grant.source else {
            return (!grant.prepared).then_some(grant.player);
        };
        let permanent = self.lookup_current_permanent(source).ok()?;
        let controller = self.state.permanents[permanent].as_ref()?.controller;
        (grant.prepared || controller == grant.player).then_some(controller)
    }

    /// The live grant letting `player` play this exact card now.
    /// Another player's card is castable too; a permanent spell enters
    /// under its caster's control (CR 110.2).
    pub(super) fn live_grants(
        &self,
        player: PlayerId,
        object: ObjectId,
    ) -> impl Iterator<Item = &PlayGrant> {
        self.grants_naming(object).filter(move |grant| {
            self.grant_player(grant) == Some(player)
                && (!grant.own_turn_only || self.active_player() == player)
                && grant
                    .after_turn
                    .is_none_or(|turn| self.state.turn.turn_number > turn)
        })
    }

    /// What casting `card` under `grant` costs: its foretell cost, the
    /// grant's alternative, or its mana cost. `None` when it has no cost to
    /// pay (a land is played, not cast).
    pub(crate) fn grant_cast_cost(&self, grant: &PlayGrant) -> Option<ManaCost> {
        self.grant_cost_for_spell(grant, &self.state[grant.object])
    }

    pub(super) fn grant_cost_for_spell(
        &self,
        grant: &PlayGrant,
        definition: &crate::state::card::PrintedCharacteristics,
    ) -> Option<ManaCost> {
        if definition.types.is_land() {
            return None;
        }
        let cost = if grant.foretold {
            definition.foretell().cloned()?
        } else {
            grant
                .cost
                .clone()
                .or_else(|| definition.mana_cost.clone())?
        };
        // Any type of mana pays every pip, as generic (CR 609.4b-style).
        Some(if grant.any_mana {
            // Spending permission changes colors, not the announced X cost.
            cost.pay_with_any_mana()
        } else {
            cost
        })
    }

    /// The permission naming this exact card, whatever its timing,
    /// for viewers to inspect. A grant tied to a source that has left the
    /// battlefield can never be used again and is not shown.
    pub(crate) fn play_grants_on(
        &self,
        object: ObjectId,
    ) -> impl Iterator<Item = (&PlayGrant, PlayerId)> {
        self.grants_naming(object)
            .filter_map(|grant| Some((grant, self.grant_player(grant)?)))
    }

    /// The player whose next turn ends `player`'s hand-only casting
    /// restriction, while one applies.
    pub(crate) fn hand_only_until(&self, player: PlayerId) -> Option<PlayerId> {
        self.state
            .cast_permissions
            .hand_only
            .iter()
            .find(|restriction| restriction.player == player)
            .map(|restriction| restriction.until_turn_of)
    }

    /// Exiled candidates only. Priority already visits the player's graveyard.
    pub(crate) fn exile_granted_objects(
        &self,
        player: PlayerId,
    ) -> impl Iterator<Item = ObjectId> + '_ {
        let mut seen = std::collections::BTreeSet::new();
        self.state
            .cast_permissions
            .grants
            .iter()
            .filter(move |grant| {
                self.grant_player(grant) == Some(player)
                    && self.state.zones.zone_of(grant.object) == Some(ZoneType::Exile)
            })
            .map(|grant| grant.object)
            .filter(move |object| seen.insert(*object))
    }
}
