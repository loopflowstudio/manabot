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
// A card face down in exile (CR 406.3) is hidden by its grant: Foretell
// (CR 702.143) marks it `foretold`, visible to its owner; Flameshape marks
// it `face_down`, visible to the grant's player only while the grant's
// `condition` holds. Determinization refills both kinds of hidden slot at
// once (`refill_hidden_exile_slots`).

use crate::{
    agent::action::AgentError,
    flow::{
        decision::{EffectFrame, SpendCompletion},
        game::Game,
        proposed_event::ZoneEntry,
    },
    state::{
        ability::LinkedGrant,
        game_object::{CardId, ObjectId, PlayerId},
        mana::ManaCost,
        zone::ZoneType,
    },
};

/// The rule chosen to play one exact object. A grant keeps its key when other grants expire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PlayRoute {
    Hand,
    FreeFromHand,
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
    /// A card someone else owns, in exile with a counter of this kind, on
    /// its player's turn (Tinybones's stash counters).
    ExileCounter(crate::state::permanent::CounterKind),
    /// The card's own printed permission to cast it from its owner's
    /// graveyard for an additional cost other than flashback (Quilled
    /// Greatwurm's counters).
    PrintedGraveyard,
}

/// The observed code of a route (0 is an ordinary hand play): offers name
/// overlapping routes by it, and the `play_route` action column reads it, so
/// a new route appends a code.
pub fn play_route_code(route: PlayRoute) -> u8 {
    match route {
        PlayRoute::Hand => 1,
        PlayRoute::FreeFromHand => 2,
        PlayRoute::LibraryTop => 3,
        PlayRoute::PrintedFlashback => 4,
        PlayRoute::GrantedFlashback => 5,
        PlayRoute::Grant(_) => 6,
        PlayRoute::GraveyardType { .. } => 7,
        PlayRoute::ExileCounter(_) => 8,
        PlayRoute::PrintedGraveyard => 9,
    }
}

/// Permission for one player to cast, or with `play_lands` play, a card in one exact zone appearance.
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
    /// "You may play": a land face may be played with this grant (CR 305.1).
    /// Otherwise the grant only casts spells ("you may cast").
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub play_lands: bool,
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
    /// The grant's caster tracks the player who controls its battlefield source.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub follows_source_controller: bool,
    /// This cast uses flashback and exiles the spell when it leaves the stack.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flashback: bool,
    /// This exact exile copy survives while linked to its battlefield source.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub prepared: bool,
    /// The card lies face down in exile (CR 406.3), seen by `player` only
    /// while `condition` holds. A foretold card is face down too, but its
    /// owner always sees it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub face_down: bool,
    /// The grant is usable, and a face-down card visible, only while this
    /// holds for `player` ("if you control a Wizard").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<crate::state::ability::Condition>,
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
            play_lands: false,
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
            follows_source_controller: false,
            flashback: false,
            face_down: false,
            condition: None,
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
    /// Each surviving card of an airbent object can be cast independently.
    pub(super) fn grant_airbend_cast(&mut self, object_id: ObjectId) {
        let owner = self.state[object_id].owner;
        self.grant_if_exiled(object_id, |object| PlayGrant {
            cost: Some(ManaCost::parse(AIRBEND_COST)),
            ..PlayGrant::new(object, owner)
        });
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
        let face_down = ZoneEntry {
            face_down: true,
            ..ZoneEntry::default()
        };
        self.exile_with_play_grant(self.state.object_of(card), face_down, |object| PlayGrant {
            after_turn: Some(turn),
            foretold: true,
            ..PlayGrant::new(object, player)
        });
        self.restart_priority_round();
    }

    /// Whether `object` is a foretold card: cast for its foretell cost, and
    /// fillable only by a foretell-bearing card when determinized.
    pub(crate) fn is_foretold(&self, object: ObjectId) -> bool {
        self.grants_naming(object).any(|grant| grant.foretold)
    }

    /// Whether `object` lies face down in exile (CR 406.3).
    pub(crate) fn is_face_down_in_exile(&self, object: ObjectId) -> bool {
        self.state.zones.zone_of(object) == Some(ZoneType::Exile)
            && self
                .grants_naming(object)
                .any(|grant| grant.foretold || grant.face_down)
    }

    /// Whether `viewer` may look at face-down `object`: a foretold card's
    /// owner, or a face-down grant's player while its condition holds, and
    /// whoever shares their information.
    pub(crate) fn can_view_face_down_exile(&self, object: ObjectId, viewer: PlayerId) -> bool {
        self.grants_naming(object).any(|grant| {
            if grant.foretold {
                return self.shares_information(viewer, self.state[object].owner);
            }
            grant.face_down
                && self.grant_player(grant).is_some_and(|player| {
                    self.shares_information(viewer, player)
                        && self.grant_condition_holds(grant, player)
                })
        })
    }

    /// Whether `object` is face down in exile and hidden from `viewer`.
    pub(crate) fn hides_face_down_exile(&self, object: ObjectId, viewer: PlayerId) -> bool {
        self.is_face_down_in_exile(object) && !self.can_view_face_down_exile(object, viewer)
    }

    fn grant_condition_holds(&self, grant: &PlayGrant, player: PlayerId) -> bool {
        grant
            .condition
            .as_ref()
            .is_none_or(|condition| self.check_static_condition(condition, player))
    }

    /// `owner`'s face-down exiled cards hidden from `viewer`, in exile
    /// order. These are unseen cards: determinization refills them.
    pub(crate) fn hidden_face_down_exile(
        &self,
        owner: PlayerId,
        viewer: PlayerId,
    ) -> Vec<ObjectId> {
        self.state
            .zones
            .zone_objects(ZoneType::Exile, owner)
            .iter()
            .copied()
            .filter(|object| self.hides_face_down_exile(*object, viewer))
            .collect()
    }

    /// Counts of `owner`'s hidden face-down exile slots for `viewer`:
    /// (foretold, any other face down).
    pub(crate) fn hidden_face_down_counts(&self, owner: PlayerId, viewer: PlayerId) -> (u32, u32) {
        let hidden = self.hidden_face_down_exile(owner, viewer);
        let foretold = hidden
            .iter()
            .filter(|object| self.is_foretold(**object))
            .count() as u32;
        (foretold, hidden.len() as u32 - foretold)
    }

    /// Determinization: refill every face-down exile slot of `owner` hidden
    /// from `viewer` from the ordered unseen `pool`, which holds those slots'
    /// own cards. Foretold slots take the pool's first foretell-bearing
    /// cards; the other face-down slots then take its first remaining cards.
    /// A chosen card already in a slot of its kind keeps it; every other slot
    /// takes one incoming card and keeps its grants, and the cards leaving
    /// exile stay in `pool` for the caller to deal to the hand or library.
    /// Slots and grants move in one simultaneous exchange, so a card moving
    /// between slots never doubles in exile. Returns false, changing nothing,
    /// when the pool cannot fill every slot.
    pub(crate) fn refill_hidden_exile_slots(
        &mut self,
        owner: PlayerId,
        viewer: PlayerId,
        pool: &mut Vec<ObjectId>,
    ) -> bool {
        let (foretold_slots, face_down_slots): (Vec<ObjectId>, Vec<ObjectId>) = self
            .hidden_face_down_exile(owner, viewer)
            .into_iter()
            .partition(|object| self.is_foretold(*object));
        let bearers = pool
            .iter()
            .filter(|object| self.state[**object].retained_base().1.foretell().is_some())
            .count();
        if bearers < foretold_slots.len()
            || pool.len() < foretold_slots.len() + face_down_slots.len()
        {
            return false;
        }
        let mut foretold = Vec::with_capacity(foretold_slots.len());
        pool.retain(|object| {
            let take = foretold.len() < foretold_slots.len()
                && self.state[*object].retained_base().1.foretell().is_some();
            if take {
                foretold.push(*object);
            }
            !take
        });
        let face_down: Vec<ObjectId> = pool.drain(..face_down_slots.len()).collect();
        let mut exchange = fill_slots(&foretold_slots, &foretold);
        exchange.extend(fill_slots(&face_down_slots, &face_down));
        exchange.retain(|(old, new)| old != new);
        self.state.zones.exchange_face_down_exile(owner, &exchange);
        for grant in &mut self.state.cast_permissions.grants {
            if let Some((_, new)) = exchange.iter().find(|(old, _)| *old == grant.object) {
                grant.object = *new;
            }
        }
        true
    }

    /// Link the exiled arrival `exiled` to `source`: while `player` controls
    /// `source`, they may cast it during their turn as `linked` allows (Hama,
    /// Azula). Each arrival of a melded pair costs its own mana value.
    pub(crate) fn grant_exile_linked(
        &mut self,
        exiled: ObjectId,
        player: PlayerId,
        source: ObjectId,
        linked: LinkedGrant,
    ) {
        let mana_value = self.object_mana_value(exiled);
        self.grant_if_exiled(exiled, |object| {
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

    /// Exile `object` with `entry` (face down per CR 406.3) and add the play
    /// grant `grant` builds for the exiled object. A token, or a card a
    /// replacement kept out of exile, receives no grant.
    fn exile_with_play_grant(
        &mut self,
        object: ObjectId,
        entry: ZoneEntry,
        grant: impl FnOnce(ObjectId) -> PlayGrant,
    ) {
        let exiled = self.move_object_with_entry(object, ZoneType::Exile, entry);
        self.grant_if_exiled(exiled, grant);
    }

    /// Add the grant `grant` builds for the exiled `object_id`. A token, or a
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
            return (!grant.follows_source_controller).then_some(grant.player);
        };
        let permanent = self.lookup_current_permanent(source).ok()?;
        let controller = self.state.permanents[permanent].as_ref()?.controller;
        (grant.follows_source_controller || controller == grant.player).then_some(controller)
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
                && self.grant_condition_holds(grant, player)
                && (!grant.own_turn_only || self.active_player() == player)
                && grant
                    .after_turn
                    .is_none_or(|turn| self.state.turn.turn_number > turn)
        })
    }

    /// What casting `definition` under `grant` shows as its cost: the base
    /// cost, spent with any type of mana when the grant allows it. `None`
    /// when it has no cost to pay (a land is played, not cast).
    pub(crate) fn grant_cast_cost(
        &self,
        grant: &PlayGrant,
        definition: &crate::state::card::PrintedCharacteristics,
    ) -> Option<ManaCost> {
        let cost = self.grant_base_cost(grant, definition)?;
        // Spending permission changes colors, not the announced X cost.
        Some(if grant.any_mana {
            cost.pay_with_any_mana()
        } else {
            cost
        })
    }

    /// The price a viewer reads on `grant`'s card: the grant's cost for its
    /// first spell face, the front unless that is a land and a modal back
    /// face is a spell (CR 712.8a). A card with no spell face has none.
    pub(crate) fn grant_card_cast_cost(&self, grant: &PlayGrant) -> Option<ManaCost> {
        use crate::{cards::content_pack::default_content_pack, state::card::SpellProfile};
        let object = &self.state[grant.object];
        let pack = default_content_pack();
        let face = pack
            .spell_definition(object, SpellProfile::Front)
            .filter(|front| !front.types.is_land())
            .or_else(|| {
                object
                    .modal_back
                    .then(|| pack.spell_definition(object, SpellProfile::BackFace))
                    .flatten()
                    .filter(|back| !back.types.is_land())
            })?;
        self.grant_cast_cost(grant, face)
    }

    /// The mana or alternative cost `grant` selects for `definition`: its
    /// foretell cost, the grant's alternative, or its mana cost. Additional
    /// costs and reductions apply to these colored pips (CR 601.2f) before
    /// any-type spending (CR 601.2h, `Game::effective_spell_cost`).
    pub(super) fn grant_base_cost(
        &self,
        grant: &PlayGrant,
        definition: &crate::state::card::PrintedCharacteristics,
    ) -> Option<ManaCost> {
        if definition.types.is_land() {
            return None;
        }
        if grant.foretold {
            definition.foretell().cloned()
        } else {
            grant.cost.clone().or_else(|| definition.mana_cost.clone())
        }
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

    /// Exiled objects `player` may have a play for, each once: those they
    /// hold grants for, then counter-marked cards while a permanent of theirs
    /// lets them play some. Priority already visits their graveyard, and
    /// `play_routes` decides which routes are open now.
    pub(crate) fn exile_playable_objects(
        &self,
        player: PlayerId,
    ) -> impl Iterator<Item = ObjectId> + '_ {
        let mut seen = std::collections::BTreeSet::new();
        let counter_plays = self.with_layer_board(|board| {
            !board.play_permissions[player.0]
                .exile_counter_plays
                .is_empty()
        });
        let marked = self
            .state
            .object_counters
            .keys()
            .copied()
            .filter(move |object| {
                counter_plays && self.state.zones.zone_of(*object) == Some(ZoneType::Exile)
            });
        self.state
            .cast_permissions
            .grants
            .iter()
            .filter(move |grant| {
                self.grant_player(grant) == Some(player)
                    && self.state.zones.zone_of(grant.object) == Some(ZoneType::Exile)
            })
            .map(|grant| grant.object)
            .chain(marked)
            .filter(move |object| seen.insert(*object))
    }
}

/// Pair each slot with the chosen card filling it: a chosen card already in
/// one of `slots` keeps it, and the other slots take the rest in order.
fn fill_slots(slots: &[ObjectId], chosen: &[ObjectId]) -> Vec<(ObjectId, ObjectId)> {
    let mut incoming = chosen.iter().filter(|object| !slots.contains(object));
    slots
        .iter()
        .map(|old| {
            let new = if chosen.contains(old) {
                *old
            } else {
                *incoming.next().expect("one incoming card per vacated slot")
            };
            (*old, new)
        })
        .collect()
}
