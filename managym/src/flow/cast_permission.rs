// cast_permission.rs
// Permissions to cast exact cards outside their owner's hand, and the
// prohibitions that override them.
//
// `CastPermissions` lives on `GameState`. A grant names the exact exiled
// object (CR 400.7): once that card leaves exile it is a new object and the
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
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId},
        mana::ManaCost,
        zone::ZoneType,
    },
};

/// `player` may play one exact exiled object: cast it (for `cost` rather
/// than its mana cost, when set) or, for a land, play it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ExileCastGrant {
    pub object: ObjectRef,
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
    /// The grant lasts while `player` controls this battlefield object
    /// ("for as long as you control Hama", "cards exiled with Azula").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ObjectRef>,
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
}

impl ExileCastGrant {
    /// A grant with no further terms: cast for the card's mana cost, any time
    /// its timing allows.
    pub fn new(object: ObjectRef, player: PlayerId) -> Self {
        Self {
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
    pub exile_grants: Vec<ExileCastGrant>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hand_only: Vec<HandOnly>,
}

impl CastPermissions {
    pub fn is_empty(&self) -> bool {
        self.exile_grants.is_empty() && self.hand_only.is_empty()
    }
}

/// Airbend's alternative cost (CR 701.65a).
const AIRBEND_COST: &str = "{2}";

/// The cost of the foretell special action (CR 702.143a).
pub(crate) const FORETELL_ACTION_COST: &str = "{2}";

impl Game {
    /// Airbend the permanent: exile it and, unless it was a token, let its
    /// owner cast that exiled card for {2}.
    pub(crate) fn airbend_permanent(&mut self, permanent_id: PermanentId) {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return;
        };
        let object_id = permanent.object;
        let owner = self.state.objects[object_id].owner;
        self.exile_with_play_grant(object_id, |object| ExileCastGrant {
            cost: Some(ManaCost::parse(AIRBEND_COST)),
            ..ExileCastGrant::new(object, owner)
        });
    }

    /// Airbend a spell (CR 701.65a): it leaves the stack for exile, and its
    /// owner may cast that card for {2}; a copy ceases to exist (CR 707.10a).
    pub(crate) fn airbend_spell(&mut self, object_id: ObjectId) {
        let owner = self.state.objects[object_id].owner;
        self.exile_with_play_grant(object_id, |object| ExileCastGrant {
            cost: Some(ManaCost::parse(AIRBEND_COST)),
            ..ExileCastGrant::new(object, owner)
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
            self.exile_with_play_grant(object_id, |object| ExileCastGrant {
                this_turn: until == PlayWindow::ThisTurn,
                through_next_turn: (until == PlayWindow::ThroughYourNextTurn).then_some(turn),
                ..ExileCastGrant::new(object, player)
            });
        }
    }

    /// Whether `player` may foretell `card` now: a card with foretell in
    /// their hand, during their turn, with {2} available (CR 702.143a).
    pub(crate) fn can_foretell(&self, player: PlayerId, card: CardId) -> bool {
        self.active_player() == player
            && self.state.objects[card].foretell().is_some()
            && self.state.zones.contains(card, ZoneType::Hand, player)
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
        self.move_object_with_entry(
            card,
            ZoneType::Exile,
            ZoneEntry {
                face_down: true,
                ..ZoneEntry::default()
            },
        );
        self.grant_if_exiled(card, |object| ExileCastGrant {
            after_turn: Some(turn),
            foretold: true,
            ..ExileCastGrant::new(object, player)
        });
        self.restart_priority_round();
    }

    /// Whether `card` is a foretold card lying face down in exile, hidden
    /// from everyone but its owner.
    pub(crate) fn is_foretold(&self, object: impl Into<ObjectId>) -> bool {
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
        let slots: Vec<ObjectId> = self.foretold_cards(player).map(ObjectId::from).collect();
        let bearers = pool
            .iter()
            .filter(|object_id| self.state.objects[**object_id].foretell().is_some())
            .count();
        if bearers < slots.len() {
            return false;
        }
        let mut chosen = Vec::with_capacity(slots.len());
        pool.retain(|object_id| {
            let take =
                chosen.len() < slots.len() && self.state.objects[*object_id].foretell().is_some();
            if take {
                chosen.push(*object_id);
            }
            !take
        });
        let mut incoming = chosen.iter().filter(|object_id| !slots.contains(object_id));
        for &old in slots.iter().filter(|old| !chosen.contains(old)) {
            let new = *incoming.next().expect("one incoming card per vacated slot");
            let from = self
                .current_object_ref(old)
                .expect("a foretold card is exiled");
            let to = self
                .current_object_ref(new)
                .expect("an unseen card is hidden");
            self.state.zones.replace_face_down_exile(player, old, new);
            for grant in &mut self.state.cast_permissions.exile_grants {
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
            .filter(|card| self.is_foretold(*card))
    }

    /// Exile `card` linked to `source`: while `player` controls `source`,
    /// they may cast it during their turn as `linked` allows (Hama, Azula).
    /// Without a source the card is only exiled.
    pub(crate) fn exile_linked(
        &mut self,
        object_id: ObjectId,
        player: PlayerId,
        source: Option<ObjectRef>,
        linked: LinkedGrant,
    ) {
        let Some(source) = source else {
            self.move_object(object_id, ZoneType::Exile);
            return;
        };
        let mana_value = self.object_mana_value(object_id);
        self.exile_with_play_grant(object_id, |object| {
            let base = ExileCastGrant {
                source: Some(source),
                own_turn_only: true,
                ..ExileCastGrant::new(object, player)
            };
            match linked {
                LinkedGrant::WaterbendManaValue => ExileCastGrant {
                    cost: Some(ManaCost::default().with_generic(mana_value)),
                    waterbend: true,
                    ..base
                },
                LinkedGrant::FlashAnyMana => ExileCastGrant {
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
        grant: impl FnOnce(ObjectRef) -> ExileCastGrant,
    ) {
        self.move_object(object, ZoneType::Exile);
        self.grant_if_exiled(object, grant);
    }

    /// Add the grant `grant` builds for `card`'s exiled object. A token, or a
    /// card a replacement kept out of exile, receives no grant.
    fn grant_if_exiled(
        &mut self,
        object_id: impl Into<ObjectId>,
        grant: impl FnOnce(ObjectRef) -> ExileCastGrant,
    ) {
        let object_id: ObjectId = object_id.into();
        if self.state.zones.zone_of(object_id) != Some(ZoneType::Exile)
            || !self.state.objects[object_id].is_card()
        {
            return;
        }
        if let Some(object) = self.current_object_ref(object_id) {
            let grant = grant(object);
            self.state.cast_permissions.exile_grants.push(grant);
        }
    }

    /// "This turn" grants end in the cleanup step (CR 514.2), and "until the
    /// end of your next turn" ones in that of the permitted player's next
    /// turn.
    pub(crate) fn expire_turn_cast_grants(&mut self) {
        let (active, turn) = (self.active_player(), self.state.turn.turn_number);
        self.state.cast_permissions.exile_grants.retain(|grant| {
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

    /// The grants naming this exact card while it is in exile. Producers
    /// exile a new object for every grant, so a card holds at most one.
    fn grants_naming(
        &self,
        object_id: impl Into<ObjectId>,
    ) -> impl Iterator<Item = &ExileCastGrant> {
        let object_id: ObjectId = object_id.into();
        let object = (self.state.zones.zone_of(object_id) == Some(ZoneType::Exile))
            .then(|| self.current_object_ref(object_id))
            .flatten();
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .filter(move |grant| Some(grant.object) == object)
    }

    /// The live grant letting `player` play this exact exiled card now.
    /// Another player's card is castable too; a permanent spell enters
    /// under its caster's control (CR 110.2).
    pub(crate) fn exile_grant(
        &self,
        player: PlayerId,
        object: impl Into<ObjectId>,
    ) -> Option<&ExileCastGrant> {
        self.grants_naming(object).find(|grant| {
            grant.player == player
                && (!grant.own_turn_only || self.active_player() == player)
                && grant
                    .after_turn
                    .is_none_or(|turn| self.state.turn.turn_number > turn)
                && grant.source.is_none_or(|source| {
                    self.lookup_current_permanent(source).is_ok_and(|id| {
                        self.state.permanents[id]
                            .as_ref()
                            .is_some_and(|permanent| permanent.controller == player)
                    })
                })
        })
    }

    /// What casting `card` under `grant` costs: its foretell cost, the
    /// grant's alternative, or its mana cost. `None` when it has no cost to
    /// pay (a land is played, not cast).
    pub(crate) fn grant_cast_cost(
        &self,
        grant: &ExileCastGrant,
        object: ObjectId,
    ) -> Option<ManaCost> {
        let definition = &self.state.objects[object];
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
            ManaCost {
                x: cost.x,
                ..ManaCost::default().with_generic(cost.mana_value)
            }
        } else {
            cost
        })
    }

    /// The permission naming this exact exiled card, whatever its timing,
    /// for viewers to inspect. A grant tied to a source that has left the
    /// battlefield can never be used again and is not shown.
    pub(crate) fn exile_grant_on(&self, object: impl Into<ObjectId>) -> Option<&ExileCastGrant> {
        self.grants_naming(object).find(|grant| {
            grant
                .source
                .is_none_or(|source| self.lookup_current_permanent(source).is_ok())
        })
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

    /// Whether `player` may play `card` from where it is now: their hand,
    /// a graveyard casting cost, or an exile grant. A granted land play is
    /// not a cast, so a hand-only restriction does not stop it.
    pub(crate) fn may_play_from_zone(&self, player: PlayerId, object: impl Into<ObjectId>) -> bool {
        let object: ObjectId = object.into();
        if self.state.zones.contains(object, ZoneType::Hand, player) {
            return true;
        }
        if self.state.objects[object].types.is_land() {
            return self.exile_grant(player, object).is_some();
        }
        self.zone_cast_cost(player, object).is_some()
    }

    /// Cards named by `player`'s exile grants, stale ones included; callers
    /// check [`Game::may_play_from_zone`].
    pub(crate) fn exile_granted_cards(&self, player: PlayerId) -> Vec<CardId> {
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .filter(|grant| grant.player == player)
            .filter_map(|grant| self.state.card_id(grant.object.entity.into()))
            .collect()
    }
}
