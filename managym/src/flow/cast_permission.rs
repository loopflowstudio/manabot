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

use crate::{
    flow::game::Game,
    state::{
        game_object::{CardId, ObjectRef, PermanentId, PlayerId},
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

impl Game {
    /// Airbend the permanent: exile it and, unless it was a token, let its
    /// owner cast that exiled card for {2}.
    pub(crate) fn airbend_permanent(&mut self, permanent_id: PermanentId) {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return;
        };
        let card = permanent.card;
        self.move_card(card, ZoneType::Exile);
        if self.state.zones.zone_of(card) != Some(ZoneType::Exile)
            || self.state.cards[card].is_token
        {
            return;
        }
        let owner = self.state.cards[card].owner;
        self.grant_exile_play(card, owner, Some(ManaCost::parse(AIRBEND_COST)), false);
    }

    /// "Exile the top card of your library. You may play that card this turn."
    pub(crate) fn exile_top_card_play_this_turn(&mut self, player: PlayerId) {
        let Some(card) = self.state.zones.top(ZoneType::Library, player) else {
            return;
        };
        self.move_card(card, ZoneType::Exile);
        if self.state.zones.zone_of(card) == Some(ZoneType::Exile) {
            self.grant_exile_play(card, player, None, true);
        }
    }

    fn grant_exile_play(
        &mut self,
        card: CardId,
        player: PlayerId,
        cost: Option<ManaCost>,
        this_turn: bool,
    ) {
        let Some(object) = self.current_object_ref(card) else {
            return;
        };
        self.journal_cast_permissions();
        self.state
            .cast_permissions
            .exile_grants
            .push(ExileCastGrant {
                object,
                player,
                cost,
                this_turn,
            });
    }

    /// "This turn" grants end in the cleanup step (CR 514.2).
    pub(crate) fn expire_turn_cast_grants(&mut self) {
        if self
            .state
            .cast_permissions
            .exile_grants
            .iter()
            .any(|grant| grant.this_turn)
        {
            self.journal_cast_permissions();
            self.state
                .cast_permissions
                .exile_grants
                .retain(|grant| !grant.this_turn);
        }
    }

    /// "Until your next turn, your opponents can't cast spells from anywhere
    /// other than their hands."
    pub(crate) fn restrict_opponent_to_hand(&mut self, controller: PlayerId) {
        self.journal_cast_permissions();
        self.state.cast_permissions.hand_only.push(HandOnly {
            player: PlayerId(1 - controller.0),
            until_turn_of: controller,
        });
    }

    /// Restrictions lasting "until your next turn" end as that turn begins.
    pub(crate) fn expire_cast_restrictions(&mut self, active: PlayerId) {
        if self
            .state
            .cast_permissions
            .hand_only
            .iter()
            .any(|restriction| restriction.until_turn_of == active)
        {
            self.journal_cast_permissions();
            self.state
                .cast_permissions
                .hand_only
                .retain(|restriction| restriction.until_turn_of != active);
        }
    }

    pub(crate) fn casts_only_from_hand(&self, player: PlayerId) -> bool {
        self.state
            .cast_permissions
            .hand_only
            .iter()
            .any(|restriction| restriction.player == player)
    }

    /// The live grant letting `player` play this exact exiled card.
    fn exile_grant(&self, player: PlayerId, card: CardId) -> Option<&ExileCastGrant> {
        if self.state.zones.zone_of(card) != Some(ZoneType::Exile) {
            return None;
        }
        let object = self.current_object_ref(card)?;
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .find(|grant| grant.object == object && grant.player == player)
    }

    /// The cost `player` may pay to cast `card` from exile. Hand-only
    /// restrictions stop casting, not land plays.
    pub(crate) fn exile_cast_cost(&self, player: PlayerId, card: CardId) -> Option<ManaCost> {
        if self.casts_only_from_hand(player) {
            return None;
        }
        let grant = self.exile_grant(player, card)?;
        grant
            .cost
            .clone()
            .or_else(|| self.state.cards[card].mana_cost.clone())
    }

    /// Whether `player` may play `card` from where it is now: their hand,
    /// a graveyard casting cost, or an exile grant (lands included).
    pub(crate) fn may_play_from_zone(&self, player: PlayerId, card: CardId) -> bool {
        if self.state.zones.contains(card, ZoneType::Hand, player) {
            return true;
        }
        if self.state.cards[card].types.is_land() {
            return self.exile_grant(player, card).is_some();
        }
        self.zone_cast_cost(player, card).is_some()
    }

    /// Exiled cards `player` may currently play.
    pub(crate) fn playable_exiled_cards(&self, player: PlayerId) -> Vec<CardId> {
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .filter(|grant| grant.player == player)
            .map(|grant| CardId::from(grant.object.entity))
            .filter(|card| self.may_play_from_zone(player, *card))
            .collect()
    }
}
