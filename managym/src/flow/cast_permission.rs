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

/// "Its owner may cast it for `cost` rather than its mana cost" for one
/// exact exiled object.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ExileCastGrant {
    pub object: ObjectRef,
    pub player: PlayerId,
    pub cost: ManaCost,
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
        let Some(object) = self.current_object_ref(card) else {
            return;
        };
        self.journal_cast_permissions();
        self.state
            .cast_permissions
            .exile_grants
            .push(ExileCastGrant {
                object,
                player: self.state.cards[card].owner,
                cost: ManaCost::parse(AIRBEND_COST),
            });
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

    /// The cost `player` may pay to cast `card` from exile, if a live grant
    /// names this exact exiled object.
    pub(crate) fn exile_cast_cost(&self, player: PlayerId, card: CardId) -> Option<ManaCost> {
        if self.state.zones.zone_of(card) != Some(ZoneType::Exile)
            || self.casts_only_from_hand(player)
        {
            return None;
        }
        let object = self.current_object_ref(card)?;
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .find(|grant| grant.object == object && grant.player == player)
            .map(|grant| grant.cost.clone())
    }

    /// Exiled cards `player` may currently cast.
    pub(crate) fn castable_exiled_cards(&self, player: PlayerId) -> Vec<CardId> {
        if self.casts_only_from_hand(player) {
            return Vec::new();
        }
        self.state
            .cast_permissions
            .exile_grants
            .iter()
            .filter(|grant| grant.player == player)
            .map(|grant| CardId::from(grant.object.entity))
            .filter(|card| self.exile_cast_cost(player, *card).is_some())
            .collect()
    }
}
