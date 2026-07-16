// identity.rs
// Exact rules-object identity plus player/game-over helpers.

use crate::{
    agent::action::ActionSpace,
    flow::{event::ObjectEventRef, game::Game},
    state::{
        card::CardDefinition,
        game_object::{
            CardId, EntityId, ObjectLki, ObjectLookupError, ObjectRef, PermanentId, PlayerId,
        },
        zone::ZoneType,
    },
};

impl Game {
    /// Exact identity of the object currently represented by `card` in a
    /// zone. Cards removed from all zones have no current rules object.
    pub fn current_object_ref(&self, card: CardId) -> Option<ObjectRef> {
        self.state.zones.zone_of(card)?;
        let incarnation = *self.state.object_incarnations.get(card.0)?;
        Some(ObjectRef {
            entity: EntityId::from(card),
            incarnation,
        })
    }

    /// Exact identity of a live battlefield permanent.
    pub fn permanent_object_ref(&self, permanent_id: PermanentId) -> Option<ObjectRef> {
        let permanent = self.state.permanents.get(permanent_id.0)?.as_ref()?;
        let object_ref = self.current_object_ref(permanent.card)?;
        (self.lookup_current_permanent(object_ref) == Ok(permanent_id)).then_some(object_ref)
    }

    /// Resolve an exact ref to the current battlefield representation.
    /// Never follows the physical card into a newer incarnation.
    pub fn lookup_current_permanent(
        &self,
        object_ref: ObjectRef,
    ) -> Result<PermanentId, ObjectLookupError> {
        let card = CardId::from(object_ref.entity);
        let Some(current_incarnation) = self.state.object_incarnations.get(card.0).copied() else {
            return Err(ObjectLookupError::MissingEntity);
        };
        if current_incarnation != object_ref.incarnation {
            return Err(ObjectLookupError::StaleIncarnation);
        }
        if self.state.zones.zone_of(card) != Some(ZoneType::Battlefield) {
            return Err(ObjectLookupError::WrongZone);
        }
        let permanent_id = self
            .state
            .card_to_permanent
            .get(card.0)
            .copied()
            .flatten()
            .ok_or(ObjectLookupError::WrongZone)?;
        let permanent = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(Option::as_ref)
            .ok_or(ObjectLookupError::WrongZone)?;
        if permanent.card != card {
            return Err(ObjectLookupError::WrongZone);
        }
        Ok(permanent_id)
    }

    /// Snapshot the current battlefield object for triggers and LKI.
    pub(crate) fn snapshot_current_permanent(&self, card: CardId) -> Option<ObjectLki> {
        let object_ref = self.current_object_ref(card)?;
        let permanent_id = self.lookup_current_permanent(object_ref).ok()?;
        let permanent = self.state.permanents[permanent_id].as_ref()?;
        let definition = &self.state.cards[card];
        Some(ObjectLki {
            object_ref,
            card,
            from_zone: ZoneType::Battlefield,
            owner: definition.owner,
            controller: permanent.controller,
            definition_id: definition.definition_id,
            presentation_id: permanent.id,
        })
    }

    pub fn object_lki(&self, object_ref: ObjectRef) -> Option<&ObjectLki> {
        self.state.object_lki.get(&object_ref)
    }

    /// Capture the stable viewer identity for a current or departed object.
    pub(crate) fn object_event_ref(&self, object_ref: ObjectRef) -> Option<ObjectEventRef> {
        let entity = if let Ok(permanent_id) = self.lookup_current_permanent(object_ref) {
            self.state.permanents[permanent_id].as_ref()?.id
        } else if let Some(lki) = self.object_lki(object_ref) {
            lki.presentation_id
        } else {
            self.state.cards.get(CardId::from(object_ref.entity).0)?.id
        };
        Some(ObjectEventRef {
            entity,
            incarnation: object_ref.incarnation,
        })
    }

    pub(crate) fn permanent_event_ref(&self, permanent_id: PermanentId) -> Option<ObjectEventRef> {
        self.object_event_ref(self.permanent_object_ref(permanent_id)?)
    }

    /// Resolve a departed object's immutable meaning through the match's
    /// shared content pack. LKI stores only `CardDefId`, never a definition
    /// clone.
    pub fn object_lki_definition(&self, object_ref: ObjectRef) -> Option<&CardDefinition> {
        let lki = self.object_lki(object_ref)?;
        self.state.content.definition(lki.definition_id)
    }

    pub(crate) fn record_object_lki(&mut self, lki: ObjectLki) {
        self.state.object_lki.insert(lki.object_ref, lki);
    }

    pub(crate) fn advance_object_incarnation(&mut self, card: CardId) {
        let incarnation = self
            .state
            .object_incarnations
            .get_mut(card.0)
            .expect("card entity must have an incarnation");
        *incarnation = incarnation
            .checked_next()
            .expect("rules object incarnation overflow");
    }

    pub fn action_space(&self) -> Option<&ActionSpace> {
        self.current_action_space.as_ref()
    }

    pub fn active_player(&self) -> PlayerId {
        self.state.turn.active_player
    }

    pub fn non_active_player(&self) -> PlayerId {
        PlayerId((self.state.turn.active_player.0 + 1) % 2)
    }

    pub fn agent_player(&self) -> PlayerId {
        self.current_action_space
            .as_ref()
            .and_then(|space| space.player)
            .unwrap_or(PlayerId(0))
    }

    pub fn players_starting_with_active(&self) -> [PlayerId; 2] {
        [self.active_player(), self.non_active_player()]
    }

    pub fn players_starting_with_agent(&self) -> [PlayerId; 2] {
        let agent = self.agent_player();
        [agent, PlayerId((agent.0 + 1) % 2)]
    }

    pub fn next_player(&self, player: PlayerId) -> PlayerId {
        PlayerId((player.0 + 1) % 2)
    }

    pub fn is_active_player(&self, player: PlayerId) -> bool {
        player == self.active_player()
    }

    pub fn is_game_over(&self) -> bool {
        self.state.players.iter().filter(|p| p.alive).count() < 2
    }

    pub fn winner_index(&self) -> Option<usize> {
        if !self.is_game_over() {
            return None;
        }
        self.state.players.iter().position(|p| p.alive)
    }
}
