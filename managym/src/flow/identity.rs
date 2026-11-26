// identity.rs
// Exact rules-object identity plus player/game-over helpers.

use crate::{
    agent::action::ActionSpace,
    cards::content_pack::default_content_pack,
    flow::{event::ObjectEventRef, game::Game},
    state::{
        card::{CardDefId, ObjectDefinition},
        game_object::{
            BattlefieldFacts, ObjectId, ObjectLki, ObjectLookupError, PermanentId, PlayerId,
            PublicId,
        },
        zone::ZoneType,
    },
};

/// How a face-down permanent showing no name is called in human labels.
pub(crate) const FACE_DOWN_NAME: &str = "a face-down permanent";
/// How a face-up permanent with no name (a copy of a manifest) is called.
pub(crate) const NAMELESS_PERMANENT_NAME: &str = "a nameless permanent";

/// Exact public reference and the storage slot a decision or event may cite.
pub(crate) struct ObservedIdentity {
    pub reference: ObjectEventRef,
    pub visible_slot: Option<u32>,
}

impl Game {
    /// The object a live battlefield permanent is.
    pub fn permanent_object(&self, permanent_id: PermanentId) -> Option<ObjectId> {
        let object = self.state.permanents.get(permanent_id.0)?.as_ref()?.object;
        (self.lookup_current_permanent(object) == Ok(permanent_id)).then_some(object)
    }

    /// Resolve an object to its battlefield permanent. Never follows the
    /// physical card to a later appearance.
    pub fn lookup_current_permanent(
        &self,
        object: ObjectId,
    ) -> Result<PermanentId, ObjectLookupError> {
        self.state.object(object)?;
        self.state
            .object_to_permanent
            .get(object.slot())
            .copied()
            .flatten()
            .filter(|id| {
                self.state.zones.zone_of(object) == Some(ZoneType::Battlefield)
                    && self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| permanent.object == object)
            })
            .ok_or(ObjectLookupError::WrongZone)
    }

    /// The definition `viewer` may learn for a battlefield permanent: what
    /// `observed_source_definition` publishes, plus the printing behind a
    /// face-down permanent for a viewer who may look at it (CR 708.5) or
    /// while turning it face up shows it.
    pub(crate) fn observed_permanent_definition(
        &self,
        viewer: PlayerId,
        permanent: &crate::state::permanent::Permanent,
    ) -> Option<CardDefId> {
        let object = &self.state[permanent.object];
        if !object.is_face_down() {
            return self.observed_source_definition(permanent.object);
        }
        // Announcing the special action shows the creature and its cost
        // before the controller chooses which mana to spend.
        use super::decision::{FrameCompletion, ResolutionRef};
        let showing = self.state.suspended_decision.as_ref().is_some_and(|pending| {
            pending.continuation().into_iter().any(|part| matches!(part, ResolutionRef::Frame(frame)
                if matches!(frame.completion, FrameCompletion::TurnFaceUp(id) if id == permanent.object)))
        });
        (showing || self.shares_information(viewer, permanent.controller))
            .then(|| object.retained_base().0)
    }

    /// A permanent's alias survives departure in last-known information. Each
    /// manifest entry mints a fresh alias, so incarnation zero identifies it
    /// without disclosing the physical card's previous zone changes (CR 708.2).
    pub(crate) fn observed_identity(&self, object: ObjectId) -> Option<ObservedIdentity> {
        let (entity, manifested) = if let Ok(id) = self.lookup_current_permanent(object) {
            let permanent = self.state.permanents[id].as_ref()?;
            (permanent.id, permanent.manifested)
        } else if let Some(lki) = self.object_lki(object) {
            (lki.presentation_id, lki.battlefield.manifested)
        } else {
            (self.state.objects.provenance(object)?.0, false)
        };
        let hides_card = manifested || self.observed_face_down(object);
        Some(ObservedIdentity {
            reference: ObjectEventRef {
                entity,
                incarnation: if hides_card { 0 } else { object.generation() },
            },
            visible_slot: (!hides_card).then_some(object.slot() as u32),
        })
    }

    pub(crate) fn hidden_card_alias(&self, object: ObjectId) -> Option<ObjectEventRef> {
        let identity = self.observed_identity(object)?;
        identity
            .visible_slot
            .is_none()
            .then_some(identity.reference)
    }

    pub(crate) fn observed_object_id(&self, object: ObjectId) -> PublicId {
        if let Some(alias) = self.hidden_card_alias(object) {
            return alias.entity;
        }
        self.state
            .objects
            .provenance(object)
            .expect("object provenance")
            .0
    }

    /// The catalog definition a viewer may learn for a source object, live or
    /// departed: its last shown face, never the printing behind a face-down
    /// permanent (`ANONYMOUS` gives `None`).
    pub(crate) fn observed_source_definition(&self, object: ObjectId) -> Option<CardDefId> {
        if self.state.live(object).is_some() {
            return self.state[object].definition_id.catalog_id();
        }
        match self.object_lki(object) {
            Some(lki) => lki.definition_id.catalog_id(),
            None => self.state.source_base(object).0.catalog_id(),
        }
    }

    /// Whether an object is face down now, or was when it left the
    /// battlefield (CR 708.2).
    pub(crate) fn observed_face_down(&self, object: ObjectId) -> bool {
        if self.state.live(object).is_some() {
            self.state[object].is_face_down()
        } else {
            self.object_lki(object)
                .is_some_and(|lki| lki.battlefield.face_down)
        }
    }

    /// How a label names an object that shows no name.
    pub(crate) fn unnamed_label(&self, object: ObjectId) -> &'static str {
        if self.observed_face_down(object) {
            FACE_DOWN_NAME
        } else if let Some(label) = self
            .observed_source_definition(object)
            .and_then(|id| default_content_pack().definition(id)?.label)
        {
            label
        } else {
            NAMELESS_PERMANENT_NAME
        }
    }

    /// The name a viewer may read for an object: its live effective name,
    /// else its last effective battlefield name once departed, else the name
    /// of the definition it shows. A face-down permanent has no name unless an
    /// effect gives it one (Witness Protection); neither has a face-up copy of
    /// one, which reads as a nameless permanent.
    pub(crate) fn observed_object_name(&self, object: ObjectId) -> String {
        let name = if let Ok(permanent) = self.lookup_current_permanent(object) {
            Some(self.characteristics(permanent).name.clone())
        } else if self.state.live(object).is_none() {
            self.object_lki(object)
                .map(|lki| lki.battlefield.name.clone())
        } else {
            None
        };
        name.filter(|name| !name.is_empty())
            .or_else(|| {
                let definition = self.observed_source_definition(object)?;
                default_content_pack()
                    .definition(definition)
                    .map(|row| row.registry_name().to_owned())
            })
            .unwrap_or_else(|| self.unnamed_label(object).to_owned())
    }

    /// Snapshot the current battlefield object for triggers and LKI.
    pub(crate) fn snapshot_current_permanent(&self, object: ObjectId) -> Option<ObjectLki> {
        let permanent_id = self.lookup_current_permanent(object).ok()?;
        let permanent = self.state.permanents[permanent_id].as_ref()?;
        let definition = &self.state[object];
        let characteristics = self.characteristics(permanent_id);
        Some(ObjectLki {
            copiable: None,
            transforms: permanent.transforms,
            inherited_from: self.inherited_definition_id(permanent_id),
            granted_abilities: characteristics
                .triggered_abilities
                .iter()
                .skip(if characteristics.abilities_lost {
                    0
                } else {
                    definition.abilities.len()
                })
                .cloned()
                .collect(),
            keywords: characteristics.keywords,
            creature_types: characteristics.creature_types.clone(),
            permanent: permanent_id,
            battlefield: std::sync::Arc::new(BattlefieldFacts {
                types: characteristics.types.clone(),
                subtypes: characteristics.subtypes.clone(),
                supertypes: definition.supertypes.clone(),
                colors: characteristics.colors.clone(),
                name: characteristics.name.clone(),
                mana_value: self.object_mana_value(object),
                is_legendary: characteristics.is_legendary,
                token: !definition.is_card(),
                attacking: permanent.attacking,
                in_combat: self.permanent_in_combat(permanent_id),
                tapped: permanent.tapped,
                manifested: permanent.manifested,
                face_down: definition.is_face_down(),
            }),
            attached_to: permanent.attached_to,
            meld_partner: None,
            abilities_lost: characteristics.abilities_lost,
            dealt_damage_this_turn: permanent.dealt_damage_this_turn,
            power: characteristics.power,
            toughness: characteristics.toughness,
            counters: permanent.counters.clone(),
            object,
            from_zone: ZoneType::Battlefield,
            owner: definition.owner,
            controller: permanent.controller,
            definition_id: definition.definition_id,
            presentation_id: permanent.id,
        })
    }

    pub fn object_lki(&self, object_ref: ObjectId) -> Option<&ObjectLki> {
        self.state.object_lki.get(object_ref)
    }

    /// Last known information of the object that occupied a battlefield
    /// slot, once it has left.
    pub(crate) fn departed_permanent_lki(&self, permanent_id: PermanentId) -> Option<&ObjectLki> {
        self.state.object_lki.for_permanent(permanent_id)
    }

    pub(crate) fn object_event_ref(&self, object: ObjectId) -> Option<ObjectEventRef> {
        Some(self.observed_identity(object)?.reference)
    }

    pub(crate) fn permanent_event_ref(&self, permanent_id: PermanentId) -> Option<ObjectEventRef> {
        self.object_event_ref(self.permanent_object(permanent_id)?)
    }

    /// Resolve a departed object's immutable meaning through the match's
    /// shared content pack by the final shown definition id.
    pub fn object_lki_definition(&self, object_ref: ObjectId) -> Option<&ObjectDefinition> {
        let lki = self.object_lki(object_ref)?;
        default_content_pack().definition(lki.definition_id)
    }

    pub(crate) fn record_object_lki(&mut self, lki: ObjectLki) {
        self.state.object_lki_mut().insert(lki);
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

    /// Public schedules in creation order, including overridden controls.
    pub fn player_control_views(&self) -> Vec<crate::experience::PlayerControlView> {
        self.state
            .turn
            .player_controls
            .iter()
            .enumerate()
            .map(|(index, control)| crate::experience::PlayerControlView {
                player: control.player.0 as u8,
                controller: control.controller.0 as u8,
                whole_turn: control.duration
                    == crate::state::ability::PlayerControlDuration::NextTurn,
                active: control.active,
                effective: control.active
                    && !self.state.turn.player_controls[index + 1..]
                        .iter()
                        .any(|later| later.active && later.player == control.player),
            })
            .collect()
    }

    /// Who chooses for `player`. Actions and costs still belong to `player`.
    pub fn decision_controller(&self, player: PlayerId) -> PlayerId {
        self.state
            .turn
            .player_controls
            .iter()
            .rev()
            .find(|control| control.player == player && control.active)
            .map_or(player, |control| control.controller)
    }

    /// A controller sees what the controlled player may see in the game.
    /// Outside-game cards are deliberately excluded at their own authority.
    pub(crate) fn shares_information(&self, viewer: PlayerId, player: PlayerId) -> bool {
        viewer == player || self.decision_controller(player) == viewer
    }

    pub(crate) fn remember_controlled_hands(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            if self.decision_controller(player) != player {
                self.remember_hand(player);
            }
        }
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
        self.state.is_game_over()
    }

    pub fn winner_index(&self) -> Option<usize> {
        use super::game::{GameOutcome, GameProcedure};
        match self.state.procedure {
            GameProcedure::Finished(GameOutcome::Victory { winner }) => Some(winner.0),
            GameProcedure::Finished(GameOutcome::Losses { losses }) => {
                losses.iter().position(|loss| !loss.any())
            }
            _ => None,
        }
    }
}
