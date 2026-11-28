//! Counterfactual edits at a live decision. A batch resolves observed ids
//! against the original position and applies each edit to a rollback snapshot.
//! Quiet moves record card facts without running replacements or rules events.

use std::collections::BTreeSet;

use crate::{
    agent::{
        action::{ActionSpaceKind, AgentError},
        policy_history,
    },
    cards::content_pack::default_content_pack,
    flow::{
        combat::DeclarationKind, game::Game, proposed_event::ZoneEntry,
        trigger::DelayedTriggerEvent,
    },
    state::{
        card::CardDefId,
        game_object::{ObjectId, PermanentId, PlayerId},
        knowledge::LibraryKnowledge,
        permanent::CounterKind,
        zone::ZoneType,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermanentDestination {
    Graveyard,
    Exile,
    LibraryBottom,
}

/// An edit to a live game, addressed by observation ids, native seats or catalog names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GameStateEdit {
    MovePermanent {
        permanent: i32,
        to: PermanentDestination,
    },
    SetTapped {
        permanent: i32,
        tapped: bool,
    },
    SetDamage {
        permanent: i32,
        damage: i32,
    },
    SetCounters {
        permanent: i32,
        // The journal tag already uses "kind".
        counter: CounterKind,
        count: i32,
    },
    SetLife {
        player: usize,
        life: i32,
    },
    CreateToken {
        key: String,
        controller: usize,
        ready: bool,
        tapped: bool,
    },
    AddToBattlefield {
        card: i32,
        controller: Option<usize>,
        ready: bool,
        tapped: bool,
        attached_to: Option<i32>,
    },
    AddFromLibrary {
        player: usize,
        name: String,
        controller: Option<usize>,
        ready: bool,
        tapped: bool,
        attached_to: Option<i32>,
    },
}

fn edit_player(player: usize, seats: usize) -> Result<PlayerId, AgentError> {
    if player >= seats {
        return Err(AgentError(format!(
            "game state edits: invalid player {player}"
        )));
    }
    Ok(PlayerId(player))
}

pub(crate) fn validate_edit_token(key: &str) -> Result<(), AgentError> {
    if default_content_pack().token_id(key).is_none() {
        return Err(AgentError(format!(
            "game state edits: unknown token key {key:?}"
        )));
    }
    Ok(())
}

impl Game {
    /// Rebuild the same live decision without executing actions or running SBAs.
    pub(crate) fn edit_game_state(&mut self, edits: &[GameStateEdit]) -> Result<(), AgentError> {
        if self.is_game_over() {
            return Err(AgentError("game state edits: game is over".into()));
        }
        if self.state.opening().is_some()
            || self.pending_choice.is_some()
            || self.state.suspended_decision.is_some()
        {
            return Err(AgentError(
                "game state edits require a decision outside opening and suspended choices".into(),
            ));
        }
        let space = self
            .action_space()
            .ok_or_else(|| AgentError("game state edits require a live decision".into()))?;
        let declaration = space.declarer();
        if space.actions.is_empty()
            || (declaration.is_none() && space.kind != ActionSpaceKind::Priority)
        {
            return Err(AgentError(
                "game state edits require Priority, DeclareAttacker or DeclareBlocker".into(),
            ));
        }
        if edits.is_empty() {
            return Ok(());
        }
        let mut edited = self.with_game_state_edits(edits)?;
        if let Some((role, player, focal)) = declaration {
            let completed = self
                .state
                .combat
                .as_ref()
                .expect("declaration has combat")
                .completed_declarers(role);
            edited.refresh_declaration(role, player, focal, completed)?;
        } else {
            let player = space.actions[0].player();
            edited.refresh_priority_actions(player);
        }
        *self = edited;
        Ok(())
    }

    fn refresh_declaration(
        &mut self,
        role: DeclarationKind,
        player: PlayerId,
        focal: PermanentId,
        mut completed_before: BTreeSet<PermanentId>,
    ) -> Result<(), AgentError> {
        let mut eligible = match role {
            DeclarationKind::Attackers => self.eligible_attackers(player),
            DeclarationKind::Blockers => self.eligible_blockers(player),
        };
        if !eligible.contains(&focal) {
            return Err(AgentError(
                "game state edits made the current declarer ineligible".into(),
            ));
        }
        let combat = self.state.combat.as_ref().expect("declaration has combat");
        let completed = combat.completed_declarers(role);
        // A blocker has still answered even if its attacker left combat.
        completed_before
            .retain(|id| !completed.contains(id) && self.permanent_object(*id).is_some());
        let combat = self.state.combat.as_mut().expect("declaration has combat");
        combat.detached_declarers.extend(completed_before);
        // An edit invalidates earlier interchangeability assumptions: keep
        // declined creatures answered without suppressing remaining copies.
        let declined = match role {
            DeclarationKind::Attackers => &mut combat.declined_attackers,
            DeclarationKind::Blockers => &mut combat.declined_blockers,
        };
        combat.detached_declarers.extend(declined.drain(..));
        eligible.retain(|id| {
            *id != focal && !completed.contains(id) && !combat.detached_declarers.contains(id)
        });
        let space = match role {
            DeclarationKind::Attackers => {
                combat.attackers_to_declare = eligible;
                self.declare_attacker_action_space(player, focal)
            }
            DeclarationKind::Blockers => {
                combat.blockers_to_declare = eligible;
                self.declare_blocker_action_space(player, focal)
            }
        };
        if space.actions.is_empty() {
            return Err(AgentError(
                "game state edits left the current declaration without legal actions".into(),
            ));
        }
        self.publish_action_space(space);
        Ok(())
    }

    fn with_game_state_edits(&self, edits: &[GameStateEdit]) -> Result<Self, AgentError> {
        let mut edited = self.rollback_snapshot();
        let seats = self.state.players.len();
        let live_permanent = |edited: &Game, permanent| {
            // Slots added in this batch are absent from the original position.
            let id = self.edit_permanent(permanent)?;
            if edited.state.permanents[id].is_none() {
                return Err(AgentError(format!(
                    "game state edits: permanent {permanent} already moved in this batch"
                )));
            }
            Ok(id)
        };
        let mut auras = Vec::new();
        for edit in edits {
            let (object, controller, ready, tapped, attached_to) = match *edit {
                GameStateEdit::MovePermanent { permanent, to } => {
                    let id = live_permanent(&edited, permanent)?;
                    edited.quiet_move_permanent(id, to);
                    continue;
                }
                GameStateEdit::SetTapped { permanent, tapped } => {
                    let id = live_permanent(&edited, permanent)?;
                    edited.state.set_tapped(id, tapped);
                    continue;
                }
                GameStateEdit::SetDamage { permanent, damage } => {
                    if damage < 0 {
                        return Err(AgentError(
                            "game state edits: damage must be nonnegative".into(),
                        ));
                    }
                    let id = live_permanent(&edited, permanent)?;
                    edited.state.set_damage(id, damage);
                    continue;
                }
                GameStateEdit::SetCounters {
                    permanent,
                    counter,
                    count,
                } => {
                    if count < 0 {
                        return Err(AgentError(
                            "game state edits: counter count must be nonnegative".into(),
                        ));
                    }
                    let id = live_permanent(&edited, permanent)?;
                    let current = edited.state.permanents[id]
                        .as_ref()
                        .expect("live permanent")
                        .counter_count(counter);
                    edited.state.change_counters(id, counter, count - current);
                    continue;
                }
                GameStateEdit::SetLife { player, life } => {
                    let player = edit_player(player, seats)?;
                    *edited.state.life_mut(player).expect("validated seat") = life;
                    continue;
                }
                GameStateEdit::CreateToken {
                    ref key,
                    controller,
                    ready,
                    tapped,
                } => {
                    let controller = edit_player(controller, seats)?;
                    validate_edit_token(key)?;
                    let token = edited
                        .prepare_token(key, controller)
                        .expect("validated token key");
                    let entry = ZoneEntry {
                        tapped,
                        ..Default::default()
                    };
                    edited.quiet_enter(token, controller, ready, entry);
                    continue;
                }
                GameStateEdit::AddToBattlefield {
                    card,
                    controller,
                    ready,
                    tapped,
                    attached_to,
                } => (
                    self.edit_card(card)?,
                    controller,
                    ready,
                    tapped,
                    attached_to,
                ),
                GameStateEdit::AddFromLibrary {
                    player,
                    ref name,
                    controller,
                    ready,
                    tapped,
                    attached_to,
                } => {
                    let owner = edit_player(player, seats)?;
                    let object = self.edit_library_card(&edited, owner, name)?;
                    (object, controller, ready, tapped, attached_to)
                }
            };

            let card = &self.state[object];
            let Some(from) = edited.state.zones.zone_of(object) else {
                return Err(AgentError(format!(
                    "game state edits: card {} already added in this batch",
                    card.id.0
                )));
            };
            let controller = controller
                .map(|seat| edit_player(seat, seats))
                .transpose()?
                .unwrap_or(card.owner);
            let face = card.retained_base().1;
            if face.types.is_instant_or_sorcery() {
                return Err(AgentError(format!(
                    "game state edits: {:?} cannot enter the battlefield",
                    face.name
                )));
            }
            let host = match (face.has_subtype("Aura"), attached_to) {
                (true, Some(id)) => Some(
                    self.permanent_object(self.edit_permanent(id)?)
                        .expect("validated host"),
                ),
                (false, None) => None,
                (true, None) => {
                    return Err(AgentError(
                        "game state edits: an Aura requires attached_to".into(),
                    ))
                }
                (false, Some(_)) => {
                    return Err(AgentError(
                        "game state edits: attached_to requires an Aura".into(),
                    ))
                }
            };
            let entry = ZoneEntry {
                tapped,
                attached_to: host,
                inspected_library_departure: true,
                ..Default::default()
            };
            edited.forget_zone_departure(object, Some(from), ZoneType::Battlefield, &entry);
            if from == ZoneType::Exile {
                edited
                    .state
                    .exile_links
                    .retain(|link| link.exiled_object != object);
            }
            policy_history::expire_links(&mut edited, card.id, None);
            let permanent = edited.quiet_enter(object, controller, ready, entry);
            if host.is_some() {
                auras.push((permanent, &face.name, card.id.0 as i32));
            }
        }
        edited.assert_stack_consistent();
        edited.refresh_control();
        for (aura, name, card) in auras {
            if edited.aura_attachment_illegal(aura, &edited.characteristics(aura)) {
                return Err(AgentError(format!(
                    "game state edits: added Aura {name:?} (card {card}) has an illegal attachment"
                )));
            }
        }
        edited.assert_library_knowledge();
        Ok(edited)
    }

    fn edit_permanent(&self, permanent: i32) -> Result<PermanentId, AgentError> {
        let id = self
            .state
            .permanents
            .iter()
            .position(|row| row.as_ref().is_some_and(|row| row.id.0 as i32 == permanent))
            .map(PermanentId)
            .ok_or_else(|| {
                AgentError(format!("game state edits: unknown permanent {permanent}"))
            })?;
        if self.is_phased_out(id) {
            return Err(AgentError(format!(
                "game state edits: permanent {permanent} is phased out"
            )));
        }
        Ok(id)
    }

    fn edit_card(&self, id: i32) -> Result<ObjectId, AgentError> {
        let (object, card) = self
            .state
            .latest_objects()
            .find(|(_, card)| card.id.0 as i32 == id)
            .ok_or_else(|| AgentError(format!("game state edits: unknown card {id}")))?;
        if !card.is_card()
            || !matches!(
                self.state.zones.zone_of(object),
                Some(ZoneType::Hand | ZoneType::Library | ZoneType::Graveyard | ZoneType::Exile)
            )
        {
            return Err(AgentError(format!(
                "game state edits: object {id} must be a card in hand, library, graveyard or exile"
            )));
        }
        Ok(object)
    }

    /// Use the original library's order and visibility, skipping consumed appearances.
    fn edit_library_card(
        &self,
        edited: &Game,
        owner: PlayerId,
        name: &str,
    ) -> Result<ObjectId, AgentError> {
        let player = &self.state.players[owner.0];
        let definition = default_content_pack().definition_id(name);
        let library = self.state.zones.zone_objects(ZoneType::Library, owner);
        let mut fallback = None;
        for (index, &object) in library.iter().enumerate() {
            if Some(self.state[object].retained_base().0) != definition
                || edited.state.zones.zone_of(object) != Some(ZoneType::Library)
            {
                continue;
            }
            fallback.get_or_insert(object);
            let position = library.len() - 1 - index;
            if !self.state.library_order_pins[owner.0].contains(&object)
                && !player.known_library.public.contains_key(&position)
                && !player.known_library.own.contains_key(&position)
                && !(position == 0 && self.looks_at_library_top(owner))
            {
                return Ok(object);
            }
        }
        fallback.ok_or_else(|| {
            AgentError(format!(
                "game state edits: no {name:?} in player {}'s library",
                owner.0
            ))
        })
    }

    fn quiet_enter(
        &mut self,
        object: ObjectId,
        controller: PlayerId,
        ready: bool,
        mut entry: ZoneEntry,
    ) -> PermanentId {
        let from = self.state.zones.zone_of(object);
        let face = self.state.entry_object(object).retained_base().1;
        if let Some(loyalty) = face
            .loyalty
            .filter(|&n| n > 0 && face.types.is_planeswalker())
        {
            entry.counters.change(CounterKind::Loyalty, loyalty);
        }
        if face.is_saga() {
            entry.counters.change(CounterKind::Lore, 1);
        }
        let arrival = self
            .state
            .mint_appearance(object, controller, ZoneType::Battlefield, false);
        let entry_turn = if ready {
            0
        } else {
            self.state.turn.turn_number
        };
        let permanent = self.install_permanent(arrival, controller, entry, entry_turn, !ready);
        self.capture_object_move(arrival, from, Some(ZoneType::Battlefield), controller);
        permanent
    }

    fn quiet_move_permanent(&mut self, permanent: PermanentId, to: PermanentDestination) {
        let object = self
            .permanent_object(permanent)
            .expect("validated permanent");
        let lki = self
            .snapshot_current_permanent(object)
            .expect("live permanent");
        let alias = lki.presentation_id;
        let shown = lki.definition_id.catalog_id();
        let row = &self.state[object];
        let id = row.id;
        let physical_card = row.is_card();
        let controller = lki.controller;
        let owner_knows_card = !row.is_face_down() || lki.owner == controller;

        self.remove_permanent_from_combat(permanent);
        if let Some(combat) = self.state.combat.as_mut() {
            combat.remove_defender(object);
        }
        let partner = self.retire_permanent(lki);
        self.end_effects_on(object);
        self.state.delayed_triggers_mut().retain(|trigger| {
            !matches!(trigger.event, DelayedTriggerEvent::Leaves { watched, .. } if watched == object)
        });
        self.state.exile_links.retain(|link| link.source != object);
        policy_history::expire_links(self, id, Some(alias));

        if physical_card {
            self.place_edited_card(object, to, controller, shown, owner_knows_card);
            if let Some(partner) = partner {
                self.place_edited_card(partner, to, controller, None, true);
            }
        } else {
            self.state.zones.remove_object(object);
            self.capture_object_move(object, Some(ZoneType::Battlefield), None, controller);
        }
    }

    fn place_edited_card(
        &mut self,
        object: ObjectId,
        to: PermanentDestination,
        controller: PlayerId,
        shown: Option<CardDefId>,
        owner_knows_card: bool,
    ) {
        let owner = self.state.entry_object(object).owner;
        let zone = match to {
            PermanentDestination::Graveyard => ZoneType::Graveyard,
            PermanentDestination::Exile => ZoneType::Exile,
            PermanentDestination::LibraryBottom => ZoneType::Library,
        };
        let arrival = self.state.mint_appearance(object, owner, zone, false);
        self.capture_object_move(arrival, Some(ZoneType::Battlefield), Some(zone), controller);
        if to == PermanentDestination::LibraryBottom {
            self.state.library_order_pins_mut(owner).insert(arrival);
            let bottom = self.state.zones.size(zone, owner) - 1;
            self.state.zones.reposition(zone, owner, arrival, 0);
            let definition = self.state[arrival].definition_id;
            let knowledge = self.state.known_library_mut(owner);
            // A hidden or changed face supplies no public fact about the
            // printing in the library. Only the controller may look at the
            // card represented by a face-down permanent.
            if shown == Some(definition) {
                knowledge.public.insert(bottom, definition);
                LibraryKnowledge::bound(&mut knowledge.public);
            }
            if owner_knows_card {
                knowledge.own.insert(bottom, definition);
                LibraryKnowledge::bound(&mut knowledge.own);
            }
        }
    }
}

#[cfg(test)]
mod tests;
