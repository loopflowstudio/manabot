//! Counterfactual edits at a live decision. Validation resolves observed ids
//! before any writes. Quiet moves change exact appearances without running
//! replacements, registering rules events, or advancing the turn.

use std::collections::{BTreeMap, BTreeSet};

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

struct CardAddition {
    object: ObjectId,
    controller: PlayerId,
    entry: ZoneEntry,
    ready: bool,
}

fn edit_player(player: usize, seats: usize) -> Result<PlayerId, AgentError> {
    if player >= seats {
        return Err(AgentError(format!(
            "game state edits: invalid player {player}"
        )));
    }
    Ok(PlayerId(player))
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
        let mut edited = self.rollback_snapshot();
        edited.apply_game_state_edits(edits)?;
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

    /// Resolve every target before writing, then refresh control for the whole batch.
    fn apply_game_state_edits(&mut self, edits: &[GameStateEdit]) -> Result<(), AgentError> {
        let permanents = self.validate_permanent_edits(edits)?;
        let mut additions = self.validate_additions(edits)?.into_iter();
        let mut auras = Vec::new();
        for edit in edits {
            match *edit {
                GameStateEdit::MovePermanent { permanent, to } => {
                    self.quiet_move_permanent(permanents[&permanent], to);
                }
                GameStateEdit::SetTapped { permanent, tapped } => {
                    self.state.set_tapped(permanents[&permanent], tapped);
                }
                GameStateEdit::SetDamage { permanent, damage } => {
                    self.state.set_damage(permanents[&permanent], damage);
                }
                GameStateEdit::SetCounters {
                    permanent,
                    counter,
                    count,
                } => {
                    let id = permanents[&permanent];
                    let current = self.state.permanents[id]
                        .as_ref()
                        .expect("validated permanent")
                        .counter_count(counter);
                    self.state.change_counters(id, counter, count - current);
                }
                GameStateEdit::SetLife { player, life } => {
                    *self
                        .state
                        .life_mut(PlayerId(player))
                        .expect("validated seat") = life;
                }
                GameStateEdit::CreateToken {
                    ref key,
                    controller,
                    ready,
                    tapped,
                } => {
                    let controller = PlayerId(controller);
                    let token = self
                        .prepare_token(key, controller)
                        .expect("validated token key");
                    let entry = ZoneEntry {
                        tapped,
                        ..Default::default()
                    };
                    let permanent = self.quiet_enter(token, controller, ready, entry);
                    self.capture_quiet_token(self.permanent_object(permanent).expect("new token"));
                }
                GameStateEdit::AddToBattlefield { .. } | GameStateEdit::AddFromLibrary { .. } => {
                    let CardAddition {
                        object,
                        controller,
                        entry,
                        ready,
                    } = additions.next().expect("validated card addition");
                    let from = self.state.zones.zone_of(object);
                    self.forget_zone_departure(object, from, ZoneType::Battlefield, &entry);
                    policy_history::expire_links(self, self.state[object].id, None);
                    let attached = entry.attached_to.is_some();
                    let permanent = self.quiet_enter(object, controller, ready, entry);
                    if attached {
                        auras.push(permanent);
                    }
                }
            }
        }
        self.assert_stack_consistent();
        self.refresh_control();
        for aura in auras {
            if self.aura_attachment_illegal(aura, &self.characteristics(aura)) {
                let id = self.state.permanents[aura]
                    .as_ref()
                    .expect("added Aura")
                    .id
                    .0;
                return Err(AgentError(format!(
                    "game state edits: added Aura {id} has an illegal attachment"
                )));
            }
        }
        self.assert_library_knowledge();
        Ok(())
    }

    fn validate_permanent_edits(
        &self,
        edits: &[GameStateEdit],
    ) -> Result<BTreeMap<i32, PermanentId>, AgentError> {
        let mut permanents = BTreeMap::new();
        let mut moved = BTreeSet::new();
        for edit in edits {
            let permanent = match *edit {
                GameStateEdit::SetLife { player, .. } => {
                    edit_player(player, self.state.players.len())?;
                    continue;
                }
                GameStateEdit::CreateToken { .. }
                | GameStateEdit::AddToBattlefield { .. }
                | GameStateEdit::AddFromLibrary { .. } => continue,
                GameStateEdit::SetDamage { damage, .. } if damage < 0 => {
                    return Err(AgentError(
                        "game state edits: damage must be nonnegative".into(),
                    ));
                }
                GameStateEdit::SetCounters { count, .. } if count < 0 => {
                    return Err(AgentError(
                        "game state edits: counter count must be nonnegative".into(),
                    ));
                }
                GameStateEdit::MovePermanent { permanent, .. }
                | GameStateEdit::SetTapped { permanent, .. }
                | GameStateEdit::SetDamage { permanent, .. }
                | GameStateEdit::SetCounters { permanent, .. } => permanent,
            };
            if moved.contains(&permanent) {
                return Err(AgentError(format!(
                    "game state edits: permanent {permanent} already moved in this batch"
                )));
            }
            if let std::collections::btree_map::Entry::Vacant(entry) = permanents.entry(permanent) {
                entry.insert(self.edit_permanent(permanent)?);
            }
            if matches!(edit, GameStateEdit::MovePermanent { .. }) {
                moved.insert(permanent);
            }
        }
        Ok(permanents)
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

    fn validate_additions(&self, edits: &[GameStateEdit]) -> Result<Vec<CardAddition>, AgentError> {
        let seats = self.state.players.len();
        let mut additions = Vec::new();
        let mut claimed = BTreeSet::new();
        for edit in edits {
            let (controller, ready, tapped, attached_to) = match *edit {
                GameStateEdit::CreateToken {
                    ref key,
                    controller,
                    ..
                } => {
                    edit_player(controller, seats)?;
                    if default_content_pack().token_id(key).is_none() {
                        return Err(AgentError(format!(
                            "game state edits: unknown token key {key:?}"
                        )));
                    }
                    continue;
                }
                GameStateEdit::AddToBattlefield {
                    controller,
                    ready,
                    tapped,
                    attached_to,
                    ..
                }
                | GameStateEdit::AddFromLibrary {
                    controller,
                    ready,
                    tapped,
                    attached_to,
                    ..
                } => (controller, ready, tapped, attached_to),
                _ => continue,
            };
            let object = match *edit {
                GameStateEdit::AddToBattlefield { card, .. } => self.edit_card(card)?,
                GameStateEdit::AddFromLibrary {
                    player, ref name, ..
                } => {
                    let owner = edit_player(player, seats)?;
                    self.edit_library_card(owner, name, &claimed)?
                }
                _ => unreachable!("card addition"),
            };
            let card = &self.state[object];
            if !claimed.insert(object) {
                return Err(AgentError(format!(
                    "game state edits: card {} already added in this batch",
                    card.id.0
                )));
            }
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
            let mut entry = ZoneEntry {
                tapped,
                attached_to: host,
                inspected_library_departure: true,
                ..Default::default()
            };
            if let Some(loyalty) = face
                .loyalty
                .filter(|&n| n > 0 && face.types.is_planeswalker())
            {
                entry.counters.change(CounterKind::Loyalty, loyalty);
            }
            if face.is_saga() {
                entry.counters.change(CounterKind::Lore, 1);
            }
            additions.push(CardAddition {
                object,
                controller,
                entry,
                ready,
            });
        }
        Ok(additions)
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

    fn edit_library_card(
        &self,
        owner: PlayerId,
        name: &str,
        claimed: &BTreeSet<ObjectId>,
    ) -> Result<ObjectId, AgentError> {
        let player = &self.state.players[owner.0];
        let definition = default_content_pack().definition_id(name);
        let library = self.state.zones.zone_objects(ZoneType::Library, owner);
        let mut fallback = None;
        for (index, &object) in library.iter().enumerate() {
            if Some(self.state[object].retained_base().0) != definition || claimed.contains(&object)
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
        entry: ZoneEntry,
    ) -> PermanentId {
        let arrival = self
            .state
            .mint_appearance(object, controller, ZoneType::Battlefield, false);
        let entry_turn = if ready {
            0
        } else {
            self.state.turn.turn_number
        };
        self.install_permanent(arrival, controller, entry, entry_turn, !ready)
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
        let owner_knows_card = !row.is_face_down() || lki.owner == lki.controller;

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
            self.place_edited_card(object, to, shown, owner_knows_card);
            if let Some(partner) = partner {
                self.place_edited_card(partner, to, None, true);
            }
        } else {
            self.state.zones.remove_object(object);
        }
        self.assert_library_knowledge();
    }

    fn place_edited_card(
        &mut self,
        object: ObjectId,
        to: PermanentDestination,
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
