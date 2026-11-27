//! Counterfactual edits at a live decision. Validation resolves observed ids
//! before any writes. Quiet moves retire exact appearances without running
//! replacements, registering rules events, or advancing the turn.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    agent::{
        action::{ActionSpaceKind, AgentError},
        policy_history,
    },
    flow::{combat::DeclarationKind, game::Game, trigger::DelayedTriggerEvent},
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

/// An edit to an observed battlefield permanent or native player seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
        if let Some((role, player, focal)) = declaration {
            // Mandatory blocks and eligibility depend on the edited layers.
            // Commit only once the original creature still has a legal offer.
            let mut edited = self.rollback_snapshot();
            edited.apply_game_state_edits(edits)?;
            let completed = self
                .state
                .combat
                .as_ref()
                .expect("declaration has combat")
                .completed_declarers(role);
            edited.refresh_declaration(role, player, focal, completed)?;
            *self = edited;
        } else {
            let player = space.actions[0].player();
            self.apply_game_state_edits(edits)?;
            self.refresh_priority_actions(player);
        }
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
        let permanents = self.validate_game_state_edits(edits)?;
        for &edit in edits {
            match edit {
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
            }
        }
        self.assert_stack_consistent();
        self.refresh_control();
        Ok(())
    }

    fn validate_game_state_edits(
        &self,
        edits: &[GameStateEdit],
    ) -> Result<BTreeMap<i32, PermanentId>, AgentError> {
        let mut permanents = BTreeMap::new();
        let mut moved = BTreeSet::new();
        for &edit in edits {
            let permanent = match edit {
                GameStateEdit::SetLife { player, .. } => {
                    if player >= self.state.players.len() {
                        return Err(AgentError(format!(
                            "game state edits: invalid player {player}"
                        )));
                    }
                    continue;
                }
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
                entry.insert(id);
            }
            if matches!(edit, GameStateEdit::MovePermanent { .. }) {
                moved.insert(permanent);
            }
        }
        Ok(permanents)
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
