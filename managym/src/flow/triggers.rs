// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{event::GameEvent, game::Game, trigger::PendingTrigger},
    state::{
        ability::{Ability, TargetSpec, TriggerCondition, TriggerSource},
        game_object::{CardId, PlayerId, Target},
        stack_object::{StackObject, TriggeredAbilityOnStack},
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn choose_trigger_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let pending_trigger = self
            .state
            .pending_trigger_choice
            .take()
            .ok_or_else(|| AgentError("no pending target choice".to_string()))?;

        if pending_trigger.controller != player {
            return Err(AgentError("wrong player for target selection".to_string()));
        }

        let action_target = match target {
            Target::Player(p) => ActionTarget::Player(p),
            Target::Permanent(p) => ActionTarget::Permanent(p),
            Target::StackSpell(_) => {
                return Err(AgentError(
                    "invalid target for triggered ability".to_string(),
                ));
            }
        };

        let Some(target_spec) = self.trigger_target_spec(&pending_trigger) else {
            return Err(AgentError("triggered ability no longer exists".to_string()));
        };
        if !self.is_valid_target_for_spec(action_target, target_spec) {
            return Err(AgentError("selected target is not legal".to_string()));
        }

        self.place_triggered_ability_on_stack(pending_trigger, Some(target));
        Ok(())
    }

    pub(crate) fn flush_triggers(&mut self) -> Option<ActionSpace> {
        while let Some(trigger) = self
            .state
            .pending_trigger_choice
            .take()
            .or_else(|| self.pop_next_pending_trigger())
        {
            let Some(target_spec) = self.trigger_target_spec(&trigger) else {
                continue;
            };

            let legal_targets = self.legal_targets_for_spec(target_spec);
            if legal_targets.is_empty() {
                // CR 603.3d — Triggered abilities with no legal required targets are removed.
                continue;
            }

            let controller = trigger.controller;
            self.state.pending_trigger_choice = Some(trigger);
            return Some(ActionSpace {
                player: Some(controller),
                kind: ActionSpaceKind::ChooseTarget,
                actions: legal_targets
                    .into_iter()
                    .map(|legal_target| Action::ChooseTarget {
                        player: controller,
                        target: legal_target,
                    })
                    .collect(),
                focus: Vec::new(),
            });
        }

        None
    }

    fn trigger_target_spec<'a>(&'a self, trigger: &PendingTrigger) -> Option<&'a TargetSpec> {
        let ability = self
            .state
            .cards
            .get(trigger.source_card.0)
            .and_then(|card| card.abilities.get(trigger.ability_index))?;

        let Ability::Triggered { effect, .. } = ability;
        effect.target_spec()
    }

    fn pop_next_pending_trigger(&mut self) -> Option<PendingTrigger> {
        let active = self.active_player();
        let next_index = self
            .state
            .pending_triggers
            .iter()
            .enumerate()
            .min_by_key(|(_, trigger)| {
                let apnap_rank = if trigger.controller == active {
                    0_u8
                } else {
                    1_u8
                };
                (apnap_rank, trigger.enqueue_order)
            })
            .map(|(index, _)| index)?;
        Some(self.state.pending_triggers.remove(next_index))
    }

    fn place_triggered_ability_on_stack(
        &mut self,
        trigger: PendingTrigger,
        target: Option<Target>,
    ) {
        let source_card_registry_key = self.state.cards[trigger.source_card].registry_key;
        let targets = target.into_iter().collect();
        self.state
            .stack_objects
            .push(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
                id: self.state.id_gen.next_id(),
                controller: trigger.controller,
                source_card: trigger.source_card,
                source_card_registry_key,
                ability_index: trigger.ability_index,
                targets,
            }));
        self.state.priority.start_round(self.active_player());
    }

    pub(crate) fn legal_targets_for_spec(&self, target_spec: &TargetSpec) -> Vec<ActionTarget> {
        match target_spec {
            TargetSpec::Creature | TargetSpec::CreatureOrPlayer => {
                let mut out = Vec::new();
                if matches!(target_spec, TargetSpec::CreatureOrPlayer) {
                    out.push(ActionTarget::Player(PlayerId(0)));
                    out.push(ActionTarget::Player(PlayerId(1)));
                }
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            out.push(ActionTarget::Permanent(permanent_id));
                        }
                    }
                }
                out
            }
            TargetSpec::Spell => Vec::new(),
        }
    }

    pub(crate) fn is_valid_target_for_spec(
        &self,
        target: ActionTarget,
        target_spec: &TargetSpec,
    ) -> bool {
        match (target, target_spec) {
            (ActionTarget::Player(_), TargetSpec::CreatureOrPlayer) => true,
            (
                ActionTarget::Permanent(permanent_id),
                TargetSpec::Creature | TargetSpec::CreatureOrPlayer,
            ) => {
                let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                    return false;
                };
                let card = &self.state.cards[permanent.card];
                card.types.is_creature()
                    && self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
            }
            _ => false,
        }
    }

    pub(crate) fn process_game_events(&mut self) {
        let events = std::mem::take(&mut self.state.pending_events);
        for event in events {
            if let GameEvent::CardMoved {
                card,
                from,
                to,
                controller,
            } = event
            {
                self.check_triggers_for_card_moved(card, from, to, controller);
            }
        }
    }

    fn check_triggers_for_card_moved(
        &mut self,
        card: CardId,
        from: Option<ZoneType>,
        to: ZoneType,
        controller: PlayerId,
    ) {
        let mut triggered_abilities = Vec::new();
        if let Some(source_card) = self.state.cards.get(card.0) {
            for (ability_index, ability) in source_card.abilities.iter().enumerate() {
                let Ability::Triggered {
                    condition,
                    intervening_if,
                    ..
                } = ability;

                if !self.trigger_condition_matches_card_moved(condition, from, to) {
                    continue;
                }
                if let Some(intervening) = intervening_if.as_ref() {
                    if !self.check_trigger_condition(intervening, card) {
                        continue;
                    }
                }
                triggered_abilities.push(ability_index);
            }
        }

        for ability_index in triggered_abilities {
            self.state.pending_triggers.push(PendingTrigger {
                source_card: card,
                ability_index,
                controller,
                enqueue_order: self.state.trigger_enqueue_counter,
            });
            self.state.trigger_enqueue_counter =
                self.state.trigger_enqueue_counter.saturating_add(1);
        }
    }

    fn trigger_condition_matches_card_moved(
        &self,
        condition: &TriggerCondition,
        from: Option<ZoneType>,
        to: ZoneType,
    ) -> bool {
        match condition {
            TriggerCondition::EntersTheBattlefield {
                source: TriggerSource::This,
            } => from != Some(ZoneType::Battlefield) && to == ZoneType::Battlefield,
        }
    }

    pub(crate) fn check_trigger_condition(
        &self,
        condition: &TriggerCondition,
        source_card: CardId,
    ) -> bool {
        match condition {
            TriggerCondition::EntersTheBattlefield {
                source: TriggerSource::This,
            } => self.state.zones.zone_of(source_card) == Some(ZoneType::Battlefield),
        }
    }
}
