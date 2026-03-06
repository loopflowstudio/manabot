// play.rs
// Spell casting, land plays, and activated abilities.

use crate::{
    agent::action::AgentError,
    flow::{
        event::GameEvent,
        game::{Game, PendingChoice},
    },
    state::{
        game_object::{CardId, PlayerId, Target},
        mana::ManaCost,
        stack_object::{ActivatedAbilityOnStack, StackObject},
        zone::ZoneType,
    },
};

impl Game {
    pub fn can_cast_sorceries(&self, player: PlayerId) -> bool {
        // CR 117.1a, 307.1 — Sorcery-speed actions are available only to the active player
        // during a main phase with an empty stack.
        self.is_active_player(player)
            && self.stack_is_empty()
            && self.state.turn.can_cast_sorceries()
    }

    pub fn can_cast_instants(&self, _player: PlayerId) -> bool {
        // CR 117.1a — Any player with priority may cast an instant.
        true
    }

    pub fn can_play_land(&self, player: PlayerId) -> bool {
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to one per turn.
        self.can_cast_sorceries(player) && self.state.turn.lands_played < 1
    }

    pub fn can_pay_mana_cost(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.producible_mana(player).can_pay(cost)
    }

    pub(crate) fn play_land(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let card_ref = &self.state.cards[card];
        if !card_ref.types.is_land() {
            return Err(AgentError("only land cards can be played".to_string()));
        }
        if card_ref.owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        self.state.turn.lands_played += 1;
        self.move_card(card, ZoneType::Battlefield);
        self.invalidate_mana_cache(player);

        Ok(())
    }

    pub(crate) fn cast_spell_action(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        if self.pending_choice.is_some() {
            return Err(AgentError("a choice is already pending".to_string()));
        }

        let (is_land, owner, is_instant_speed) = {
            let card_ref = &self.state.cards[card];
            (
                card_ref.types.is_land(),
                card_ref.owner,
                card_ref.types.is_instant_speed(),
            )
        };

        if is_land {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        if is_instant_speed {
            if !self.can_cast_instants(player) {
                return Err(AgentError("cannot cast instant now".to_string()));
            }
        } else if !self.can_cast_sorceries(player) {
            return Err(AgentError(
                "cannot cast sorcery-speed spell now".to_string(),
            ));
        }

        if let Some(legal_targets) = self.legal_targets_for_spell(card) {
            // CR 601.2c — Choose target(s) as part of casting.
            if legal_targets.is_empty() {
                return Err(AgentError("no legal targets".to_string()));
            }
            self.pending_choice = Some(PendingChoice::ChooseTarget {
                player,
                card,
                legal_targets,
            });
            return Ok(());
        }

        self.pay_spell_cost(player, card)?;

        self.cast_spell(player, card, None)?;
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    pub(crate) fn choose_target_action(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::ChooseTarget {
            player: chooser,
            card,
            legal_targets,
        }) = self.pending_choice.as_ref()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };

        if *chooser != player {
            return Err(AgentError("wrong player for target choice".to_string()));
        }
        if !legal_targets.contains(&target) {
            return Err(AgentError("target is not legal".to_string()));
        }

        let card = *card;
        self.pay_spell_cost(player, card)?;

        self.cast_spell(player, card, Some(target))?;
        self.pending_choice = None;
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    pub(crate) fn pay_spell_cost(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        let Some(cost) = self.state.cards[card].mana_cost.clone() else {
            return Ok(());
        };
        // CR 601.2f, 601.2h — Determine/pay costs as part of casting.
        self.produce_mana(player, &cost)?;
        self.spend_mana(player, &cost)
    }

    pub(crate) fn cast_spell(
        &mut self,
        player: PlayerId,
        card: CardId,
        target: Option<Target>,
    ) -> Result<(), AgentError> {
        let owner = self.state.cards[card].owner;
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        // CR 601.2i — A cast spell is put onto the stack.
        self.push_spell_to_stack(card, player, target);
        self.emit(GameEvent::SpellCast { card, target });
        Ok(())
    }

    pub(crate) fn activate_ability_action(
        &mut self,
        player: PlayerId,
        permanent_id: crate::state::game_object::PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        if permanent.controller != player {
            return Err(AgentError(
                "source permanent is not controlled by player".to_string(),
            ));
        }
        if self.state.zones.zone_of(permanent.card) != Some(ZoneType::Battlefield) {
            return Err(AgentError(
                "source permanent must be on battlefield".to_string(),
            ));
        }

        let source_permanent_object_id = permanent.id;
        let source_card = permanent.card;
        let source_card_registry_key = self.state.cards[source_card].registry_key;
        let Some(ability) = self.state.cards[source_card]
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        self.produce_mana(player, &ability.mana_cost)?;
        self.spend_mana(player, &ability.mana_cost)?;

        let id = self.state.id_gen.next_id();
        self.push_to_stack(StackObject::ActivatedAbility(ActivatedAbilityOnStack {
            id,
            controller: player,
            source_card_registry_key,
            source_card,
            source_permanent_object_id,
            ability_index,
            targets: Vec::new(),
        }));
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    pub(crate) fn legal_targets_for_spell(&self, card: CardId) -> Option<Vec<Target>> {
        let spec = self.state.cards[card]
            .spell_effect
            .as_ref()?
            .target_spec()?;
        Some(self.legal_targets_for_target_spec(spec))
    }

    fn legal_targets_for_target_spec(
        &self,
        spec: &crate::state::ability::TargetSpec,
    ) -> Vec<Target> {
        match spec {
            crate::state::ability::TargetSpec::CreatureOrPlayer => {
                let mut targets = vec![Target::Player(PlayerId(0)), Target::Player(PlayerId(1))];
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            targets.push(Target::Permanent(permanent_id));
                        }
                    }
                }
                targets
            }
            crate::state::ability::TargetSpec::Creature => {
                let mut targets = Vec::new();
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            targets.push(Target::Permanent(permanent_id));
                        }
                    }
                }
                targets
            }
            crate::state::ability::TargetSpec::Spell => self
                .state
                .stack_objects
                .iter()
                .rev()
                .filter_map(|object| match object {
                    StackObject::Spell(spell) => Some(Target::StackSpell(spell.card)),
                    _ => None,
                })
                .collect(),
        }
    }
}
