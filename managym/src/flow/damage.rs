// damage.rs
// Damage application, life changes, and cleanup.

use crate::{
    flow::{
        event::{DamageTarget, EventSubject, GameEvent},
        game::Game,
        proposed_event::{ProposedDamageTarget, ProposedEvent},
    },
    state::game_object::{CardId, ObjectRef, PermanentId, PlayerId},
};

impl Game {
    pub(crate) fn apply_player_damage(
        &mut self,
        source: Option<ObjectRef>,
        player: PlayerId,
        amount: i32,
    ) {
        self.apply_proposed_event(ProposedEvent::Damage {
            source,
            target: ProposedDamageTarget::Player(player),
            amount,
            combat: false,
        });
    }

    pub(crate) fn apply_permanent_damage(
        &mut self,
        source: Option<ObjectRef>,
        permanent_id: PermanentId,
        amount: i32,
    ) {
        let Some(target) = self.permanent_object_ref(permanent_id) else {
            return;
        };
        self.apply_proposed_event(ProposedEvent::Damage {
            source,
            target: ProposedDamageTarget::Object(target),
            amount,
            combat: false,
        });
    }

    /// Deal noncombat damage to a creature and return the excess damage
    /// actually dealt (CR 120.4a): the damage marked beyond lethal damage,
    /// which counts damage already marked (CR 120.6) and is 1 from a
    /// deathtouch source (CR 702.2c). Replacements apply first, so prevented
    /// damage is never excess.
    pub(crate) fn apply_creature_damage_measuring_excess(
        &mut self,
        source: Option<ObjectRef>,
        permanent_id: PermanentId,
        amount: i32,
    ) -> i32 {
        let marked = |game: &Self| {
            game.state.permanents[permanent_id]
                .as_ref()
                .map(|permanent| permanent.damage)
        };
        let Some(before) = marked(self) else {
            return 0;
        };
        if !self.permanent_is_creature(permanent_id) {
            self.apply_permanent_damage(source, permanent_id, amount);
            return 0;
        }
        let remaining = (self.effective_toughness(permanent_id) - before).max(0);
        let deathtouch = source.is_some_and(|object_ref| {
            self.source_keywords(object_ref)
                .is_some_and(|keywords| keywords.deathtouch)
        });
        let lethal = if deathtouch {
            remaining.min(1)
        } else {
            remaining
        };
        self.apply_permanent_damage(source, permanent_id, amount);
        let dealt = marked(self).unwrap_or(before) - before;
        (dealt - lethal).max(0)
    }

    pub(crate) fn apply_combat_player_damage(
        &mut self,
        source: Option<ObjectRef>,
        player: PlayerId,
        amount: i32,
    ) {
        self.apply_proposed_event(ProposedEvent::Damage {
            source,
            target: ProposedDamageTarget::Player(player),
            amount,
            combat: true,
        });
    }

    pub(crate) fn apply_combat_permanent_damage(
        &mut self,
        source: Option<ObjectRef>,
        permanent_id: PermanentId,
        amount: i32,
    ) {
        let Some(target) = self.permanent_object_ref(permanent_id) else {
            return;
        };
        self.apply_proposed_event(ProposedEvent::Damage {
            source,
            target: ProposedDamageTarget::Object(target),
            amount,
            combat: true,
        });
    }

    pub(crate) fn commit_damage(
        &mut self,
        source: Option<ObjectRef>,
        target: ProposedDamageTarget,
        amount: i32,
        combat: bool,
    ) -> bool {
        if amount <= 0 {
            return false;
        }
        let source_card = source.map(|object_ref| CardId::from(object_ref.entity));
        let source_has_deathtouch = source.is_some_and(|object_ref| {
            self.source_keywords(object_ref)
                .is_some_and(|keywords| keywords.deathtouch)
        });
        let lifelink_controller = source.and_then(|object_ref| {
            self.source_keywords(object_ref)
                .is_some_and(|keywords| keywords.lifelink)
                .then(|| self.source_controller(object_ref))
                .flatten()
        });
        let combat_source = combat
            .then(|| source.and_then(|object_ref| self.object_event_ref(object_ref)))
            .flatten();

        match target {
            ProposedDamageTarget::Player(player) => {
                self.journal_player(player);
                let Some(player_state) = self.state.players.get_mut(player.0) else {
                    return false;
                };
                let old_life = player_state.life;
                player_state.life = player_state.life.saturating_sub(amount);
                let new_life = player_state.life;

                self.emit(GameEvent::DamageDealt {
                    source: source_card,
                    target: DamageTarget::Player(player),
                    amount: amount as u32,
                });
                if let Some(source) = combat_source {
                    self.emit_observation_event(GameEvent::CombatDamageDealt {
                        source,
                        target: EventSubject::Player(player),
                        amount: amount as u32,
                    });
                }
                self.emit(GameEvent::LifeChanged {
                    player,
                    old: old_life,
                    new: new_life,
                });
            }
            ProposedDamageTarget::Object(object_ref) => {
                let event_target = self.object_event_ref(object_ref);
                let Ok(permanent_id) = self.lookup_current_permanent(object_ref) else {
                    return false;
                };
                self.journal_permanent(permanent_id);
                let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
                    return false;
                };
                permanent.take_damage(amount);
                if source_has_deathtouch {
                    permanent.deathtouch_damage = true;
                }
                self.emit(GameEvent::DamageDealt {
                    source: source_card,
                    target: DamageTarget::Permanent(permanent_id),
                    amount: amount as u32,
                });
                if let (Some(source), Some(target)) = (combat_source, event_target) {
                    self.emit_observation_event(GameEvent::CombatDamageDealt {
                        source,
                        target: EventSubject::Object(target),
                        amount: amount as u32,
                    });
                }
            }
        }

        if let Some(controller) = lifelink_controller {
            self.change_life(controller, amount);
        }
        true
    }

    pub(crate) fn change_life(&mut self, player: PlayerId, delta: i32) -> bool {
        self.apply_proposed_event(ProposedEvent::LifeChange { player, delta })
    }

    pub(crate) fn commit_life_change(&mut self, player: PlayerId, delta: i32) -> bool {
        if delta > 0
            && [PlayerId(0), PlayerId(1)].into_iter().any(|owner| {
                self.battlefield_permanents(owner).into_iter().any(|id| {
                    self.state.permanents[id]
                        .as_ref()
                        .is_some_and(|p| self.state.cards[p.card].prevents_life_gain)
                })
            })
        {
            return false;
        }
        self.journal_player(player);
        let Some(player_state) = self.state.players.get_mut(player.0) else {
            return false;
        };
        let old_life = player_state.life;
        player_state.life = player_state.life.saturating_add(delta);
        let new_life = player_state.life;
        if old_life == new_life {
            return false;
        }
        self.emit(GameEvent::LifeChanged {
            player,
            old: old_life,
            new: new_life,
        });
        true
    }
    pub(crate) fn clear_damage(&mut self) {
        for index in 0..self.state.permanents.len() {
            let permanent_id = PermanentId(index);
            if self.state.permanents[permanent_id].is_some() {
                self.journal_permanent(permanent_id);
                self.state.permanents[permanent_id]
                    .as_mut()
                    .expect("checked permanent")
                    .clear_damage();
            }
        }
    }

    /// Damage-source keywords include until-EOT grants while the source is
    /// on the battlefield (Enter the Avatar State's lifelink), and static
    /// grants to a resolving spell (Lo and Li's Lesson spells).
    fn source_keywords(&self, source: ObjectRef) -> Option<crate::state::card::Keywords> {
        if let Ok(permanent_id) = self.lookup_current_permanent(source) {
            return Some(self.effective_keywords(permanent_id));
        }
        if let Some(lki) = self.object_lki(source) {
            return Some(lki.keywords);
        }
        let card = CardId::from(source.entity);
        if let Some(controller) = self.stack_spell_controller(card) {
            return Some(self.spell_keywords(card, controller));
        }
        self.state.cards.get(card.0).map(|card| card.keywords)
    }

    fn source_controller(&self, source: ObjectRef) -> Option<PlayerId> {
        if let Ok(permanent_id) = self.lookup_current_permanent(source) {
            return self.state.permanents[permanent_id]
                .as_ref()
                .map(|permanent| permanent.controller);
        }
        if let Some(lki) = self.object_lki(source) {
            return Some(lki.controller);
        }
        let card = CardId::from(source.entity);
        self.state.cards.get(card.0).map(|card| card.owner)
    }

    pub(crate) fn gain_life(&mut self, player: PlayerId, amount: i32) {
        if amount > 0 {
            self.change_life(player, amount);
        }
    }

    /// The cleanup step (CR 514.2): "until end of turn" effects end and
    /// "once each turn" limits and "dealt damage this turn" reset.
    pub(crate) fn end_turn_modifiers(&mut self) {
        self.end_turn_effects();
        for index in 0..self.state.permanents.len() {
            let permanent_id = PermanentId(index);
            if self.state.permanents[permanent_id]
                .as_ref()
                .is_some_and(|permanent| {
                    !permanent.activated_this_turn.is_empty() || permanent.dealt_damage_this_turn
                })
            {
                self.journal_permanent(permanent_id);
                self.state.permanents[permanent_id]
                    .as_mut()
                    .expect("checked permanent")
                    .clear_turn_history();
            }
        }
    }
}
