// damage.rs
// Damage application, life changes, and cleanup.

use crate::state::card::Keyword;
use crate::{
    flow::{
        event::{DamageTarget, EventSubject, GameEvent},
        game::Game,
        proposed_event::{ProposedDamageTarget, ProposedEvent},
    },
    state::{
        ability::DelayedExpiry,
        card::StaticAbility,
        game_object::{CardId, ObjectRef, PermanentId, PlayerId},
        permanent::CounterKind,
    },
};

impl Game {
    /// Excess after replacement, measured against the pre-commit creature.
    /// Deathtouch makes one remaining damage lethal (CR 120.4a).
    pub(crate) fn excess_for_damage(
        &self,
        source: Option<ObjectRef>,
        permanent_id: PermanentId,
        amount: i32,
    ) -> i32 {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return 0;
        };
        let characteristics = self.characteristics(permanent_id);
        if !characteristics.is_creature() {
            return 0;
        }
        let remaining = (characteristics.toughness - permanent.damage).max(0);
        let deathtouch = source.is_some_and(|object| {
            self.source_keywords(object)
                .is_some_and(|keywords| keywords.has(Keyword::Deathtouch))
        });
        let lethal = if deathtouch {
            remaining.min(1)
        } else {
            remaining
        };
        (amount - lethal).max(0)
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
                .is_some_and(|keywords| keywords.has(Keyword::Deathtouch))
        });
        let lifelink_controller = source.and_then(|object_ref| {
            self.source_keywords(object_ref)
                .is_some_and(|keywords| keywords.has(Keyword::Lifelink))
                .then(|| self.source_controller(object_ref))
                .flatten()
        });
        let combat_source = combat
            .then(|| source.and_then(|object_ref| self.object_event_ref(object_ref)))
            .flatten();

        match target {
            ProposedDamageTarget::Player(player) => {
                let Some(life) = self.state.life_mut(player) else {
                    return false;
                };
                let old_life = *life;
                *life = life.saturating_sub(amount);
                let new_life = *life;

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
                if self.state.permanents[permanent_id].is_none() {
                    return false;
                }
                // CR 120.3: a permanent with several of these types receives
                // every applicable result, but this remains one damage event.
                let view = self.characteristics(permanent_id);
                if view.is_creature() {
                    self.state
                        .mark_damage(permanent_id, amount, source_has_deathtouch);
                } else if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                    permanent.dealt_damage_this_turn = true;
                }
                if view.types.is_planeswalker() {
                    self.commit_counter_change(object_ref, CounterKind::Loyalty, -amount, None);
                }
                if view.types.is_battle() {
                    self.commit_counter_change(object_ref, CounterKind::Defense, -amount, None);
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
            self.commit_life_change(controller, amount);
        }
        true
    }

    pub(crate) fn change_life(&mut self, player: PlayerId, delta: i32) -> bool {
        self.apply_immediate_event(ProposedEvent::LifeChange { player, delta })
    }

    pub(crate) fn commit_life_change(&mut self, player: PlayerId, delta: i32) -> bool {
        if delta > 0
            && [PlayerId(0), PlayerId(1)].into_iter().any(|owner| {
                self.battlefield_permanents(owner).into_iter().any(|id| {
                    self.characteristics(id)
                        .static_abilities
                        .contains(&StaticAbility::PlayersCantGainLife)
                })
            })
        {
            return false;
        }
        let Some(life) = self.state.life_mut(player) else {
            return false;
        };
        let old_life = *life;
        *life = life.saturating_add(delta);
        let new_life = *life;
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
            self.state.clear_damage(permanent_id);
        }
    }

    /// Damage-source keywords include until-EOT grants while the source is
    /// on the battlefield (Enter the Avatar State's lifelink), and static
    /// grants to a resolving spell (Lo and Li's Lesson spells).
    fn source_keywords(&self, source: ObjectRef) -> Option<crate::state::card::Keywords> {
        if let Ok(permanent_id) = self.lookup_current_permanent(source) {
            return Some(self.characteristics(permanent_id).keywords);
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
        if self
            .state
            .delayed_triggers
            .iter()
            .any(|trigger| trigger.expiry == DelayedExpiry::ThisTurn)
        {
            self.state
                .delayed_triggers_mut()
                .retain(|trigger| trigger.expiry != DelayedExpiry::ThisTurn);
        }
        for index in 0..self.state.permanents.len() {
            let permanent_id = PermanentId(index);
            if self.state.permanents[permanent_id]
                .as_ref()
                .is_some_and(|permanent| {
                    !permanent.activated_this_turn.is_empty() || permanent.dealt_damage_this_turn
                })
            {
                self.state.clear_permanent_turn_history(permanent_id);
            }
        }
    }
}
