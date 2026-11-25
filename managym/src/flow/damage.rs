// damage.rs
// Damage application, life changes, and cleanup.

use crate::{
    flow::{
        event::{CombatPlayerHit, DamageTarget, EventSubject, GameEvent},
        game::Game,
        proposed_event::{EventResult, ProposedDamageTarget, ProposedEvent},
    },
    state::{
        ability::DelayedExpiry,
        card::{Keyword, Keywords, StaticAbility},
        game_object::{ObjectId, PermanentId, PlayerId},
        permanent::CounterKind,
    },
};

/// Life a lifelink source's damage gains its controller, committed by the
/// damage's simultaneous batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LifelinkGain {
    pub source: ObjectId,
    pub controller: PlayerId,
    pub amount: i32,
}

impl Game {
    /// CR 119.9, 120.3f: one gain per source across simultaneous recipients.
    /// Compute these results after damage prevention but before committing any
    /// member, so a gain's replacement choice also suspends the damage.
    pub(super) fn lifelink_gains(
        &self,
        events: &[super::proposed_event::PendingEvent],
    ) -> Vec<LifelinkGain> {
        let mut totals: Vec<LifelinkGain> = Vec::new();
        for event in events {
            let ProposedEvent::Damage {
                source: Some(source),
                amount,
                target,
                ..
            } = event.event
            else {
                continue;
            };
            if amount <= 0
                || self.damage_is_prevented(target)
                || !self.source_keywords(source).has(Keyword::Lifelink)
            {
                continue;
            }
            let Some(controller) = self.source_controller(source) else {
                continue;
            };
            if let Some(total) = totals.iter_mut().find(|total| total.source == source) {
                total.amount = total.amount.saturating_add(amount);
            } else {
                totals.push(LifelinkGain {
                    source,
                    controller,
                    amount,
                });
            }
        }
        totals
    }

    fn damage_is_prevented(&self, target: ProposedDamageTarget) -> bool {
        matches!(target, ProposedDamageTarget::Player(player) if self.player_has(player, &crate::state::effect::Modification::ProtectionFromEverything))
    }

    /// Excess after replacement, measured against the pre-commit creature.
    /// Deathtouch makes one remaining damage lethal (CR 120.4a).
    fn excess_for_damage(
        &self,
        permanent_id: PermanentId,
        view: &crate::flow::statics::Characteristics,
        amount: i32,
        deathtouch: bool,
    ) -> i32 {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return 0;
        };
        if !view.is_creature() {
            return 0;
        }
        let remaining = (view.toughness - permanent.damage).max(0);
        let lethal = if deathtouch {
            remaining.min(1)
        } else {
            remaining
        };
        (amount - lethal).max(0)
    }

    pub(crate) fn commit_damage(
        &mut self,
        source: Option<ObjectId>,
        target: ProposedDamageTarget,
        amount: i32,
        combat: bool,
    ) -> EventResult {
        let mut result = EventResult::default();
        if amount <= 0 || self.damage_is_prevented(target) {
            return result;
        }
        let keywords = source
            .map(|object| self.source_keywords(object))
            .unwrap_or_default();
        let source_has_deathtouch = keywords.has(Keyword::Deathtouch);
        let combat_source = combat
            .then(|| source.and_then(|object_ref| self.object_event_ref(object_ref)))
            .flatten();

        match target {
            ProposedDamageTarget::Player(player) => {
                // Damage is still dealt to a player whose life total can't
                // change; only the life loss doesn't happen (CR 119.7).
                let locked = self.life_total_cant_change(player);
                let Some(life) = self.state.life_mut(player) else {
                    return result;
                };
                let old_life = *life;
                if !locked {
                    *life = life.saturating_sub(amount);
                }
                let new_life = *life;
                if let Some(source) = source.filter(|_| combat) {
                    result.combat_hit = Some(CombatPlayerHit {
                        source,
                        recipient: player,
                        amount,
                    });
                    self.state
                        .damage_history_mut()
                        .record_combat_player(source, player);
                }

                self.emit(GameEvent::DamageDealt {
                    source,
                    target: DamageTarget::Player(player),
                    amount: amount as u32,
                    combat,
                });
                if let Some(source) = combat_source {
                    self.emit_observation_event(GameEvent::CombatDamageDealt {
                        source,
                        target: EventSubject::Player(player),
                        amount: amount as u32,
                    });
                }
                if old_life != new_life {
                    self.emit(GameEvent::LifeChanged {
                        player,
                        old: old_life,
                        new: new_life,
                        gain_ordinal: 0,
                    });
                }
            }
            ProposedDamageTarget::Object(object_ref) => {
                let Ok(permanent_id) = self.lookup_current_permanent(object_ref) else {
                    return result;
                };
                // CR 120.3: a permanent with several of these types receives
                // every applicable result, but this remains one damage event.
                let view = self.layer_view(permanent_id);
                result.excess_damage = Some(self.excess_for_damage(
                    permanent_id,
                    &view,
                    amount,
                    source_has_deathtouch,
                ));
                let event_target = self.object_event_ref(object_ref);
                if let Some(source) = source {
                    self.state.damage_history_mut().record(source, object_ref);
                }
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
                    source,
                    target: DamageTarget::Permanent(permanent_id),
                    amount: amount as u32,
                    combat,
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

        result.combat_damage = source
            .filter(|_| combat)
            .map(|source| (source, amount as u32));
        result.committed = true;
        result
    }

    pub(crate) fn change_life(&mut self, player: PlayerId, delta: i32) -> bool {
        self.apply_immediate_event(ProposedEvent::LifeChange { player, delta })
    }

    pub(super) fn can_gain_life(&self, player: PlayerId) -> bool {
        !self.life_total_cant_change(player)
            && !self.static_sources().any(|(id, _, view)| {
                !self.is_phased_out(id)
                    && view
                        .static_abilities
                        .contains(&StaticAbility::PlayersCantGainLife)
            })
    }

    /// Apply the change and return the signed amount that actually occurred.
    pub(crate) fn commit_life_change(&mut self, player: PlayerId, delta: i32) -> i32 {
        let blocked = if delta > 0 {
            !self.can_gain_life(player)
        } else {
            self.life_total_cant_change(player)
        };
        if delta == 0 || blocked {
            return 0;
        }
        let Some(life) = self.state.life_mut(player) else {
            return 0;
        };
        let old_life = *life;
        *life = life.saturating_add(delta);
        let new_life = *life;
        if old_life == new_life {
            return 0;
        }
        self.emit(GameEvent::LifeChanged {
            player,
            old: old_life,
            new: new_life,
            gain_ordinal: 0,
        });
        new_life - old_life
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
    fn source_keywords(&self, source: ObjectId) -> Keywords {
        if let Ok(permanent_id) = self.lookup_current_permanent(source) {
            return self.layer_view(permanent_id).keywords;
        }
        if let Some(lki) = self.object_lki(source) {
            return lki.keywords;
        }
        if let Some(spell) = self.spell_on_stack(source) {
            return self.spell_keywords(source, spell.controller);
        }
        self.state.source_base(source).1.keywords
    }

    pub(crate) fn source_controller(&self, source: ObjectId) -> Option<PlayerId> {
        self.damage_source_facts(source)
            .map(|(controller, _)| controller)
    }

    /// Read controller and creature status from the same incarnation of the source.
    pub(crate) fn damage_source_facts(&self, source: ObjectId) -> Option<(PlayerId, bool)> {
        if let Ok(id) = self.lookup_current_permanent(source) {
            let permanent = self.state.permanents[id].as_ref()?;
            return Some((
                permanent.controller,
                self.permanent_is_battlefield_creature(id),
            ));
        }
        if let Some(controller) = self
            .spell_on_stack(source)
            .map(|spell| spell.controller)
            .or_else(|| {
                self.state
                    .spell_lki
                    .get(&source)
                    .map(|spell| spell.controller)
            })
        {
            return Some((
                controller,
                self.state.source_base(source).1.types.is_creature(),
            ));
        }
        self.object_lki(source)
            .map(|lki| (lki.controller, lki.was_creature()))
            .or_else(|| {
                let (_, owner, _) = self.state.objects.provenance(source)?;
                Some((owner, self.state.source_base(source).1.types.is_creature()))
            })
    }

    #[cfg(test)]
    pub(crate) fn gain_life(&mut self, player: PlayerId, amount: i32) {
        if amount > 0 {
            self.change_life(player, amount);
        }
    }

    /// The cleanup step (CR 514.2): "until end of turn" effects end and
    /// "once each turn" limits and "dealt damage this turn" reset.
    pub(crate) fn end_turn_modifiers(&mut self) {
        self.end_turn_effects();
        self.expire_turn_cast_grants();
        self.end_skipped_combat_phases(self.active_player());
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
