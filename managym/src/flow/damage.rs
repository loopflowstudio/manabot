// damage.rs
// Damage application, life changes, and cleanup.

use crate::state::card::{Keyword, Keywords};
use crate::{
    flow::{
        event::{CombatPlayerHit, DamageTarget, EventSubject, GameEvent},
        game::Game,
        proposed_event::{EventResult, ProposedDamageTarget, ProposedEvent},
    },
    state::{
        ability::DelayedExpiry,
        card::StaticAbility,
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
    /// Commit each lifelink source's total from one simultaneous damage
    /// event as a single gain, in first-damage order: a creature that deals
    /// damage to several recipients at once gains life once (CR 119.9), and
    /// two lifelink sources are two gains.
    pub(crate) fn commit_lifelink_gains(&mut self, gains: impl IntoIterator<Item = LifelinkGain>) {
        let mut totals: Vec<LifelinkGain> = Vec::new();
        for gain in gains {
            match totals.iter_mut().find(|total| total.source == gain.source) {
                Some(total) => total.amount = total.amount.saturating_add(gain.amount),
                None => totals.push(gain),
            }
        }
        for total in totals {
            self.commit_life_change(total.controller, total.amount);
        }
    }

    /// Excess after replacement, measured against the pre-commit creature.
    /// Deathtouch makes one remaining damage lethal (CR 120.4a).
    pub(crate) fn excess_for_damage(
        &self,
        source: Option<ObjectId>,
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
        let deathtouch =
            source.is_some_and(|object| self.source_keywords(object).has(Keyword::Deathtouch));
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
        if amount <= 0 {
            return result;
        }
        let keywords = source
            .map(|object| self.source_keywords(object))
            .unwrap_or_default();
        let source_has_deathtouch = keywords.has(Keyword::Deathtouch);
        let lifelink = source
            .filter(|_| keywords.has(Keyword::Lifelink))
            .and_then(|source| {
                self.source_controller(source)
                    .map(|controller| LifelinkGain {
                        source,
                        controller,
                        amount,
                    })
            });
        let combat_source = combat
            .then(|| source.and_then(|object_ref| self.object_event_ref(object_ref)))
            .flatten();

        match target {
            ProposedDamageTarget::Player(player) => {
                // Protection from everything prevents all damage (CR 702.16e).
                if self.has_protection_from_everything(player) {
                    return result;
                }
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
                        .turn
                        .damage_pairs
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
                result.excess_damage = Some(self.excess_for_damage(source, permanent_id, amount));
                let event_target = self.object_event_ref(object_ref);
                if let Some(source) = source {
                    self.state.turn.damage_pairs.record(source, object_ref);
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

        // The batch commits the gain: one source's damage dealt at the same
        // time is one life-gain event (CR 120.3f, 119.9).
        result.lifelink = lifelink;
        result.committed = true;
        result
    }

    pub(crate) fn change_life(&mut self, player: PlayerId, delta: i32) -> bool {
        self.apply_immediate_event(ProposedEvent::LifeChange { player, delta })
    }

    /// Apply the change and return the signed amount that actually occurred.
    pub(crate) fn commit_life_change(&mut self, player: PlayerId, delta: i32) -> i32 {
        if self.life_total_cant_change(player) {
            return 0;
        }
        if delta > 0
            && [PlayerId(0), PlayerId(1)].into_iter().any(|owner| {
                self.battlefield_permanents(owner).into_iter().any(|id| {
                    self.characteristics(id)
                        .static_abilities
                        .contains(&StaticAbility::PlayersCantGainLife)
                })
            })
        {
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
            return self.characteristics(permanent_id).keywords;
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
        if self.state.live(source).is_some() {
            if let Some(spell) = self.spell_on_stack(source) {
                return Some((spell.controller, self.state[source].types.is_creature()));
            }
        }
        if let Some(spell) = self.state.spell_lki.get(&source) {
            return Some((
                spell.controller,
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
