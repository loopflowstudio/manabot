// damage.rs
// Damage application, life changes, and cleanup.

use crate::{
    flow::{
        event::{DamageTarget, GameEvent},
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
        });
    }

    pub(crate) fn commit_damage(
        &mut self,
        source: Option<ObjectRef>,
        target: ProposedDamageTarget,
        amount: i32,
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

        match target {
            ProposedDamageTarget::Player(player) => {
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
                self.emit(GameEvent::LifeChanged {
                    player,
                    old: old_life,
                    new: new_life,
                });
            }
            ProposedDamageTarget::Object(object_ref) => {
                let Ok(permanent_id) = self.lookup_current_permanent(object_ref) else {
                    return false;
                };
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
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_damage();
        }
    }

    /// Damage-source keywords include until-EOT grants while the source is
    /// on the battlefield (Enter the Avatar State's lifelink).
    fn source_keywords(&self, source: ObjectRef) -> Option<crate::state::card::Keywords> {
        if let Ok(permanent_id) = self.lookup_current_permanent(source) {
            return Some(self.effective_keywords(permanent_id));
        }
        if let Some(definition) = self.object_lki_definition(source) {
            return Some(definition.keywords.clone());
        }
        let card = CardId::from(source.entity);
        self.state
            .cards
            .get(card.0)
            .map(|card| card.keywords.clone())
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

    pub(crate) fn clear_temporary_modifiers(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_temporary_modifiers();
        }
    }
}
