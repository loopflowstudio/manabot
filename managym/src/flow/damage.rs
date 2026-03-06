// damage.rs
// Damage application and clearing.

use crate::{
    flow::{
        event::{DamageTarget, GameEvent},
        game::Game,
    },
    state::game_object::{CardId, PermanentId, PlayerId},
};

impl Game {
    pub(crate) fn apply_player_damage(
        &mut self,
        source: Option<CardId>,
        player: PlayerId,
        amount: i32,
    ) {
        if amount <= 0 {
            return;
        }
        let lifelink_controller = source.and_then(|card| {
            self.source_has_lifelink(card)
                .then_some(self.state.cards[card].owner)
        });

        let Some(player_state) = self.state.players.get_mut(player.0) else {
            return;
        };
        let old_life = player_state.life;
        player_state.take_damage(amount);
        let new_life = player_state.life;

        self.emit(GameEvent::DamageDealt {
            source,
            target: DamageTarget::Player(player),
            amount: amount as u32,
        });
        self.emit(GameEvent::LifeChanged {
            player,
            old: old_life,
            new: new_life,
        });

        if let Some(controller) = lifelink_controller {
            self.gain_life(controller, amount);
        }
    }

    pub(crate) fn apply_permanent_damage(
        &mut self,
        source: Option<CardId>,
        permanent_id: PermanentId,
        amount: i32,
    ) {
        if amount <= 0 {
            return;
        }
        let source_has_deathtouch = source.is_some_and(|card| self.source_has_deathtouch(card));
        let lifelink_controller = source.and_then(|card| {
            self.source_has_lifelink(card)
                .then_some(self.state.cards[card].owner)
        });

        if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
            permanent.take_damage(amount);
            if source_has_deathtouch {
                permanent.deathtouch_damage = true;
            }
            self.emit(GameEvent::DamageDealt {
                source,
                target: DamageTarget::Permanent(permanent_id),
                amount: amount as u32,
            });
        }

        if let Some(controller) = lifelink_controller {
            self.gain_life(controller, amount);
        }
    }

    pub(crate) fn clear_damage(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_damage();
        }
    }

    fn source_has_lifelink(&self, source: CardId) -> bool {
        self.state.cards[source].keywords.lifelink
    }

    fn source_has_deathtouch(&self, source: CardId) -> bool {
        self.state.cards[source].keywords.deathtouch
    }

    fn gain_life(&mut self, player: PlayerId, amount: i32) {
        if amount <= 0 {
            return;
        }
        let Some(player_state) = self.state.players.get_mut(player.0) else {
            return;
        };
        let old_life = player_state.life;
        player_state.life += amount;
        let new_life = player_state.life;
        self.emit(GameEvent::LifeChanged {
            player,
            old: old_life,
            new: new_life,
        });
    }

    pub(crate) fn clear_temporary_modifiers(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_temporary_modifiers();
        }
    }
}
