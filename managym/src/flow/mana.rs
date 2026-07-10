// mana.rs
// Mana production, spending, and caching.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        game_object::PlayerId,
        mana::{Mana, ManaCost},
    },
};

impl Game {
    pub fn cached_producible_mana(&mut self, player: PlayerId) -> Mana {
        if let Some(cached) = &self.state.mana_cache[player.0] {
            return cached.clone();
        }
        let mana = self.producible_mana(player);
        self.state.mana_cache[player.0] = Some(mana.clone());
        mana
    }

    pub fn invalidate_mana_cache(&mut self, player: PlayerId) {
        self.state.mana_cache[player.0] = None;
    }

    /// Mana already in `player`'s pools (regular + until-end-of-combat)
    /// plus everything their untapped permanents could produce.
    pub(crate) fn available_mana(&self, player: PlayerId) -> Mana {
        let mut total = self.producible_mana(player);
        total.add(&self.state.players[player.0].mana_pool);
        total.add(&self.state.players[player.0].combat_mana_pool);
        total
    }

    /// Both pools combined (regular + until-end-of-combat).
    pub(crate) fn pooled_mana(&self, player: PlayerId) -> Mana {
        let mut total = self.state.players[player.0].mana_pool.clone();
        total.add(&self.state.players[player.0].combat_mana_pool);
        total
    }

    pub(crate) fn produce_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        if !self.available_mana(player).can_pay(cost) {
            return Err(AgentError("not enough producible mana".to_string()));
        }

        let permanents = self.battlefield_permanents(player);
        for permanent_id in permanents {
            if self.pooled_mana(player).can_pay(cost) {
                break;
            }

            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };

            let card = &self.state.cards[permanent.card];
            if permanent.tapped || card.mana_abilities.is_empty() || !permanent.can_tap(card) {
                continue;
            }
            let produced: Vec<_> = card
                .mana_abilities
                .iter()
                .map(|ability| ability.mana.clone())
                .collect();

            // CR 106.3 — Activate mana abilities to add mana to the mana pool.
            // Tapping for mana emits a PermanentTapped { for_mana: true } event.
            self.tap_permanent(permanent_id, true);
            for mana in &produced {
                self.state.players[player.0].mana_pool.add(mana);
            }
        }

        if !self.pooled_mana(player).can_pay(cost) {
            return Err(AgentError("failed to produce enough mana".to_string()));
        }

        Ok(())
    }

    /// Pay `cost` from the player's pools. Until-end-of-combat mana is
    /// spent first (it expires sooner).
    pub(crate) fn spend_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        let combined = self.pooled_mana(player);
        if !combined.can_pay(cost) {
            return Err(AgentError("insufficient mana in pool".to_string()));
        }
        let mut remaining = combined;
        remaining.pay(cost);

        // Rebuild the two pools from what's left: spent mana comes out of
        // the combat pool first, slot by slot.
        let player_state = &mut self.state.players[player.0];
        for slot in 0..remaining.mana.len() {
            let before =
                player_state.mana_pool.mana[slot] + player_state.combat_mana_pool.mana[slot];
            let spent = before - remaining.mana[slot];
            let from_combat = spent.min(player_state.combat_mana_pool.mana[slot]);
            player_state.combat_mana_pool.mana[slot] -= from_combat;
            player_state.mana_pool.mana[slot] -= spent - from_combat;
        }
        Ok(())
    }

    pub(crate) fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            total.add(&permanent.producible_mana(card));
        }
        total
    }

    /// CR 106.4 — regular pools empty as each step/phase ends. The combat
    /// pool (until-end-of-combat mana) survives; see
    /// `clear_combat_mana_pools`.
    pub(crate) fn clear_mana_pools(&mut self) {
        for player in &mut self.state.players {
            player.mana_pool.clear();
        }
    }

    /// Until-end-of-combat mana (firebending) empties as combat ends.
    pub(crate) fn clear_combat_mana_pools(&mut self) {
        for player in &mut self.state.players {
            player.combat_mana_pool.clear();
        }
    }
}
