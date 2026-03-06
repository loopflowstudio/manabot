// mana.rs
// Mana production, spending, and caching.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        game_object::{PermanentId, PlayerId},
        mana::{Mana, ManaCost},
        zone::ZoneType,
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

    pub(crate) fn produce_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        let producible = self.producible_mana(player);
        if !producible.can_pay(cost) {
            return Err(AgentError("not enough producible mana".to_string()));
        }

        let permanents = self.battlefield_permanents(player);
        for permanent_id in permanents {
            if self.state.players[player.0].mana_pool.can_pay(cost) {
                break;
            }

            let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
                continue;
            };

            let card = &self.state.cards[permanent.card];
            if permanent.tapped || card.mana_abilities.is_empty() || !permanent.can_tap(card) {
                continue;
            }

            // CR 106.3 — Activate mana abilities to add mana to the mana pool.
            permanent.tap();
            for ability in &card.mana_abilities {
                self.state.players[player.0].mana_pool.add(&ability.mana);
            }
            self.invalidate_mana_cache(player);
        }

        if !self.state.players[player.0].mana_pool.can_pay(cost) {
            return Err(AgentError("failed to produce enough mana".to_string()));
        }

        Ok(())
    }

    pub(crate) fn spend_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        if !self.state.players[player.0].mana_pool.can_pay(cost) {
            return Err(AgentError("insufficient mana in pool".to_string()));
        }
        self.state.players[player.0].mana_pool.pay(cost);
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

    pub(crate) fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card])
            .collect()
    }
}
