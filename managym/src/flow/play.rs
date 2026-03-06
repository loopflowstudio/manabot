// play.rs
// Spell casting and land plays.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        game_object::{CardId, PlayerId},
        mana::ManaCost,
        zone::ZoneType,
    },
};

impl Game {
    pub fn can_cast_sorceries(&self, player: PlayerId) -> bool {
        // CR 117.1a, 307.1 — Sorcery-speed actions are available only to the active player
        // during a main phase with an empty stack.
        self.is_active_player(player)
            && self.state.stack.is_empty()
            && self.state.turn.can_cast_sorceries()
    }

    pub fn can_play_land(&self, player: PlayerId) -> bool {
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to one per turn.
        self.can_cast_sorceries(player) && self.state.turn.lands_played < 1
    }

    pub fn can_pay_mana_cost(&mut self, player: PlayerId, cost: &ManaCost) -> bool {
        self.cached_producible_mana(player).can_pay(cost)
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
        let card_ref = &self.state.cards[card];
        if card_ref.types.is_land() {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if card_ref.owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }

        if let Some(cost) = card_ref.mana_cost.clone() {
            // CR 601.2f, 601.2h — Determine/pay costs as part of casting.
            self.produce_mana(player, &cost)?;
            self.spend_mana(player, &cost)?;
        }

        self.cast_spell(player, card)
    }

    pub(crate) fn cast_spell(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let owner = self.state.cards[card].owner;
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        // CR 601.2i — A cast spell is put onto the stack.
        self.push_spell_to_stack(card);
        Ok(())
    }
}
