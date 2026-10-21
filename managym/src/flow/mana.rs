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
    pub(crate) fn mana_ability_actions(
        &self,
        player: PlayerId,
        excluded: Option<crate::state::game_object::PermanentId>,
    ) -> Vec<crate::agent::action::Action> {
        let mut actions = Vec::new();
        for permanent in self.battlefield_permanents(player) {
            if Some(permanent) == excluded {
                continue;
            }
            let source = self.state.permanents[permanent].as_ref().unwrap();
            let card = &self.state.cards[source.card];
            for (ability_index, ability) in card.activated_abilities.iter().enumerate() {
                if ability.mana_choices().is_some() && (!ability.tap_source || source.can_tap(card))
                {
                    actions.push(crate::agent::action::Action::ActivateAbility {
                        player,
                        permanent,
                        ability_index,
                    });
                }
            }
        }
        actions
    }

    pub(crate) fn can_pay_with_mana_abilities(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        excluded: Option<crate::state::game_object::PermanentId>,
    ) -> bool {
        let available = self.available_mana(player);
        if available.can_pay(cost) {
            return true;
        }
        let mut by_source =
            std::collections::BTreeMap::<crate::state::game_object::PermanentId, Vec<Mana>>::new();
        for action in self.mana_ability_actions(player, excluded) {
            let crate::agent::action::Action::ActivateAbility {
                permanent,
                ability_index,
                ..
            } = action
            else {
                unreachable!()
            };
            let source = self.state.permanents[permanent].as_ref().unwrap();
            by_source.entry(permanent).or_default().extend(
                self.state.cards[source.card].activated_abilities[ability_index]
                    .mana_choices()
                    .unwrap(),
            );
        }
        let choices: Vec<Vec<Mana>> = by_source.into_values().collect();
        can_cover_cost(&choices, 0, available, cost, &mut Default::default())
    }

    /// A mandatory payment cannot publish a color that makes its announced
    /// costs impossible. The color choice itself has not committed any mana.
    pub(crate) fn mana_choice_completes_payment(&self, player: PlayerId, mana: &Mana) -> bool {
        let mut probe = self.clone();
        probe.state.players[player.0].mana_pool.add(mana);
        match &self.pending_choice {
            Some(crate::flow::game::PendingChoice::ManaPayment {
                payment,
                sacrifice,
                alternative,
                ..
            }) => probe.can_pay_with_mana_abilities(
                player,
                &self.payment_mana_cost(player, payment, alternative.as_ref()),
                *sacrifice,
            ),
            Some(crate::flow::game::PendingChoice::Waterbend {
                permanent,
                ability_index,
                remaining_generic,
                ..
            }) => {
                let source = self.state.permanents[*permanent].as_ref().unwrap();
                let cost = self.state.cards[source.card].activated_abilities[*ability_index]
                    .mana_cost
                    .with_generic(*remaining_generic);
                probe.can_pay_with_waterbend(
                    player,
                    &cost,
                    probe.waterbend_candidates(player).len(),
                )
            }
            _ => true,
        }
    }

    pub(crate) fn activate_mana_ability(
        &mut self,
        player: PlayerId,
        permanent: crate::state::game_object::PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if !self.mana_ability_actions(player, None).contains(
            &crate::agent::action::Action::ActivateAbility {
                player,
                permanent,
                ability_index,
            },
        ) {
            return Err(AgentError("illegal mana activation".into()));
        }
        let source = self.state.permanents[permanent].as_ref().unwrap();
        let ability = self.state.cards[source.card].activated_abilities[ability_index].clone();
        let choices = ability.mana_choices().unwrap();
        if ability.tap_source {
            self.tap_permanent(permanent, true);
        }
        if ability.sacrifice_source {
            self.sacrifice_permanent(permanent);
        }
        let old = self.state.suspended_decision.take();
        self.suspend_rule_decision(crate::flow::decision::Decision::ManaChoice {
            player,
            choices,
            resume: old
                .as_ref()
                .map(|suspended| Box::new(suspended.decision.clone())),
        });
        if let Some(old) = old {
            self.state.suspended_decision.as_mut().unwrap().frame = old.frame;
        }
        Ok(())
    }

    pub fn cached_producible_mana(&mut self, player: PlayerId) -> Mana {
        if let Some(cached) = &self.state.mana_cache[player.0] {
            return cached.clone();
        }
        let mana = self.producible_mana(player);
        self.journal_mana_cache(player);
        self.state.mana_cache[player.0] = Some(mana.clone());
        mana
    }

    pub fn invalidate_mana_cache(&mut self, player: PlayerId) {
        self.journal_mana_cache(player);
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
            self.journal_player(player);
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
        self.journal_player(player);
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
        for player in [PlayerId(0), PlayerId(1)] {
            self.journal_player(player);
            self.state.players[player.0].mana_pool.clear();
        }
    }

    /// Until-end-of-combat mana (firebending) empties as combat ends.
    pub(crate) fn clear_combat_mana_pools(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            self.journal_player(player);
            self.state.players[player.0].combat_mana_pool.clear();
        }
    }
}

/// Memoize colored resources rather than expanding all permutations of tokens.
/// Each physical source contributes at most one of its alternative outputs.
fn can_cover_cost(
    choices: &[Vec<Mana>],
    index: usize,
    mut pool: Mana,
    cost: &ManaCost,
    visited: &mut std::collections::BTreeSet<(usize, [u8; 6])>,
) -> bool {
    if pool.can_pay(cost) {
        return true;
    }
    for amount in &mut pool.mana {
        *amount = (*amount).min(cost.mana_value);
    }
    if index == choices.len() || !visited.insert((index, pool.mana)) {
        return false;
    }
    choices[index].iter().any(|mana| {
        let mut next = pool.clone();
        next.add(mana);
        can_cover_cost(choices, index + 1, next, cost, visited)
    })
}
