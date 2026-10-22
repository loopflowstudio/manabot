// mana.rs
// Mana production, spending, and caching.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        ability::Effect,
        card::ActivatedAbilityDefinition,
        game_object::{PermanentId, PlayerId},
        mana::{Mana, ManaCost},
    },
};

type ManaChoices = Vec<(ManaCost, Mana)>;
type PaymentSearchState = (Vec<usize>, [u8; 6], [u8; 6]);

impl Game {
    /// Printed abilities followed by currently granted abilities, in source order.
    pub(crate) fn activated_abilities_for(
        &self,
        permanent: PermanentId,
    ) -> Vec<ActivatedAbilityDefinition> {
        let Some(source) = self.state.permanents[permanent].as_ref() else {
            return Vec::new();
        };
        let mut abilities = self.state.cards[source.card].activated_abilities.clone();
        for grant in self.battlefield_permanents(source.controller) {
            let grant = self.state.permanents[grant].as_ref().unwrap();
            if self.state.cards[grant.card]
                .mana_grant
                .iter()
                .any(|predicate| self.permanent_matches_predicate(permanent, predicate))
            {
                abilities.push(ActivatedAbilityDefinition {
                    tap_source: true,
                    effect: Effect::any_color_mana(),
                    ..Default::default()
                });
            }
        }
        abilities
    }

    pub(crate) fn mana_ability_actions(
        &self,
        player: PlayerId,
        excluded: Option<PermanentId>,
    ) -> Vec<crate::agent::action::Action> {
        let mut actions = Vec::new();
        for permanent in self.battlefield_permanents(player) {
            if Some(permanent) == excluded || Some(permanent) == self.pending_mana_source() {
                continue;
            }
            let source = self.state.permanents[permanent].as_ref().unwrap();
            let card = &self.state.cards[source.card];
            for (ability_index, ability) in
                self.activated_abilities_for(permanent).iter().enumerate()
            {
                let Some(choices) = ability.mana_choices() else {
                    continue;
                };
                if (!ability.tap_source || source.can_tap(card))
                    && self
                        .available_mana_without(player, Some(permanent))
                        .can_pay(&ability.mana_cost)
                {
                    // A committed cost must retain at least one legal completion
                    // after this source is consumed, before publishing its color choice.
                    if self.pending_choice.is_some() || self.state.suspended_decision.is_some() {
                        let mut probe = self.clone();
                        if probe
                            .pay_mana_ability_cost(player, permanent, ability)
                            .is_err()
                        {
                            continue;
                        }
                        if !choices
                            .iter()
                            .any(|mana| probe.mana_choice_completes_payment(player, mana))
                        {
                            continue;
                        }
                    }
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
        excluded: Option<PermanentId>,
    ) -> bool {
        let pool = self.pooled_mana(player);
        if pool.can_pay(cost) {
            return true;
        }
        let mut groups: Vec<(ManaChoices, usize)> = Vec::new();
        for permanent in self.battlefield_permanents(player) {
            if Some(permanent) == excluded || Some(permanent) == self.pending_mana_source() {
                continue;
            }
            let source = self.state.permanents[permanent].as_ref().unwrap();
            let card = &self.state.cards[source.card];
            let mut choices = Vec::new();
            if source.can_tap(card) && !card.mana_abilities.is_empty() {
                choices.push((ManaCost::default(), self.fixed_source_mana(permanent)));
            }
            for ability in self.activated_abilities_for(permanent) {
                if ability.tap_source && !source.can_tap(card) {
                    continue;
                }
                if let Some(outputs) = ability.mana_choices() {
                    choices.extend(outputs.into_iter().map(|mut output| {
                        if ability.tap_source {
                            output.add(&self.tap_mana_bonus(permanent, player));
                        }
                        (ability.mana_cost.clone(), output)
                    }));
                }
            }
            if choices.is_empty() {
                continue;
            }
            if let Some((_, count)) = groups.iter_mut().find(|(existing, _)| *existing == choices) {
                *count += 1;
            } else {
                groups.push((choices, 1));
            }
        }
        let counts = groups.iter().map(|(_, count)| *count).collect();
        can_cover_cost(&groups, counts, pool, cost, &mut Default::default())
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
                probe.can_pay_with_waterbend(player, &cost)
            }
            _ => match self
                .state
                .suspended_decision
                .as_ref()
                .and_then(|s| s.decision.payment())
            {
                // Only a waterbend contribution commits an optional payment.
                Some(payment) if !payment.can_decline() => {
                    probe.can_pay_with_waterbend(player, &payment.remaining.mana_cost())
                }
                _ => true,
            },
        }
    }

    pub(crate) fn activate_mana_ability(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
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
        let ability = self.activated_abilities_for(permanent)[ability_index].clone();
        let choices = ability.mana_choices().unwrap();
        self.pay_mana_ability_cost(player, permanent, &ability)?;
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

    /// Share the upfront cost between completion probes and real activations.
    fn pay_mana_ability_cost(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
        ability: &ActivatedAbilityDefinition,
    ) -> Result<(), AgentError> {
        if ability.tap_source {
            self.tap_permanent(permanent, true);
        }
        self.produce_mana(player, &ability.mana_cost)?;
        self.spend_mana(player, &ability.mana_cost)?;
        if ability.sacrifice_source {
            self.sacrifice_permanent(permanent);
        }
        Ok(())
    }

    /// Fixed tap output plus immediate triggered bonuses; chosen outputs stay
    /// in the payment search until the player selects a color.
    fn fixed_source_mana(&self, permanent: PermanentId) -> Mana {
        let source = self.state.permanents[permanent].as_ref().unwrap();
        let card = &self.state.cards[source.card];
        let mut output = source.producible_mana(card);
        if source.can_tap(card) && !card.mana_abilities.is_empty() {
            output.add(&self.tap_mana_bonus(permanent, source.controller));
        }
        output
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
        self.available_mana_without(player, None)
    }

    fn available_mana_without(&self, player: PlayerId, excluded: Option<PermanentId>) -> Mana {
        let mut total = Mana::default();
        for permanent in self.battlefield_permanents(player) {
            if Some(permanent) == excluded || Some(permanent) == self.pending_mana_source() {
                continue;
            }
            total.add(&self.fixed_source_mana(permanent));
        }
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
        self.produce_mana_without(player, cost, None)
    }

    pub(crate) fn produce_mana_without(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
        excluded: Option<PermanentId>,
    ) -> Result<(), AgentError> {
        if !self.available_mana_without(player, excluded).can_pay(cost) {
            return Err(AgentError("not enough producible mana".to_string()));
        }

        let permanents = self.battlefield_permanents(player);
        for permanent_id in permanents {
            if Some(permanent_id) == excluded {
                continue;
            }
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

    /// Pay `cost` from eligible pools. Within each color, restricted mana is
    /// spent first; combat mana is spent first within each restriction class.
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

        // Rebuild the pools by restriction class and color, consuming the
        // expiring combat pool before the regular pool within each class.
        self.journal_player(player);
        let player_state = &mut self.state.players[player.0];
        for slot in 0..remaining.mana.len() {
            let restricted_spent = player_state.mana_pool.lesson[slot]
                + player_state.combat_mana_pool.lesson[slot]
                - remaining.lesson[slot];
            let combat = restricted_spent.min(player_state.combat_mana_pool.lesson[slot]);
            player_state.combat_mana_pool.lesson[slot] -= combat;
            player_state.mana_pool.lesson[slot] -= restricted_spent - combat;
            let spent = player_state.mana_pool.mana[slot]
                + player_state.combat_mana_pool.mana[slot]
                - remaining.mana[slot];
            let combat = spent.min(player_state.combat_mana_pool.mana[slot]);
            player_state.combat_mana_pool.mana[slot] -= combat;
            player_state.mana_pool.mana[slot] -= spent - combat;
        }
        Ok(())
    }

    pub(crate) fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        for permanent_id in self.battlefield_permanents(player) {
            total.add(&self.fixed_source_mana(permanent_id));
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

/// Search payment sequences, grouping identical sources to avoid land permutations.
/// Activation costs are paid before outputs exist; a filter cannot seed itself.
fn can_cover_cost(
    groups: &[(ManaChoices, usize)],
    counts: Vec<usize>,
    pool: Mana,
    cost: &ManaCost,
    visited: &mut std::collections::BTreeSet<PaymentSearchState>,
) -> bool {
    if pool.can_pay(cost) {
        return true;
    }
    if !visited.insert((counts.clone(), pool.mana, pool.lesson)) {
        return false;
    }
    for (index, (choices, _)) in groups.iter().enumerate() {
        if counts[index] == 0 {
            continue;
        }
        for (activation, output) in choices {
            if !pool.can_pay(activation) {
                continue;
            }
            let mut next = pool.clone();
            next.pay(activation);
            next.add(output);
            let mut remaining = counts.clone();
            remaining[index] -= 1;
            if can_cover_cost(groups, remaining, next, cost, visited) {
                return true;
            }
        }
    }
    false
}
