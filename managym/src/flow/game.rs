use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{
        action::{Action, ActionSpace, ActionSpaceKind, AgentError},
        behavior_tracker::BehaviorTracker,
    },
    cardsets::alpha::CardRegistry,
    flow::{
        combat::CombatState,
        priority::PriorityState,
        turn::{StepKind, TurnState},
    },
    state::{
        card::Card,
        game_object::{CardId, IdGenerator, PermanentId, PlayerId},
        mana::{Mana, ManaCost},
        permanent::Permanent,
        player::{Player, PlayerConfig},
        zone::{ZoneManager, ZoneType},
    },
};

#[derive(Clone, Debug)]
pub struct GameState {
    pub cards: Vec<Card>,
    pub permanents: Vec<Option<Permanent>>,
    pub card_to_permanent: Vec<Option<PermanentId>>,
    pub players: [Player; 2],
    pub zones: ZoneManager,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub combat: Option<CombatState>,
    pub mana_cache: [Option<Mana>; 2],
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub card_registry: CardRegistry,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub state: GameState,
    pub skip_trivial: bool,
    pub current_action_space: Option<ActionSpace>,
    pub skip_trivial_count: usize,
    pub trackers: [BehaviorTracker; 2],
}

impl Game {
    pub fn new(player_configs: Vec<PlayerConfig>, seed: u64, skip_trivial: bool) -> Self {
        assert_eq!(player_configs.len(), 2, "game supports exactly two players");

        let mut id_gen = IdGenerator::default();
        let registry = CardRegistry::default();

        let mut players = [
            Player::new(id_gen.next_id(), 0, player_configs[0].name.clone()),
            Player::new(id_gen.next_id(), 1, player_configs[1].name.clone()),
        ];

        let mut cards = Vec::new();
        let mut card_to_permanent = Vec::new();
        let mut zones = ZoneManager::default();

        for (player_index, config) in player_configs.iter().enumerate() {
            for (name, qty) in &config.decklist {
                for _ in 0..*qty {
                    let card = registry
                        .instantiate(name, PlayerId(player_index), id_gen.next_id())
                        .unwrap_or_else(|| panic!("unknown card in decklist: {name}"));
                    let card_id = CardId(cards.len());
                    cards.push(card);
                    card_to_permanent.push(None);
                    players[player_index].deck.push(card_id);
                }
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for player in [PlayerId(0), PlayerId(1)] {
            let deck = players[player.0].deck.clone();
            for card in deck {
                zones.move_card(card, player, ZoneType::Library);
            }
            zones.shuffle(ZoneType::Library, player, &mut rng);
            // CR 103.4, 103.5 — Each player shuffles then draws an opening hand of seven cards.
            for _ in 0..7 {
                if let Some(card) = zones.top(ZoneType::Library, player) {
                    zones.move_card(card, player, ZoneType::Hand);
                }
            }
        }

        let mut game = Self {
            state: GameState {
                cards,
                permanents: Vec::new(),
                card_to_permanent,
                players,
                zones,
                turn: TurnState::new(PlayerId(0)),
                priority: PriorityState::default(),
                combat: None,
                mana_cache: [None, None],
                rng,
                id_gen,
                card_registry: registry,
            },
            skip_trivial,
            current_action_space: None,
            skip_trivial_count: 0,
            trackers: [BehaviorTracker::new(false), BehaviorTracker::new(false)],
        };

        game.trackers[0].on_game_start();
        game.trackers[1].on_game_start();

        let _ = game.tick();
        game
    }

    pub fn action_space(&self) -> Option<&ActionSpace> {
        self.current_action_space.as_ref()
    }

    pub fn active_player(&self) -> PlayerId {
        self.state.turn.active_player
    }

    pub fn non_active_player(&self) -> PlayerId {
        PlayerId((self.state.turn.active_player.0 + 1) % 2)
    }

    pub fn agent_player(&self) -> PlayerId {
        self.current_action_space
            .as_ref()
            .and_then(|space| space.player)
            .unwrap_or(PlayerId(0))
    }

    pub fn players_starting_with_active(&self) -> [PlayerId; 2] {
        [self.active_player(), self.non_active_player()]
    }

    pub fn players_starting_with_agent(&self) -> [PlayerId; 2] {
        let agent = self.agent_player();
        [agent, PlayerId((agent.0 + 1) % 2)]
    }

    pub fn is_active_player(&self, player: PlayerId) -> bool {
        player == self.active_player()
    }

    pub fn is_game_over(&self) -> bool {
        self.state.players.iter().filter(|p| p.alive).count() < 2
    }

    pub fn winner_index(&self) -> Option<usize> {
        if !self.is_game_over() {
            return None;
        }
        self.state.players.iter().position(|p| p.alive)
    }

    pub fn can_cast_sorceries(&self, player: PlayerId) -> bool {
        // CR 117.1a, 307.1 — Sorcery-speed actions are available only to the active player
        // during a main phase with an empty stack.
        self.is_active_player(player)
            && self.state.zones.stack_order().is_empty()
            && self.state.turn.can_cast_sorceries()
    }

    pub fn can_play_land(&self, player: PlayerId) -> bool {
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to one per turn.
        self.can_cast_sorceries(player) && self.state.turn.lands_played < 1
    }

    pub fn can_pay_mana_cost(&mut self, player: PlayerId, cost: &ManaCost) -> bool {
        self.cached_producible_mana(player).can_pay(cost)
    }

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

    pub fn step(&mut self, action: usize) -> Result<bool, AgentError> {
        if self.is_game_over() {
            return Err(AgentError("game is over".to_string()));
        }

        let action_space = self
            .current_action_space
            .take()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;

        if action >= action_space.actions.len() {
            self.current_action_space = Some(action_space.clone());
            return Err(AgentError(format!(
                "Action index {action} out of bounds: {}",
                action_space.actions.len()
            )));
        }

        let selected_action = action_space.actions[action].clone();
        if let Err(error) = self.execute_action(&selected_action) {
            self.current_action_space = Some(action_space);
            return Err(error);
        }

        let game_over = self.tick();
        if game_over {
            if let Some(winner) = self.winner_index() {
                self.trackers[winner].on_game_won();
            }
        }

        Ok(game_over)
    }

    pub fn play(&mut self) {
        while !self.is_game_over() {
            let _ = self.step(0);
        }
    }

    fn tick(&mut self) -> bool {
        loop {
            let action_space = self.turn_tick();
            if self.is_game_over() {
                self.current_action_space = Some(ActionSpace::game_over());
                return true;
            }

            if let Some(space) = action_space {
                if !self.skip_trivial || space.actions.len() > 1 {
                    self.current_action_space = Some(space);
                    return false;
                }

                self.skip_trivial_count += 1;
                if let Some(action) = space.actions.first() {
                    if self.execute_action(action).is_err() {
                        return true;
                    }
                }
                continue;
            }
        }
    }

    fn turn_tick(&mut self) -> Option<ActionSpace> {
        let step = self.state.turn.current_step_kind();

        if !self.state.turn.step_initialized {
            self.on_step_start(step);
            self.state.turn.step_initialized = true;
            self.state.turn.turn_based_actions_complete = false;
        }

        if !self.state.turn.turn_based_actions_complete {
            if let Some(space) = self.perform_turn_based_actions(step) {
                return Some(space);
            }
            self.state.turn.turn_based_actions_complete = true;
        }

        if TurnState::step_has_priority(step) {
            if let Some(space) = self.tick_priority() {
                return Some(space);
            }
        }

        // CR 106.4 — Unspent mana empties at the end of each step and phase.
        self.clear_mana_pools();
        self.on_step_end(step);
        self.state.turn.advance_step();
        self.state.priority.reset();

        None
    }

    fn on_step_start(&mut self, step: StepKind) {
        self.state.priority.reset();
        match step {
            StepKind::BeginningOfCombat => {
                // CR 507.1 — Beginning of combat creates/refreshes combat state.
                self.state.combat = Some(CombatState::default());
            }
            StepKind::DeclareAttackers => {
                // CR 508.1 — Active player declares attackers.
                let active = self.active_player();
                let eligible = self.eligible_attackers(active);
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                combat.attackers_to_declare = eligible;
            }
            StepKind::DeclareBlockers => {
                // CR 509.1 — Defending player declares blockers.
                let defender = self.non_active_player();
                let eligible = self.eligible_blockers(defender);
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                combat.blockers_to_declare = eligible;
            }
            _ => {}
        }
    }

    fn on_step_end(&mut self, step: StepKind) {
        if matches!(step, StepKind::EndOfCombat) {
            // CR 511.3 — Creatures stop being attacking as combat ends.
            for permanent in self.state.permanents.iter_mut().flatten() {
                permanent.attacking = false;
            }
            self.state.combat = None;
        }
    }

    fn perform_turn_based_actions(&mut self, step: StepKind) -> Option<ActionSpace> {
        match step {
            StepKind::Untap => {
                // CR 502.2 — Active player untaps permanents they control.
                let active = self.active_player();
                self.mark_permanents_not_summoning_sick(active);
                self.untap_all_permanents(active);
                None
            }
            StepKind::Draw => {
                // CR 504.1 — Active player draws one card in the draw step.
                let active = self.active_player();
                self.draw_cards(active, 1);
                None
            }
            StepKind::DeclareAttackers => {
                let active = self.active_player();
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                let attacker = combat.attackers_to_declare.pop()?;
                Some(ActionSpace {
                    player: Some(active),
                    kind: ActionSpaceKind::DeclareAttacker,
                    actions: vec![
                        Action::DeclareAttacker {
                            player: active,
                            permanent: attacker,
                            attack: true,
                        },
                        Action::DeclareAttacker {
                            player: active,
                            permanent: attacker,
                            attack: false,
                        },
                    ],
                    focus: Vec::new(),
                })
            }
            StepKind::DeclareBlockers => {
                let defending = self.non_active_player();
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                let blocker = combat.blockers_to_declare.pop()?;

                let mut actions = Vec::with_capacity(combat.attackers.len() + 1);
                for attacker in &combat.attackers {
                    actions.push(Action::DeclareBlocker {
                        player: defending,
                        blocker,
                        attacker: Some(*attacker),
                    });
                }
                actions.push(Action::DeclareBlocker {
                    player: defending,
                    blocker,
                    attacker: None,
                });

                Some(ActionSpace {
                    player: Some(defending),
                    kind: ActionSpaceKind::DeclareBlocker,
                    actions,
                    focus: Vec::new(),
                })
            }
            StepKind::CombatDamage => {
                // CR 510.1 — Assign and deal combat damage.
                self.resolve_combat_damage();
                None
            }
            StepKind::Cleanup => {
                // CR 514.2 — Damage marked on permanents is removed during cleanup.
                self.clear_damage();
                None
            }
            _ => None,
        }
    }

    fn tick_priority(&mut self) -> Option<ActionSpace> {
        if !self.state.priority.sba_done {
            // CR 117.5, 704.3 — Check state-based actions before granting priority.
            self.perform_state_based_actions();
            self.state.priority.sba_done = true;
            if self.is_game_over() {
                return None;
            }
        }

        let players = self.players_starting_with_active();

        while self.state.priority.pass_count < players.len() {
            let player = players[self.state.priority.pass_count];

            if self.skip_trivial && !self.can_player_act(player) {
                self.state.priority.pass_count += 1;
                continue;
            }

            let actions = self.compute_player_actions(player);
            return Some(ActionSpace {
                player: Some(player),
                kind: ActionSpaceKind::Priority,
                actions,
                focus: Vec::new(),
            });
        }

        self.state.priority.reset();
        if !self.state.zones.stack_order().is_empty() {
            // CR 117.4, 405.2 — If all players pass with a nonempty stack, resolve top object.
            self.resolve_top_of_stack();
            return self.tick_priority();
        }

        None
    }

    fn can_player_act(&mut self, player: PlayerId) -> bool {
        let can_play_land = self.can_play_land(player);
        let can_cast = self.can_cast_sorceries(player);

        let hand = self.state.zones.zone_cards(ZoneType::Hand, player).to_vec();

        let mut producible: Option<Mana> = None;

        for card_id in hand {
            let (is_land, is_castable, mana_cost) = {
                let card = &self.state.cards[card_id.0];
                (
                    card.types.is_land(),
                    card.types.is_castable(),
                    card.mana_cost.clone(),
                )
            };
            if is_land {
                if can_play_land {
                    return true;
                }
                continue;
            }

            if !is_castable || !can_cast {
                continue;
            }

            if let Some(cost) = mana_cost.as_ref() {
                if producible.is_none() {
                    producible = Some(self.cached_producible_mana(player));
                }
                if producible.as_ref().is_some_and(|m| m.can_pay(cost)) {
                    return true;
                }
            } else {
                return true;
            }
        }

        false
    }

    fn compute_player_actions(&mut self, player: PlayerId) -> Vec<Action> {
        let hand = self.state.zones.zone_cards(ZoneType::Hand, player).to_vec();

        let can_play_land = self.can_play_land(player);
        let can_cast = self.can_cast_sorceries(player);

        let mut actions = Vec::new();
        let mut producible: Option<Mana> = None;

        for card_id in hand {
            let (is_land, is_castable, mana_cost) = {
                let card = &self.state.cards[card_id.0];
                (
                    card.types.is_land(),
                    card.types.is_castable(),
                    card.mana_cost.clone(),
                )
            };
            if is_land {
                if can_play_land {
                    actions.push(Action::PlayLand {
                        player,
                        card: card_id,
                    });
                }
                continue;
            }

            if !is_castable || !can_cast {
                continue;
            }

            match mana_cost.as_ref() {
                Some(cost) => {
                    if producible.is_none() {
                        producible = Some(self.cached_producible_mana(player));
                    }
                    if producible.as_ref().is_some_and(|m| m.can_pay(cost)) {
                        actions.push(Action::CastSpell {
                            player,
                            card: card_id,
                        });
                    }
                }
                None => actions.push(Action::CastSpell {
                    player,
                    card: card_id,
                }),
            }
        }

        actions.push(Action::PassPriority { player });
        actions
    }

    fn execute_action(&mut self, action: &Action) -> Result<(), AgentError> {
        match action {
            Action::PlayLand { player, card } => self.play_land(*player, *card),
            Action::CastSpell { player, card } => self.cast_spell_action(*player, *card),
            Action::PassPriority { .. } => {
                self.state.priority.pass_priority();
                Ok(())
            }
            Action::DeclareAttacker {
                permanent, attack, ..
            } => self.declare_attacker(*permanent, *attack),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => self.declare_blocker(*blocker, *attacker),
        }
    }

    fn declare_attacker(
        &mut self,
        permanent_id: PermanentId,
        attack: bool,
    ) -> Result<(), AgentError> {
        if !attack {
            return Ok(());
        }

        let Some(permanent) = self.state.permanents[permanent_id.0].as_mut() else {
            return Err(AgentError("attacker permanent not found".to_string()));
        };
        let card = &self.state.cards[permanent.card.0];
        if !permanent.can_attack(card) {
            return Err(AgentError("permanent cannot attack".to_string()));
        }

        permanent.attack();
        if let Some(combat) = self.state.combat.as_mut() {
            combat.attackers.push(permanent_id);
            combat.attacker_to_blockers.entry(permanent_id).or_default();
        }
        Ok(())
    }

    fn declare_blocker(
        &mut self,
        blocker: PermanentId,
        attacker: Option<PermanentId>,
    ) -> Result<(), AgentError> {
        if let Some(attacker_id) = attacker {
            if let Some(combat) = self.state.combat.as_mut() {
                combat
                    .attacker_to_blockers
                    .entry(attacker_id)
                    .or_default()
                    .push(blocker);
            }
        }
        Ok(())
    }

    fn play_land(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let card_ref = &self.state.cards[card.0];
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

    fn cast_spell_action(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let card_ref = &self.state.cards[card.0];
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

    fn produce_mana(&mut self, player: PlayerId, cost: &ManaCost) -> Result<(), AgentError> {
        let producible = self.producible_mana(player);
        if !producible.can_pay(cost) {
            return Err(AgentError("not enough producible mana".to_string()));
        }

        let permanents = self.battlefield_permanents(player);
        for permanent_id in permanents {
            if self.state.players[player.0].mana_pool.can_pay(cost) {
                break;
            }

            let Some(permanent) = self.state.permanents[permanent_id.0].as_mut() else {
                continue;
            };

            let card = &self.state.cards[permanent.card.0];
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

    fn spend_mana(&mut self, player: PlayerId, cost: &ManaCost) -> Result<(), AgentError> {
        if !self.state.players[player.0].mana_pool.can_pay(cost) {
            return Err(AgentError("insufficient mana in pool".to_string()));
        }
        self.state.players[player.0].mana_pool.pay(cost);
        Ok(())
    }

    fn cast_spell(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let owner = self.state.cards[card.0].owner;
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        // CR 601.2i — A cast spell is put onto the stack.
        self.move_card(card, ZoneType::Stack);
        Ok(())
    }

    fn resolve_top_of_stack(&mut self) {
        let Some(card) = self.state.zones.stack_order().last().copied() else {
            return;
        };

        let is_permanent = self.state.cards[card.0].types.is_permanent();
        if is_permanent {
            // CR 608.3 — A resolving permanent spell enters the battlefield.
            self.move_card(card, ZoneType::Battlefield);
            let owner = self.state.cards[card.0].owner;
            self.invalidate_mana_cache(owner);
        } else {
            // CR 608.2k — Nonpermanent spells resolve then go to graveyard.
            self.move_card(card, ZoneType::Graveyard);
        }
    }

    pub fn clear_mana_pools(&mut self) {
        for player in &mut self.state.players {
            player.mana_pool.clear();
        }
    }

    fn clear_damage(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_damage();
        }
    }

    fn untap_all_permanents(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if let Some(permanent) = self.state.permanents[permanent_id.0].as_mut() {
                permanent.untap();
            }
        }
        self.invalidate_mana_cache(player);
    }

    fn mark_permanents_not_summoning_sick(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if let Some(permanent) = self.state.permanents[permanent_id.0].as_mut() {
                permanent.summoning_sick = false;
            }
        }
    }

    fn draw_cards(&mut self, player: PlayerId, amount: usize) {
        for _ in 0..amount {
            if self.state.zones.size(ZoneType::Library, player) == 0 {
                self.state.players[player.0].drew_when_empty = true;
                break;
            }

            if let Some(card) = self.state.zones.top(ZoneType::Library, player) {
                self.move_card(card, ZoneType::Hand);
            }
        }
    }

    fn lose_game(&mut self, player: PlayerId) {
        self.state.players[player.0].alive = false;
    }

    fn perform_state_based_actions(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            // CR 704.5a, 704.5b — A player loses at 0 or less life or for drawing from empty library.
            if self.state.players[player.0].life <= 0
                || self.state.players[player.0].drew_when_empty
            {
                self.lose_game(player);
            }
        }

        if self.is_game_over() {
            return;
        }

        let mut to_destroy = Vec::new();
        for permanent_id in self
            .state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(idx, perm)| perm.as_ref().map(|_| PermanentId(idx)))
        {
            let permanent = self.state.permanents[permanent_id.0].as_ref().unwrap();
            let card = &self.state.cards[permanent.card.0];
            // CR 704.5g — Creatures with lethal damage are destroyed.
            if permanent.has_lethal_damage(card) {
                to_destroy.push(permanent_id);
            }
        }

        for permanent_id in to_destroy {
            let Some(permanent) = self.state.permanents[permanent_id.0].as_ref() else {
                continue;
            };
            let card = permanent.card;
            let controller = permanent.controller;
            self.move_card(card, ZoneType::Graveyard);
            self.invalidate_mana_cache(controller);
        }
    }

    fn resolve_combat_damage(&mut self) {
        let Some(combat) = self.state.combat.take() else {
            return;
        };

        for (attacker_id, blockers) in &combat.attacker_to_blockers {
            let Some(attacker) = self.state.permanents[attacker_id.0].as_ref() else {
                continue;
            };
            let attacker_power = self.state.cards[attacker.card.0].power.unwrap_or(0);

            if blockers.is_empty() {
                // CR 510.1c — Unblocked attackers assign combat damage to defending player.
                let defender = self.non_active_player();
                self.state.players[defender.0].take_damage(attacker_power);
                continue;
            }

            for blocker_id in blockers {
                // CR 510.1a — Combat damage is dealt simultaneously by attacking and blocking creatures.
                let Some(blocker) = self.state.permanents[blocker_id.0].as_ref() else {
                    continue;
                };
                let blocker_power = self.state.cards[blocker.card.0].power.unwrap_or(0);
                self.apply_permanent_damage(*attacker_id, blocker_power);
                self.apply_permanent_damage(*blocker_id, attacker_power);
            }
        }

        self.state.combat = Some(combat);
    }

    fn apply_permanent_damage(&mut self, permanent_id: PermanentId, amount: i32) {
        if let Some(permanent) = self.state.permanents[permanent_id.0].as_mut() {
            permanent.take_damage(amount);
        }
    }

    fn eligible_attackers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id.0].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card.0];
            if permanent.can_attack(card) {
                out.push(permanent_id);
            }
        }
        out
    }

    fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id.0].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card.0];
            if permanent.can_block(card) {
                out.push(permanent_id);
            }
        }
        out
    }

    fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id.0].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card.0];
            total.add(&permanent.producible_mana(card));
        }
        total
    }

    fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card.0])
            .collect()
    }

    fn move_card(&mut self, card: CardId, to_zone: ZoneType) {
        let owner = self.state.cards[card.0].owner;
        let old_zone = self.state.zones.zone_of(card);

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.card_to_permanent[card.0].take() {
                self.state.permanents[permanent_id.0] = None;
            }
        }

        self.state.zones.move_card(card, owner, to_zone);

        if to_zone == ZoneType::Battlefield {
            let permanent_id = PermanentId(self.state.permanents.len());
            let permanent =
                Permanent::new(self.state.id_gen.next_id(), card, &self.state.cards[card.0]);
            self.state.permanents.push(Some(permanent));
            if self.state.card_to_permanent.len() <= card.0 {
                self.state.card_to_permanent.resize(card.0 + 1, None);
            }
            self.state.card_to_permanent[card.0] = Some(permanent_id);
        }
    }
}
