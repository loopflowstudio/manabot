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
        event::{DamageTarget, GameEvent},
        priority::PriorityState,
        trigger::PendingTrigger,
        turn::{StepKind, TurnState},
    },
    state::{
        ability::{Ability, TargetSpec, TriggerCondition, TriggerSource},
        card::Card,
        game_object::{CardId, CardVec, IdGenerator, PermanentId, PermanentVec, PlayerId, Target},
        mana::{Mana, ManaCost},
        permanent::Permanent,
        player::{Player, PlayerConfig},
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        target::Target as ActionTarget,
        zone::{ZoneManager, ZoneType},
    },
};

#[derive(Clone, Debug)]
pub struct GameState {
    pub cards: CardVec<Card>,
    pub permanents: PermanentVec<Option<Permanent>>,
    pub card_to_permanent: CardVec<Option<PermanentId>>,
    pub players: [Player; 2],
    pub zones: ZoneManager,
    pub turn: TurnState,
    pub priority: PriorityState,
    pub stack_objects: Vec<StackObject>,
    pub combat: Option<CombatState>,
    pub mana_cache: [Option<Mana>; 2],
    pub events: Vec<GameEvent>,
    pub pending_events: Vec<GameEvent>,
    pub pending_triggers: Vec<PendingTrigger>,
    pub pending_trigger_choice: Option<PendingTrigger>,
    pub trigger_enqueue_counter: u64,
    pub rng: ChaCha8Rng,
    pub id_gen: IdGenerator,
    pub card_registry: CardRegistry,
}

#[derive(Clone, Debug)]
pub enum PendingChoice {
    ChooseTarget {
        player: PlayerId,
        card: CardId,
        legal_targets: Vec<Target>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CombatDamagePass {
    FirstStrike,
    NormalWithFirstStrike,
    Normal,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub state: GameState,
    pub skip_trivial: bool,
    pub current_action_space: Option<ActionSpace>,
    pub pending_choice: Option<PendingChoice>,
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

        let mut cards = CardVec::default();
        let mut card_to_permanent = CardVec::default();
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
                permanents: PermanentVec::default(),
                card_to_permanent,
                players,
                zones,
                turn: TurnState::new(PlayerId(0)),
                priority: PriorityState::default(),
                stack_objects: Vec::new(),
                combat: None,
                mana_cache: [None, None],
                events: Vec::new(),
                pending_events: Vec::new(),
                pending_triggers: Vec::new(),
                pending_trigger_choice: None,
                trigger_enqueue_counter: 0,
                rng,
                id_gen,
                card_registry: registry,
            },
            skip_trivial,
            current_action_space: None,
            pending_choice: None,
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

    pub fn next_player(&self, player: PlayerId) -> PlayerId {
        PlayerId((player.0 + 1) % 2)
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
            && self.state.stack_objects.is_empty()
            && self.state.turn.can_cast_sorceries()
    }

    pub fn can_cast_instants(&self, _player: PlayerId) -> bool {
        // CR 117.1a — Any player with priority may cast an instant.
        true
    }

    pub fn can_play_land(&self, player: PlayerId) -> bool {
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to one per turn.
        self.can_cast_sorceries(player) && self.state.turn.lands_played < 1
    }

    pub fn can_pay_mana_cost(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.producible_mana(player).can_pay(cost)
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
        self.state.priority.sba_done = false;

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

    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.state.events)
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

        None
    }

    fn on_step_start(&mut self, step: StepKind) {
        self.state.priority.start_round(self.active_player());
        if step == StepKind::Untap {
            self.emit(GameEvent::TurnStarted {
                player: self.active_player(),
            });
        }
        self.emit(GameEvent::StepStarted { step });
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
                // The player who goes first skips their draw on turn 1.
                let is_first_player_first_turn =
                    self.state.turn.turn_number == 1 && active == PlayerId(0);
                if !is_first_player_first_turn {
                    self.draw_cards(active, 1);
                }
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
                let Some(blocker) = combat.blockers_to_declare.pop() else {
                    self.cleanup_illegal_menace_blocks();
                    return None;
                };
                let attackers = combat.attackers.clone();

                let mut actions = Vec::with_capacity(attackers.len() + 1);
                for attacker in &attackers {
                    if self.blocker_can_block_attacker(blocker, *attacker) {
                        actions.push(Action::DeclareBlocker {
                            player: defending,
                            blocker,
                            attacker: Some(*attacker),
                        });
                    }
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
                self.clear_temporary_modifiers();
                None
            }
            _ => None,
        }
    }

    fn tick_priority(&mut self) -> Option<ActionSpace> {
        loop {
            if let Some(choice_space) = self.pending_choice_action_space() {
                return Some(choice_space);
            }

            if !self.state.priority.sba_done {
                // CR 117.5, 704.3 — Check state-based actions before granting priority.
                self.perform_state_based_actions();
                self.state.priority.sba_done = true;
                if self.is_game_over() {
                    return None;
                }
            }

            // CR 603.3 — Flush pending triggers before granting priority.
            if self.state.pending_trigger_choice.is_some()
                || !self.state.pending_triggers.is_empty()
            {
                if let Some(space) = self.flush_triggers() {
                    return Some(space);
                }
            }

            if self.state.priority.consecutive_passes >= self.state.players.len() {
                self.state.priority.start_round(self.active_player());
                if !self.state.stack_objects.is_empty() {
                    // CR 117.4, 405.2 — If all players pass with a nonempty stack, resolve top object.
                    self.resolve_top_of_stack();
                    self.state.priority.on_non_pass_action(self.active_player());
                    continue;
                }
                return None;
            }

            let player = self.state.priority.holder;

            if self.skip_trivial && !self.can_player_act(player) {
                let next = self.next_player(player);
                self.state.priority.on_pass(next);
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
    }

    fn can_player_act(&mut self, player: PlayerId) -> bool {
        let mut producible = None;
        self.state
            .zones
            .zone_cards(ZoneType::Hand, player)
            .to_vec()
            .into_iter()
            .any(|card| {
                self.priority_action_for_card(player, card, &mut producible)
                    .is_some()
            })
            || self
                .priority_activate_ability_actions(player, &mut producible)
                .into_iter()
                .next()
                .is_some()
    }

    fn compute_player_actions(&mut self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        let mut producible = None;

        for card in self.state.zones.zone_cards(ZoneType::Hand, player).to_vec() {
            if let Some(action) = self.priority_action_for_card(player, card, &mut producible) {
                actions.push(action);
            }
        }

        actions.extend(self.priority_activate_ability_actions(player, &mut producible));
        actions.push(Action::PassPriority { player });
        actions
    }

    fn priority_activate_ability_actions(
        &mut self,
        player: PlayerId,
        producible: &mut Option<Mana>,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            for (ability_index, ability) in card.activated_abilities.iter().enumerate() {
                if producible.is_none() {
                    *producible = Some(self.producible_mana(player));
                }
                if producible
                    .as_ref()
                    .is_some_and(|mana| mana.can_pay(&ability.mana_cost))
                {
                    actions.push(Action::ActivateAbility {
                        player,
                        permanent: permanent_id,
                        ability_index,
                    });
                }
            }
        }
        actions
    }

    fn priority_action_for_card(
        &mut self,
        player: PlayerId,
        card_id: CardId,
        producible: &mut Option<Mana>,
    ) -> Option<Action> {
        let card = &self.state.cards[card_id];
        if card.types.is_land() {
            return self.can_play_land(player).then_some(Action::PlayLand {
                player,
                card: card_id,
            });
        }
        if !card.types.is_castable() {
            return None;
        }

        let can_cast_now = if card.types.is_instant_speed() {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }

        if self
            .legal_targets_for_spell(card_id)
            .is_some_and(|targets| targets.is_empty())
        {
            return None;
        }

        let mana_cost = card.mana_cost.clone();
        match mana_cost.as_ref() {
            Some(cost) => {
                if producible.is_none() {
                    *producible = Some(self.producible_mana(player));
                }
                producible
                    .as_ref()
                    .is_some_and(|m| m.can_pay(cost))
                    .then_some(Action::CastSpell {
                        player,
                        card: card_id,
                    })
            }
            None => Some(Action::CastSpell {
                player,
                card: card_id,
            }),
        }
    }

    fn pending_choice_action_space(&self) -> Option<ActionSpace> {
        let choice = self.pending_choice.as_ref()?;
        match choice {
            PendingChoice::ChooseTarget {
                player,
                card,
                legal_targets,
            } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::ChooseTarget,
                actions: legal_targets
                    .iter()
                    .copied()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target: match target {
                            Target::Player(p) => ActionTarget::Player(p),
                            Target::Permanent(p) => ActionTarget::Permanent(p),
                            Target::StackSpell(c) => ActionTarget::StackSpell(c),
                        },
                    })
                    .collect(),
                focus: vec![self.state.cards[card].id],
            }),
        }
    }

    fn legal_targets_for_spell(&self, card: CardId) -> Option<Vec<Target>> {
        let spec = self.state.cards[card].spell_effect.as_ref()?.target_spec()?;
        Some(self.legal_targets_for_target_spec(spec))
    }

    fn legal_targets_for_target_spec(&self, spec: &TargetSpec) -> Vec<Target> {
        match spec {
            TargetSpec::CreatureOrPlayer => {
                let mut targets = vec![Target::Player(PlayerId(0)), Target::Player(PlayerId(1))];
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            targets.push(Target::Permanent(permanent_id));
                        }
                    }
                }
                targets
            }
            TargetSpec::Creature => {
                let mut targets = Vec::new();
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            targets.push(Target::Permanent(permanent_id));
                        }
                    }
                }
                targets
            }
            TargetSpec::Spell => self
                .state
                .stack_objects
                .iter()
                .rev()
                .filter_map(|object| match object {
                    StackObject::Spell(spell) => Some(Target::StackSpell(spell.card)),
                    _ => None,
                })
                .collect(),
        }
    }

    fn execute_action(&mut self, action: &Action) -> Result<(), AgentError> {
        match action {
            Action::PlayLand { player, card } => {
                self.play_land(*player, *card)?;
                self.state.priority.on_non_pass_action(self.active_player());
                Ok(())
            }
            Action::CastSpell { player, card } => self.cast_spell_action(*player, *card),
            Action::ActivateAbility {
                player,
                permanent,
                ability_index,
            } => self.activate_ability_action(*player, *permanent, *ability_index),
            Action::ChooseTarget { player, target } => {
                let target = match *target {
                    ActionTarget::Player(p) => Target::Player(p),
                    ActionTarget::Permanent(p) => Target::Permanent(p),
                    ActionTarget::StackSpell(c) => Target::StackSpell(c),
                };
                if self.pending_choice.is_some() {
                    self.choose_target_action(*player, target)
                } else if self.state.pending_trigger_choice.is_some() {
                    self.choose_trigger_target(*player, target)
                } else {
                    Err(AgentError("no pending target choice".to_string()))
                }
            }
            Action::PassPriority { player } => {
                if self.state.priority.holder != *player {
                    return Err(AgentError("player does not have priority".to_string()));
                }
                let next = self.next_player(*player);
                self.state.priority.on_pass(next);
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

        let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
            return Err(AgentError("attacker permanent not found".to_string()));
        };
        let card = &self.state.cards[permanent.card];
        if !permanent.can_attack(card) {
            return Err(AgentError("permanent cannot attack".to_string()));
        }

        let controller = permanent.controller;
        permanent.attack_with_card(card);
        self.invalidate_mana_cache(controller);
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
            if !self.blocker_can_block_attacker(blocker, attacker_id) {
                return Err(AgentError(
                    "block declaration is illegal for this attacker/blocker pair".to_string(),
                ));
            }
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

    fn cast_spell_action(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        if self.pending_choice.is_some() {
            return Err(AgentError("a choice is already pending".to_string()));
        }

        let (is_land, owner, is_instant_speed) = {
            let card_ref = &self.state.cards[card];
            (
                card_ref.types.is_land(),
                card_ref.owner,
                card_ref.types.is_instant_speed(),
            )
        };

        if is_land {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        if is_instant_speed {
            if !self.can_cast_instants(player) {
                return Err(AgentError("cannot cast instant now".to_string()));
            }
        } else if !self.can_cast_sorceries(player) {
            return Err(AgentError(
                "cannot cast sorcery-speed spell now".to_string(),
            ));
        }

        if let Some(legal_targets) = self.legal_targets_for_spell(card) {
            // CR 601.2c — Choose target(s) as part of casting.
            if legal_targets.is_empty() {
                return Err(AgentError("no legal targets".to_string()));
            }
            self.pending_choice = Some(PendingChoice::ChooseTarget {
                player,
                card,
                legal_targets,
            });
            return Ok(());
        }

        self.pay_spell_cost(player, card)?;

        self.cast_spell(player, card, None)?;
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    fn choose_target_action(&mut self, player: PlayerId, target: Target) -> Result<(), AgentError> {
        let Some(PendingChoice::ChooseTarget {
            player: chooser,
            card,
            legal_targets,
        }) = self.pending_choice.as_ref()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };

        if *chooser != player {
            return Err(AgentError("wrong player for target choice".to_string()));
        }
        if !legal_targets.contains(&target) {
            return Err(AgentError("target is not legal".to_string()));
        }

        let card = *card;
        self.pay_spell_cost(player, card)?;

        self.cast_spell(player, card, Some(target))?;
        self.pending_choice = None;
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    fn choose_trigger_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let pending_trigger = self
            .state
            .pending_trigger_choice
            .take()
            .ok_or_else(|| AgentError("no pending target choice".to_string()))?;

        if pending_trigger.controller != player {
            return Err(AgentError("wrong player for target selection".to_string()));
        }

        let action_target = match target {
            Target::Player(p) => ActionTarget::Player(p),
            Target::Permanent(p) => ActionTarget::Permanent(p),
            Target::StackSpell(_) => {
                return Err(AgentError(
                    "invalid target for triggered ability".to_string(),
                ));
            }
        };

        let Some(target_spec) = self.trigger_target_spec(&pending_trigger) else {
            return Err(AgentError("triggered ability no longer exists".to_string()));
        };
        if !self.is_valid_target_for_spec(action_target, target_spec) {
            return Err(AgentError("selected target is not legal".to_string()));
        }

        self.place_triggered_ability_on_stack(pending_trigger, Some(target));
        Ok(())
    }

    fn pay_spell_cost(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let Some(cost) = self.state.cards[card].mana_cost.clone() else {
            return Ok(());
        };
        // CR 601.2f, 601.2h — Determine/pay costs as part of casting.
        self.produce_mana(player, &cost)?;
        self.spend_mana(player, &cost)
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

    fn spend_mana(&mut self, player: PlayerId, cost: &ManaCost) -> Result<(), AgentError> {
        if !self.state.players[player.0].mana_pool.can_pay(cost) {
            return Err(AgentError("insufficient mana in pool".to_string()));
        }
        self.state.players[player.0].mana_pool.pay(cost);
        Ok(())
    }

    fn activate_ability_action(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        if permanent.controller != player {
            return Err(AgentError(
                "source permanent is not controlled by player".to_string(),
            ));
        }
        if self.state.zones.zone_of(permanent.card) != Some(ZoneType::Battlefield) {
            return Err(AgentError(
                "source permanent must be on battlefield".to_string(),
            ));
        }

        let source_permanent_object_id = permanent.id;
        let source_card = permanent.card;
        let source_card_registry_key = self.state.cards[source_card].registry_key;
        let Some(ability) = self.state.cards[source_card]
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        self.produce_mana(player, &ability.mana_cost)?;
        self.spend_mana(player, &ability.mana_cost)?;

        let stack_object = StackObject::ActivatedAbility(ActivatedAbilityOnStack {
            id: self.state.id_gen.next_id(),
            controller: player,
            source_card_registry_key,
            source_card,
            source_permanent_object_id,
            ability_index,
            targets: Vec::new(),
        });
        self.state.stack_objects.push(stack_object);
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    fn cast_spell(
        &mut self,
        player: PlayerId,
        card: CardId,
        target: Option<Target>,
    ) -> Result<(), AgentError> {
        let owner = self.state.cards[card].owner;
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        // CR 601.2i — A cast spell is put onto the stack.
        self.move_card(card, ZoneType::Stack);
        let targets = target.into_iter().collect();
        let stack_object = StackObject::Spell(SpellOnStack {
            id: self.state.id_gen.next_id(),
            card,
            controller: player,
            source_card_registry_key: self.state.cards[card].registry_key,
            targets,
        });
        self.state.stack_objects.push(stack_object);
        self.emit(GameEvent::SpellCast { card, target });
        Ok(())
    }

    pub(crate) fn emit(&mut self, event: GameEvent) {
        self.state.events.push(event);
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

    fn clear_temporary_modifiers(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_temporary_modifiers();
        }
    }

    fn untap_all_permanents(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                permanent.untap();
            }
        }
        self.invalidate_mana_cache(player);
    }

    fn mark_permanents_not_summoning_sick(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
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
            let permanent = self.state.permanents[permanent_id].as_ref().unwrap();
            let card = &self.state.cards[permanent.card];
            // CR 704.5g — Creatures with lethal damage are destroyed.
            if permanent.has_lethal_damage(card) {
                to_destroy.push(permanent_id);
            }
        }

        for permanent_id in to_destroy {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
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

        let has_first_or_double_strike = self.combat_has_first_or_double_strike(&combat);
        if has_first_or_double_strike {
            self.resolve_combat_damage_pass(&combat, CombatDamagePass::FirstStrike);
            self.perform_state_based_actions();
            if !self.is_game_over() {
                self.resolve_combat_damage_pass(&combat, CombatDamagePass::NormalWithFirstStrike);
            }
        } else {
            self.resolve_combat_damage_pass(&combat, CombatDamagePass::Normal);
        }

        self.state.combat = Some(combat);
    }

    fn combat_has_first_or_double_strike(&self, combat: &CombatState) -> bool {
        for attacker_id in &combat.attackers {
            let Some(attacker) = self.state.permanents[*attacker_id].as_ref() else {
                continue;
            };
            let keywords = &self.state.cards[attacker.card].keywords;
            if keywords.first_strike || keywords.double_strike {
                return true;
            }
        }
        for blockers in combat.attacker_to_blockers.values() {
            for blocker_id in blockers {
                let Some(blocker) = self.state.permanents[*blocker_id].as_ref() else {
                    continue;
                };
                let keywords = &self.state.cards[blocker.card].keywords;
                if keywords.first_strike || keywords.double_strike {
                    return true;
                }
            }
        }
        false
    }

    fn resolve_combat_damage_pass(&mut self, combat: &CombatState, pass: CombatDamagePass) {
        let defender = self.non_active_player();

        for attacker_id in combat.attackers.iter().copied() {
            let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
                continue;
            };
            let attacker_card_id = attacker.card;
            let attacker_card = &self.state.cards[attacker_card_id];
            if !self.creature_deals_damage_in_pass(attacker_card, pass) {
                continue;
            }

            let attacker_power = attacker.effective_power(attacker_card).max(0);
            let attacker_has_trample = attacker_card.keywords.trample;
            let attacker_has_deathtouch = attacker_card.keywords.deathtouch;

            let declared_blockers = combat
                .attacker_to_blockers
                .get(&attacker_id)
                .cloned()
                .unwrap_or_default();
            let was_blocked = !declared_blockers.is_empty();
            let blockers: Vec<PermanentId> = declared_blockers
                .iter()
                .copied()
                .filter(|blocker_id| self.state.permanents[*blocker_id].is_some())
                .collect();

            if !was_blocked {
                self.apply_player_damage(Some(attacker_card_id), defender, attacker_power);
                continue;
            }

            for blocker_id in &blockers {
                let Some(blocker) = self.state.permanents[*blocker_id].as_ref() else {
                    continue;
                };
                let blocker_card = &self.state.cards[blocker.card];
                if !self.creature_deals_damage_in_pass(blocker_card, pass) {
                    continue;
                }
                let blocker_power = blocker.effective_power(blocker_card).max(0);
                self.apply_permanent_damage(Some(blocker.card), attacker_id, blocker_power);
            }

            let mut remaining_damage = attacker_power;
            for blocker_id in blockers {
                if remaining_damage <= 0 {
                    break;
                }
                let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
                    continue;
                };
                let blocker_card = &self.state.cards[blocker.card];
                let blocker_toughness = blocker.effective_toughness(blocker_card);
                let needed_damage = (blocker_toughness - blocker.damage).max(0);
                let lethal = if needed_damage == 0 {
                    0
                } else if attacker_has_deathtouch {
                    1
                } else {
                    needed_damage
                };
                let assigned = remaining_damage.min(lethal);
                self.apply_permanent_damage(Some(attacker_card_id), blocker_id, assigned);
                remaining_damage -= assigned;
            }

            if attacker_has_trample && remaining_damage > 0 {
                self.apply_player_damage(Some(attacker_card_id), defender, remaining_damage);
            }
        }
    }

    fn creature_deals_damage_in_pass(&self, card: &Card, pass: CombatDamagePass) -> bool {
        let has_first = card.keywords.first_strike;
        let has_double = card.keywords.double_strike;
        match pass {
            CombatDamagePass::FirstStrike => has_first || has_double,
            CombatDamagePass::NormalWithFirstStrike => !has_first || has_double,
            CombatDamagePass::Normal => true,
        }
    }

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

    fn eligible_attackers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            if permanent.can_attack(card) {
                out.push(permanent_id);
            }
        }
        out
    }

    fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            if permanent.can_block(card) {
                out.push(permanent_id);
            }
        }
        out
    }

    fn blocker_can_block_attacker(
        &self,
        blocker_id: PermanentId,
        attacker_id: PermanentId,
    ) -> bool {
        let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
            return false;
        };
        let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
            return false;
        };
        let blocker_card = &self.state.cards[blocker.card];
        let attacker_card = &self.state.cards[attacker.card];

        if !blocker.can_block(blocker_card) {
            return false;
        }

        if attacker_card.keywords.flying
            && !(blocker_card.keywords.flying || blocker_card.keywords.reach)
        {
            return false;
        }

        true
    }

    fn cleanup_illegal_menace_blocks(&mut self) {
        let Some(combat) = self.state.combat.as_mut() else {
            return;
        };
        for attacker_id in combat.attackers.clone() {
            let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
                continue;
            };
            let attacker_card = &self.state.cards[attacker.card];
            if !attacker_card.keywords.menace {
                continue;
            }
            let Some(blockers) = combat.attacker_to_blockers.get_mut(&attacker_id) else {
                continue;
            };
            if blockers.len() == 1 {
                blockers.clear();
            }
        }
    }

    fn producible_mana(&self, player: PlayerId) -> Mana {
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

    pub(crate) fn invalidate_mana_cache(&mut self, player: PlayerId) {
        self.state.mana_cache[player.0] = None;
    }

    fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card])
            .collect()
    }

    fn flush_triggers(&mut self) -> Option<ActionSpace> {
        while let Some(trigger) = self
            .state
            .pending_trigger_choice
            .take()
            .or_else(|| self.pop_next_pending_trigger())
        {
            let Some(target_spec) = self.trigger_target_spec(&trigger) else {
                continue;
            };

            let legal_targets = self.legal_targets_for_spec(target_spec);
            if legal_targets.is_empty() {
                // CR 603.3d — Triggered abilities with no legal required targets are removed.
                continue;
            }

            let controller = trigger.controller;
            self.state.pending_trigger_choice = Some(trigger);
            return Some(ActionSpace {
                player: Some(controller),
                kind: ActionSpaceKind::ChooseTarget,
                actions: legal_targets
                    .into_iter()
                    .map(|legal_target| Action::ChooseTarget {
                        player: controller,
                        target: legal_target,
                    })
                    .collect(),
                focus: Vec::new(),
            });
        }

        None
    }

    fn trigger_target_spec<'a>(&'a self, trigger: &PendingTrigger) -> Option<&'a TargetSpec> {
        let ability = self
            .state
            .cards
            .get(trigger.source_card.0)
            .and_then(|card| card.abilities.get(trigger.ability_index))?;

        let Ability::Triggered { effect, .. } = ability;
        effect.target_spec()
    }

    fn pop_next_pending_trigger(&mut self) -> Option<PendingTrigger> {
        let active = self.active_player();
        let next_index = self
            .state
            .pending_triggers
            .iter()
            .enumerate()
            .min_by_key(|(_, trigger)| {
                let apnap_rank = if trigger.controller == active {
                    0_u8
                } else {
                    1_u8
                };
                (apnap_rank, trigger.enqueue_order)
            })
            .map(|(index, _)| index)?;
        Some(self.state.pending_triggers.remove(next_index))
    }

    fn place_triggered_ability_on_stack(
        &mut self,
        trigger: PendingTrigger,
        target: Option<Target>,
    ) {
        let source_card_registry_key = self.state.cards[trigger.source_card].registry_key;
        let targets = target.into_iter().collect();
        self.state
            .stack_objects
            .push(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
                id: self.state.id_gen.next_id(),
                controller: trigger.controller,
                source_card: trigger.source_card,
                source_card_registry_key,
                ability_index: trigger.ability_index,
                targets,
            }));
        self.state.priority.start_round(self.active_player());
    }

    pub(crate) fn legal_targets_for_spec(&self, target_spec: &TargetSpec) -> Vec<ActionTarget> {
        match target_spec {
            TargetSpec::Creature | TargetSpec::CreatureOrPlayer => {
                let mut out = Vec::new();
                if matches!(target_spec, TargetSpec::CreatureOrPlayer) {
                    out.push(ActionTarget::Player(PlayerId(0)));
                    out.push(ActionTarget::Player(PlayerId(1)));
                }
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            out.push(ActionTarget::Permanent(permanent_id));
                        }
                    }
                }
                out
            }
            TargetSpec::Spell => {
                // Triggered abilities don't target spells, but handle it for completeness.
                Vec::new()
            }
        }
    }

    pub(crate) fn is_valid_target_for_spec(
        &self,
        target: ActionTarget,
        target_spec: &TargetSpec,
    ) -> bool {
        match (target, target_spec) {
            (ActionTarget::Player(_), TargetSpec::CreatureOrPlayer) => true,
            (
                ActionTarget::Permanent(permanent_id),
                TargetSpec::Creature | TargetSpec::CreatureOrPlayer,
            ) => {
                let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                    return false;
                };
                let card = &self.state.cards[permanent.card];
                card.types.is_creature()
                    && self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
            }
            _ => false,
        }
    }

    fn process_game_events(&mut self) {
        let events = std::mem::take(&mut self.state.pending_events);
        for event in events {
            if let GameEvent::CardMoved {
                card,
                from,
                to,
                controller,
            } = event
            {
                self.check_triggers_for_card_moved(card, from, to, controller);
            }
        }
    }

    fn check_triggers_for_card_moved(
        &mut self,
        card: CardId,
        from: Option<ZoneType>,
        to: ZoneType,
        controller: PlayerId,
    ) {
        let mut triggered_abilities = Vec::new();
        if let Some(source_card) = self.state.cards.get(card.0) {
            for (ability_index, ability) in source_card.abilities.iter().enumerate() {
                let Ability::Triggered {
                    condition,
                    intervening_if,
                    ..
                } = ability;

                if !self.trigger_condition_matches_card_moved(condition, from, to) {
                    continue;
                }
                if let Some(intervening) = intervening_if.as_ref() {
                    if !self.check_trigger_condition(intervening, card) {
                        continue;
                    }
                }
                triggered_abilities.push(ability_index);
            }
        }

        for ability_index in triggered_abilities {
            self.state.pending_triggers.push(PendingTrigger {
                source_card: card,
                ability_index,
                controller,
                enqueue_order: self.state.trigger_enqueue_counter,
            });
            self.state.trigger_enqueue_counter =
                self.state.trigger_enqueue_counter.saturating_add(1);
        }
    }

    fn trigger_condition_matches_card_moved(
        &self,
        condition: &TriggerCondition,
        from: Option<ZoneType>,
        to: ZoneType,
    ) -> bool {
        match condition {
            TriggerCondition::EntersTheBattlefield {
                source: TriggerSource::This,
            } => from != Some(ZoneType::Battlefield) && to == ZoneType::Battlefield,
        }
    }

    pub(crate) fn check_trigger_condition(
        &self,
        condition: &TriggerCondition,
        source_card: CardId,
    ) -> bool {
        match condition {
            TriggerCondition::EntersTheBattlefield {
                source: TriggerSource::This,
            } => self.state.zones.zone_of(source_card) == Some(ZoneType::Battlefield),
        }
    }

    pub fn move_card(&mut self, card: CardId, to_zone: ZoneType) {
        let owner = self.state.cards[card].owner;
        let old_zone = self.state.zones.zone_of(card);
        let mut event_controller = owner;

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.card_to_permanent[card].take() {
                if let Some(permanent) = self.state.permanents[permanent_id].as_ref() {
                    event_controller = permanent.controller;
                }
                self.state.permanents[permanent_id] = None;
            }
        }
        if old_zone == Some(ZoneType::Stack) {
            if let Some(index) = self.find_spell_on_stack_index(card) {
                self.state.stack_objects.remove(index);
            }
        }

        self.state.zones.move_card(card, owner, to_zone);

        if to_zone == ZoneType::Battlefield {
            let permanent_id = PermanentId(self.state.permanents.len());
            let permanent =
                Permanent::new(self.state.id_gen.next_id(), card, &self.state.cards[card]);
            event_controller = permanent.controller;
            self.state.permanents.push(Some(permanent));
            if self.state.card_to_permanent.len() <= card.0 {
                self.state.card_to_permanent.resize(card.0 + 1, None);
            }
            self.state.card_to_permanent[card] = Some(permanent_id);
        }

        self.state.pending_events.push(GameEvent::CardMoved {
            card,
            from: old_zone,
            to: to_zone,
            controller: event_controller,
        });
        self.process_game_events();

        if let Some(from) = old_zone {
            self.emit(GameEvent::CardMoved {
                card,
                from: Some(from),
                to: to_zone,
                controller: event_controller,
            });
        }
    }
}
