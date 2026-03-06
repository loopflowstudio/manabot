use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, ActionType},
    flow::turn::{PhaseKind, StepKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        mana::Mana,
        permanent::Permanent,
        player::PlayerConfig,
        target::Target,
        zone::ZoneType,
    },
    Game,
};

const MAX_SCENARIO_ACTIONS: usize = 20_000;

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

fn mono_land_deck(land_name: &str) -> BTreeMap<String, usize> {
    deck(&[(land_name, 40)])
}

fn land_plus_spell_deck(land_name: &str, spell_name: &str) -> BTreeMap<String, usize> {
    deck(&[(land_name, 24), (spell_name, 16)])
}

pub fn mountain_deck() -> BTreeMap<String, usize> {
    mono_land_deck("Mountain")
}

pub fn island_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Island".to_string(), 40)])
}

pub fn forest_deck() -> BTreeMap<String, usize> {
    mono_land_deck("Forest")
}

pub fn forest_elves_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Forest".to_string(), 24),
        ("Llanowar Elves".to_string(), 36),
    ])
}

pub fn bolt_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Lightning Bolt")
}

pub fn counterspell_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Island", "Counterspell")
}
pub fn ogre_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Grey Ogre")
}

pub fn ogre_only_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Grey Ogre".to_string(), 40)])
}

pub fn manowar_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Island".to_string(), 24), ("Man-o'-War".to_string(), 16)])
}

pub fn wind_drake_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Island", "Wind Drake")
}

pub fn giant_spider_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Forest", "Giant Spider")
}

pub fn raging_goblin_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Raging Goblin")
}

pub fn serra_angel_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Plains", "Serra Angel")
}

pub fn typhoid_rats_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Swamp", "Typhoid Rats")
}

pub fn war_mammoth_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Forest", "War Mammoth")
}

pub fn wall_of_stone_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Wall of Stone")
}

pub fn boggart_brute_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Boggart Brute")
}

pub fn youthful_knight_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Plains", "Youthful Knight")
}

pub fn fencing_ace_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Plains", "Fencing Ace")
}

pub fn hawk_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Plains", "Healer's Hawk")
}

pub fn craw_wurm_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Forest", "Craw Wurm")
}

pub fn shivan_deck() -> BTreeMap<String, usize> {
    land_plus_spell_deck("Mountain", "Shivan Dragon")
}

pub fn empty_deck() -> BTreeMap<String, usize> {
    BTreeMap::new()
}

pub struct Scenario {
    game: Game,
}

impl Scenario {
    pub fn new(
        player0_deck: BTreeMap<String, usize>,
        player1_deck: BTreeMap<String, usize>,
        seed: u64,
    ) -> Scenario {
        let p0 = PlayerConfig::new("p0", player0_deck);
        let p1 = PlayerConfig::new("p1", player1_deck);
        Scenario {
            game: Game::new(vec![p0, p1], seed, false),
        }
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn game_mut(&mut self) -> &mut Game {
        &mut self.game
    }

    pub fn action_space(&self) -> &ActionSpace {
        self.game
            .action_space()
            .expect("scenario should have an active action space")
    }

    pub fn current_step(&self) -> StepKind {
        self.game.state.turn.current_step_kind()
    }

    pub fn current_phase(&self) -> PhaseKind {
        self.game.state.turn.current_phase_kind()
    }

    pub fn active_player(&self) -> usize {
        self.game.active_player().0
    }

    pub fn life(&self, player: usize) -> i32 {
        self.game.state.players[player].life
    }

    pub fn zone_size(&self, player: usize, zone: ZoneType) -> usize {
        self.game.state.zones.size(zone, PlayerId(player))
    }

    pub fn assert_life(&self, player: usize, expected: i32) {
        assert_eq!(
            self.life(player),
            expected,
            "unexpected life for player {player}"
        );
    }

    pub fn assert_zone_size(&self, player: usize, zone: ZoneType, expected: usize) {
        assert_eq!(
            self.zone_size(player, zone),
            expected,
            "unexpected zone size for player {player} in {zone:?}"
        );
    }

    pub fn assert_action_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_some(),
            "expected action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn assert_action_not_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_none(),
            "did not expect action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn step_action(&mut self, index: usize) {
        self.game.step(index).expect("scenario step should succeed");
    }

    pub fn take_action_by_type(&mut self, action_type: ActionType) -> bool {
        let Some(index) = self.action_index_by_type(action_type) else {
            return false;
        };
        self.step_action(index);
        true
    }

    pub fn pass_priority(&mut self) {
        assert!(
            self.take_action_by_type(ActionType::PriorityPassPriority),
            "priority pass action should be available"
        );
    }

    pub fn assert_game_over(&self) {
        assert!(self.game.is_game_over(), "game should be over");
    }

    pub fn assert_winner(&self, player: usize) {
        assert_eq!(self.game.winner_index(), Some(player));
    }

    pub fn advance_default_action(&mut self) {
        let space = self.action_space().clone();
        let index = match space.kind {
            ActionSpaceKind::Priority => space
                .actions
                .iter()
                .position(|action| action.action_type() == ActionType::PriorityPassPriority)
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareAttacker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareBlocker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            // Picks first legal target arbitrarily — tests that care about targeting
            // should use choose_target_named or explicit action selection instead.
            ActionSpaceKind::ChooseTarget => 0,
            ActionSpaceKind::GameOver => 0,
        };
        self.step_action(index);
    }

    pub fn advance_to_step(&mut self, target: StepKind) {
        self.advance_until(
            |scenario| scenario.current_step() == target,
            format!("failed to reach step {target:?}"),
        );
    }

    pub fn advance_to_phase(&mut self, target: PhaseKind) {
        self.advance_until(
            |scenario| scenario.current_phase() == target,
            format!("failed to reach phase {target:?}"),
        );
    }

    pub fn advance_to_active_step(&mut self, active_player: usize, target: StepKind) {
        self.advance_until(
            |scenario| {
                scenario.active_player() == active_player && scenario.current_step() == target
            },
            format!("failed to reach step {target:?} for player {active_player}"),
        );
    }

    pub fn force_card_in_hand(&mut self, player: usize, card_name: &str) {
        let Some(index) = self
            .game
            .state
            .cards
            .iter()
            .position(|card| card.owner == PlayerId(player) && card.name == card_name)
        else {
            panic!("card {card_name} not found for player {player}");
        };

        self.game
            .state
            .zones
            .move_card(CardId(index), PlayerId(player), ZoneType::Hand);
    }

    pub fn battlefield_permanents_named(&self, player: usize, card_name: &str) -> Vec<PermanentId> {
        self.game
            .state
            .zones
            .zone_cards(ZoneType::Battlefield, PlayerId(player))
            .iter()
            .filter_map(|card_id| {
                let card = &self.game.state.cards[*card_id];
                if card.name != card_name {
                    return None;
                }
                self.game.state.card_to_permanent[*card_id]
            })
            .collect()
    }

    pub fn force_permanent_on_battlefield(
        &mut self,
        player: usize,
        card_name: &str,
    ) -> PermanentId {
        let mut chosen_card_id = None;
        let mut existing_battlefield_permanent = None;
        for (index, card) in self.game.state.cards.iter().enumerate() {
            if card.owner != PlayerId(player) || card.name != card_name {
                continue;
            }
            let card_id = CardId(index);
            if self.game.state.zones.zone_of(card_id) == Some(ZoneType::Battlefield) {
                if let Some(permanent_id) = self.game.state.card_to_permanent[card_id] {
                    if existing_battlefield_permanent.is_none() {
                        existing_battlefield_permanent = Some(permanent_id);
                    }
                    continue;
                }
            }
            chosen_card_id = Some(card_id);
            break;
        }

        let Some(card_id) = chosen_card_id else {
            if let Some(permanent_id) = existing_battlefield_permanent {
                return permanent_id;
            }
            panic!("card {card_name} not found for player {player}");
        };

        if self.game.state.zones.zone_of(card_id) == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.game.state.card_to_permanent[card_id] {
                return permanent_id;
            }
        } else {
            self.game
                .state
                .zones
                .move_card(card_id, PlayerId(player), ZoneType::Battlefield);
        }

        let permanent_id = PermanentId(self.game.state.permanents.len());
        let permanent = Permanent::new(
            self.game.state.id_gen.next_id(),
            card_id,
            &self.game.state.cards[card_id],
        );
        self.game.state.permanents.push(Some(permanent));
        if self.game.state.card_to_permanent.len() <= card_id.0 {
            self.game
                .state
                .card_to_permanent
                .resize(card_id.0 + 1, None);
        }
        self.game.state.card_to_permanent[card_id] = Some(permanent_id);
        permanent_id
    }

    pub fn set_player_mana_pool(&mut self, player: usize, mana: &str) {
        self.game.state.players[player].mana_pool = Mana::parse(mana);
    }

    /// Play a land and cast a creature from a player's hand in their main phase.
    /// Resolves the spell (both players pass priority) and returns with the creature on
    /// the battlefield. Expects the player already has the named cards in hand.
    pub fn play_land_and_cast_creature(
        &mut self,
        player: usize,
        land_name: &str,
        creature_name: &str,
    ) {
        self.advance_to_active_step(player, StepKind::Main);
        self.force_card_in_hand(player, land_name);
        self.force_card_in_hand(player, creature_name);
        assert!(self.take_action_by_type(ActionType::PriorityPlayLand));
        assert!(self.take_action_by_type(ActionType::PriorityCastSpell));
        self.pass_priority();
        self.pass_priority();
    }

    /// Set up a creature that has survived summoning sickness and is ready to attack.
    /// Casts the creature on player 0's first turn, advances through player 1's turn,
    /// and arrives at player 0's declare attackers step.
    pub fn setup_attacker_ready(&mut self) {
        self.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
        self.advance_to_active_step(1, StepKind::Main);
        self.advance_to_active_step(0, StepKind::DeclareAttackers);
    }

    /// Declare the first available creature as an attacker.
    pub fn declare_attack(&mut self) {
        let attack_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
            .expect("attack action should exist");
        self.step_action(attack_index);
    }

    /// Decline to attack with the first available creature.
    pub fn decline_attack(&mut self) {
        let decline_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
            .expect("decline-attack action should exist");
        self.step_action(decline_index);
    }

    /// Assign the first available blocker to an attacker.
    pub fn declare_block(&mut self) {
        let block_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| {
                matches!(
                    action,
                    Action::DeclareBlocker {
                        attacker: Some(_),
                        ..
                    }
                )
            })
            .expect("block action should exist");
        self.step_action(block_index);
    }

    /// Decline to block with the first available creature.
    pub fn decline_block(&mut self) {
        let decline_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
            .expect("no-block action should exist");
        self.step_action(decline_index);
    }

    pub fn choose_target_named(&mut self, card_name: &str) {
        let choose_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| {
                let Action::ChooseTarget { target, .. } = action else {
                    return false;
                };
                let Target::Permanent(permanent_id) = target else {
                    return false;
                };
                let Some(permanent) = self.game.state.permanents[*permanent_id].as_ref() else {
                    return false;
                };
                let card = &self.game.state.cards[permanent.card];
                card.name == card_name
            })
            .expect("target choice for named permanent should exist");
        self.step_action(choose_index);
    }

    fn action_index_by_type(&self, action_type: ActionType) -> Option<usize> {
        self.action_space()
            .actions
            .iter()
            .position(|action| action.action_type() == action_type)
    }

    pub fn advance_until<F>(&mut self, mut predicate: F, failure_message: String)
    where
        F: FnMut(&Scenario) -> bool,
    {
        for _ in 0..MAX_SCENARIO_ACTIONS {
            if self.game.is_game_over() || predicate(self) {
                return;
            }
            self.advance_default_action();
        }
        panic!("{failure_message}");
    }
}
