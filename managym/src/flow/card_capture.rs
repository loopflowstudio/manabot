// card_capture.rs
// Private game facts at setup and committed boundaries; no rule events or RNG.

use crate::agent::{
    action::{Action, ActionSpace, ActionType},
    observation::Observation,
};
use crate::flow::decision::EffectFrame;
#[cfg(test)]
use crate::state::card::SpellProfile;
use crate::{
    cards::content_pack::default_content_pack,
    flow::game::Game,
    state::{
        game_object::{ObjectId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};
use serde::{Deserialize, Serialize};

/// Both modes retain card movements. Exposure also records decisions and turn facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptureMode {
    Cards,
    Exposure,
}

/// One consumed, nontrivial native choice. Focus is the shared action projection,
/// not a claim that each focused card caused or benefited from the action.
#[derive(Clone, Debug, Serialize)]
pub struct ExposureChoice {
    pub revision: u64,
    pub actor: usize,
    pub kind: crate::agent::action::ActionSpaceKind,
    pub options: Vec<ExposureOption>,
    pub selected: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExposureOption {
    pub action_type: ActionType,
    pub objects: Vec<CardSnapshot>,
}

/// A completed effect frame, including untargeted and suspended resolutions.
/// Completion does not assert an effect changed state or a strategy succeeded.
#[derive(Clone, Debug, Serialize)]
pub struct ExposureResolution {
    pub revision: u64,
    pub controller: usize,
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub source: Option<ObjectId>,
    pub definition: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExposureCapture {
    pub choices: Vec<ExposureChoice>,
    pub resolutions: Vec<ExposureResolution>,
    pub mana: [ManaUse; 2],
    pub mana_turns: [Vec<TurnMana>; 2],
    pub attackers_committed: [u64; 2],
    pub attackers_lost_for_nothing: [u64; 2],
    pub blockers_committed: [u64; 2],
    pub blockers_lost_for_nothing: [u64; 2],
    #[serde(skip)]
    pub(crate) combat: CombatDeclarations,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TurnMana {
    pub available: u64,
    pub paid: u64,
}

/// Permanent ids identify battlefield incarnations even across a blink.
#[derive(Clone, Debug, Default)]
pub(crate) struct CombatDeclarations {
    pub attackers: Vec<PermanentId>,
    pub blocks: Vec<(PermanentId, PermanentId)>,
    pub deaths: std::collections::BTreeSet<PermanentId>,
}

impl ExposureCapture {
    pub(crate) fn paid(&mut self, player: PlayerId, amount: u64) {
        if let Some(turn) = self.mana_turns[player.0].last_mut() {
            turn.paid += amount;
        }
    }

    pub(crate) fn finish_combat(&mut self, attacking: PlayerId, defending: PlayerId) {
        let combat = std::mem::take(&mut self.combat);
        self.attackers_committed[attacking.0] += combat.attackers.len() as u64;
        self.blockers_committed[defending.0] += combat.blocks.len() as u64;
        for attacker in combat.attackers {
            if combat.deaths.contains(&attacker)
                && !combat
                    .blocks
                    .iter()
                    .any(|(blocker, target)| *target == attacker && combat.deaths.contains(blocker))
            {
                self.attackers_lost_for_nothing[attacking.0] += 1;
            }
        }
        for (blocker, attacker) in combat.blocks {
            if combat.deaths.contains(&blocker) && !combat.deaths.contains(&attacker) {
                self.blockers_lost_for_nothing[defending.0] += 1;
            }
        }
    }
}

/// A pooled unit's origin names its source, not whether it was floated at
/// priority. Track the range of priority-floated units that can remain after
/// each spend, without changing the rules' mana classes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ManaUse {
    pub priority_produced: u64,
    pub regular_expired: u64,
    pub combat_expired: u64,
    pub priority_expired_min: u64,
    pub priority_expired_max: u64,
    #[serde(skip)]
    held_min: u16,
    #[serde(skip)]
    held_max: u16,
}

impl ManaUse {
    pub fn float(&mut self, amount: u16) {
        self.priority_produced += u64::from(amount);
        self.held_min += amount;
        self.held_max += amount;
    }

    /// A removal can consume at most the floated units held and must
    /// consume any that cannot fit in the remaining pool.
    pub fn remove(&mut self, amount: u16, remaining: u16, expiry: bool) {
        if expiry {
            self.regular_expired += u64::from(amount);
            self.priority_expired_min += u64::from(self.held_min.saturating_sub(remaining));
            self.priority_expired_max += u64::from(self.held_max.min(amount));
            self.priority_expired_max = self.priority_expired_max.min(self.priority_produced);
        }
        self.held_min = self.held_min.saturating_sub(amount);
        self.held_max = self.held_max.min(remaining);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardSnapshot {
    pub copy: usize,
    pub definition: u32,
    pub name: String,
    pub incarnation: u32,
    pub owner: usize,
    pub zone: Option<ZoneType>,
    pub origin: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CardMove {
    pub sequence: usize,
    #[serde(rename = "card")]
    pub object: CardSnapshot,
    pub from: Option<ZoneType>,
    pub controller: usize,
    pub cause: &'static str,
    pub turn: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct CardCapture {
    pub version: u32,
    pub content_digest: String,
    pub initial: Vec<CardSnapshot>,
    pub opening: [Vec<usize>; 2],
    pub moves: Vec<CardMove>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureCapture>,
}

fn snapshot(game: &Game, object: ObjectId) -> CardSnapshot {
    let (definition, data) = game.state.source_base(object);
    let (_, owner, _) = game
        .state
        .objects
        .provenance(object)
        .expect("captured provenance");
    let player = &game.state.players[owner.0];
    let original = game.capture_originals.get(&object.slot());
    CardSnapshot {
        copy: object.slot(),
        definition: original.map_or(definition.0, |object| object.definition),
        name: original.map_or_else(|| data.name.clone(), |object| object.name.clone()),
        incarnation: object.generation(),
        owner: owner.0,
        zone: game.state.zones.zone_of(
            game.state
                .card_id(object)
                .map_or(object, |card| game.state.object_of(card)),
        ),
        origin: match game.state.card_id(object) {
            Some(card) if player.deck.contains(&card) => "deck",
            Some(card) if player.sideboard.contains(&card) => "outside",
            _ => "generated",
        }
        .into(),
    }
}

pub(crate) fn snapshots(game: &Game) -> Vec<CardSnapshot> {
    game.state
        .zones
        .latest()
        .map(|object| snapshot(game, object))
        .collect()
}

impl CardCapture {
    pub(crate) fn new(game: &Game, initial: Vec<CardSnapshot>, mode: CaptureMode) -> Self {
        Self {
            version: 1,
            content_digest: default_content_pack().content_digest(),
            initial,
            opening: std::array::from_fn(|seat| {
                game.state
                    .zones
                    .zone_objects(ZoneType::Hand, PlayerId(seat))
                    .iter()
                    .map(|c| c.slot())
                    .collect()
            }),
            moves: Vec::new(),
            exposure: (mode == CaptureMode::Exposure).then(Default::default),
        }
    }
}

impl Game {
    pub(crate) fn exposure_mut(&mut self) -> Option<&mut ExposureCapture> {
        self.card_capture.as_mut()?.exposure.as_mut()
    }

    pub(crate) fn finish_combat_capture(&mut self) {
        // Combat attribution is read before turn progression expires any control.
        // Attribute losses to the same chooser as the declaration denominator.
        let attacking = self.decision_controller(self.active_player());
        let defending = self.decision_controller(self.non_active_player());
        if let Some(exposure) = self.exposure_mut() {
            exposure.finish_combat(attacking, defending);
        }
    }

    pub(crate) fn mana_capture(&mut self, player: PlayerId) -> Option<&mut ManaUse> {
        Some(&mut self.card_capture.as_mut()?.exposure.as_mut()?.mana[player.0])
    }

    /// Transactions preserve diagnostic sinks; search clones deliberately omit them.
    pub(crate) fn rollback_snapshot(&self) -> Self {
        let mut snapshot = self.clone();
        snapshot.decision_memory_enabled = self.decision_memory_enabled;
        snapshot.card_capture = self.card_capture.clone();
        snapshot.statistics = self.statistics.clone();
        snapshot.presentation = self.presentation.clone();
        snapshot.capture_originals = self.capture_originals.clone();
        snapshot.capture_drawing = self.capture_drawing;
        snapshot
    }

    /// Bind an atomic action to the same published choice as indexed stepping.
    pub(crate) fn exposure_for_action(&self, action: &Action) -> Option<ExposureChoice> {
        let space = self.current_action_space.as_ref()?;
        let index = space
            .actions
            .iter()
            .position(|candidate| candidate == action)?;
        self.exposure_choice(space, index)
    }

    pub(crate) fn exposure_choice(
        &self,
        space: &ActionSpace,
        selected: usize,
    ) -> Option<ExposureChoice> {
        if self.statistics.is_none()
            && self
                .card_capture
                .as_ref()
                .and_then(|c| c.exposure.as_ref())
                .is_none()
        {
            return None;
        }
        if space.actions.len() <= 1 {
            return None;
        }
        Some(ExposureChoice {
            revision: self.decision_epoch,
            actor: space.player.expect("published chooser").0,
            kind: space.kind,
            selected,
            options: space
                .actions
                .iter()
                .map(|action| {
                    let objects = Observation::action_focus(self, action)
                        .into_iter()
                        .filter_map(|id| {
                            self.state
                                .latest_objects()
                                .find(|(_, object)| object.id == id)
                                .map(|(object, _)| object)
                                .or_else(|| {
                                    self.state
                                        .permanents
                                        .iter()
                                        .flatten()
                                        .find(|p| p.id == id)
                                        .map(|p| p.object)
                                })
                        })
                        .map(|object_id| snapshot(self, object_id))
                        .collect();
                    ExposureOption {
                        action_type: action.action_type(),
                        objects,
                    }
                })
                .collect(),
        })
    }

    pub(crate) fn commit_capture_choice(&mut self, choice: ExposureChoice) {
        if let Some(stats) = &mut self.statistics {
            stats.choice(&choice);
        }
        if let Some(exposure) = self.card_capture.as_mut().and_then(|c| c.exposure.as_mut()) {
            exposure.choices.push(choice);
        }
    }

    pub(crate) fn capture_resolution(&mut self, frame: &EffectFrame) {
        // Attribute to the captured physical copy's initial definition. Copy/face
        // changes do not turn this into a count of executed mechanic programs.
        let definition = frame.source().and_then(|source| {
            self.capture_originals
                .get(&source.slot())
                .map(|card| card.definition)
        });
        if let Some(stats) = &mut self.statistics {
            stats.resolution(frame.controller.0, definition);
        }
        if let Some(exposure) = self
            .card_capture
            .as_mut()
            .and_then(|capture| capture.exposure.as_mut())
        {
            exposure.resolutions.push(ExposureResolution {
                revision: self.decision_epoch,
                controller: frame.controller.0,
                source: frame.source_ref(),
                definition,
            });
        }
    }

    pub(crate) fn capture_object_move(
        &mut self,
        object: ObjectId,
        from: Option<ZoneType>,
        to: Option<ZoneType>,
        controller: PlayerId,
    ) {
        if self.card_capture.is_none() && self.statistics.is_none() {
            return;
        }
        let data = snapshot(self, object);
        self.capture_originals
            .entry(data.copy)
            .or_insert_with(|| data.clone());
        let zones = &self.state.zones;

        // Only the actual committed movement caused by draw_cards consumes this
        // marker: `object` is what the marked library card became (CR 400.7).
        // Triggered movements during its processing cannot inherit it.
        let drawn = self
            .capture_drawing
            .and_then(|drawing| zones.successor(drawing))
            == Some(object)
            && from == Some(ZoneType::Library)
            && to == Some(ZoneType::Hand);
        if drawn {
            self.capture_drawing = None;
        }
        let cause = if drawn {
            "draw"
        } else if to == Some(ZoneType::Hand) {
            match from {
                Some(ZoneType::Library) => "selection",
                Some(_) => "return",
                None if data.origin == "outside" => "outside",
                None if data.origin == "generated" => "generated",
                None => "unknown",
            }
        } else {
            "move"
        };
        if let Some(stats) = &mut self.statistics {
            stats.movement(&data, drawn);
        }
        let Some(capture) = self.card_capture.as_mut() else {
            return;
        };
        capture.moves.push(CardMove {
            sequence: capture.moves.len(),
            object: data,
            from,
            controller: controller.0,
            cause,
            turn: self.state.turn.turn_number,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ability::Effect, game_object::Target, player::PlayerConfig};
    use std::collections::BTreeMap;

    fn configs_with_card(card: &str) -> Vec<PlayerConfig> {
        ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([(card.into(), 20)])))
            .collect()
    }

    fn copied_spell(name: &str, targets: Vec<Target>) -> (Game, ObjectId) {
        let (caster, copier) = (PlayerId(0), PlayerId(1));
        let mut game = Game::new_capturing(configs_with_card(name), 17, true);
        let card = game.state.zones.zone_objects(ZoneType::Hand, caster)[0];
        let indices = (0..targets.len()).collect();
        let spell = game.push_spell_to_stack(card, caster, targets, indices, Default::default());
        let Some(Target::StackSpell(copy)) =
            game.copy_stack_object(Target::StackSpell(spell), copier)
        else {
            panic!("expected a spell copy");
        };
        assert_ne!(copy.slot(), spell.slot());
        (game, copy)
    }

    fn copy_moves(game: &Game, copy: ObjectId) -> impl Iterator<Item = &CardMove> {
        game.card_capture
            .as_ref()
            .unwrap()
            .moves
            .iter()
            .filter(move |m| m.object.copy == copy.slot())
    }

    fn copy_history(game: &Game, copy: ObjectId) -> Vec<(Option<ZoneType>, Option<ZoneType>, u32)> {
        copy_moves(game, copy)
            .map(|m| (m.from, m.object.zone, m.object.incarnation))
            .collect()
    }

    #[test]
    fn capture_tracks_a_copied_spell_from_stack_entry_until_it_ceases() {
        use ZoneType::{Graveyard, Stack};

        let (mut game, copy) = copied_spell("Lightning Bolt", vec![Target::Player(PlayerId(0))]);
        game.resolve_top_of_stack();
        game.perform_state_based_actions();

        assert_eq!(
            copy_history(&game, copy),
            [
                (None, Some(Stack), 0),
                (Some(Stack), Some(Graveyard), 1),
                (Some(Graveyard), None, 1)
            ]
        );
    }

    #[test]
    fn capture_tracks_a_copied_permanent_spell_as_the_token_it_becomes() {
        use crate::state::game_object::ObjectKind;
        use ZoneType::{Battlefield, Graveyard, Stack};

        let (mut game, copy) = copied_spell("Gray Ogre", vec![]);
        game.resolve_top_of_stack();
        let token = game.state.zones.successor(copy).unwrap();
        assert_eq!(game.state[token].kind(), ObjectKind::Token);
        game.move_object(token, Graveyard);
        game.perform_state_based_actions();

        assert_eq!(
            copy_history(&game, copy),
            [
                (None, Some(Stack), 0),
                (Some(Stack), Some(Battlefield), 1),
                (Some(Battlefield), Some(Graveyard), 2),
                (Some(Graveyard), None, 2),
            ]
        );
    }

    #[test]
    fn capture_matches_the_final_snapshot_of_an_unresolved_spell_copy() {
        let (mut game, copy) = copied_spell("Gray Ogre", vec![]);
        game.state.players[0].life = 0;
        game.perform_state_based_actions();
        assert!(game.is_game_over());

        let mut moves = copy_moves(&game, copy);
        let entry = moves.next().expect("spell copy's stack entry");
        assert!(moves.next().is_none());
        assert_eq!(
            (entry.from, entry.object.zone, entry.object.incarnation),
            (None, Some(ZoneType::Stack), 0)
        );
        assert_eq!(entry.object.name, "Gray Ogre");
        assert_eq!(entry.object.origin, "generated");
        assert_eq!((entry.object.owner, entry.controller), (1, 1));
        assert_eq!(entry.cause, "move");
        assert!(snapshots(&game).contains(&entry.object));
    }

    #[test]
    fn copying_triggered_and_activated_abilities_records_no_card_movements() {
        use crate::state::{mana::Mana, stack_object::StackObject};

        let mut game = Game::new_capturing(configs_with_card("The Fire Nation Drill"), 17, true);
        let player = game.active_player();
        let copier = game.non_active_player();
        let card = game.state.zones.zone_objects(ZoneType::Hand, player)[0];
        let drill = game.move_object(card, ZoneType::Battlefield);
        assert!(game.flush_triggers().is_none());
        let permanent = game.state.permanent_of(drill).unwrap();
        *game.state.mana_pool_mut(player) = Mana::parse("R").into();
        game.activate_ability_action(player, permanent, 0).unwrap();

        let [StackObject::TriggeredAbility(trigger), StackObject::ActivatedAbility(activation)] =
            game.state.stack_objects.as_slice()
        else {
            panic!("expected both Drill abilities on the stack");
        };
        let moves_before = game.card_capture.as_ref().unwrap().moves.len();
        for id in [trigger.id, activation.id] {
            let original = Target::StackAbility(id);
            let copy = game.copy_stack_object(original, copier).unwrap();
            assert!(matches!(copy, Target::StackAbility(_)));
            assert_ne!(copy, original);
            assert_eq!(game.state.stack_objects.last().unwrap().as_target(), copy);
        }
        assert_eq!(
            game.card_capture.as_ref().unwrap().moves.len(),
            moves_before
        );
    }

    #[test]
    fn turn_budget_includes_modal_and_pooled_mana_and_counts_only_real_spends() {
        use crate::{
            flow::decision::SpendCompletion,
            state::mana::{Mana, ManaCost},
        };
        let payer = PlayerId(0);
        let mut players = configs_with_card("Mountain");
        players[0] = PlayerConfig::new(
            "a",
            BTreeMap::from([
                ("Izzet Guildgate".into(), 20),
                ("Prophetic Prism".into(), 1),
            ]),
        );
        let mut game = Game::construct(
            &players.into(),
            17,
            true,
            Some(CaptureMode::Exposure),
            false,
        );
        game.keep_opening_hands();
        game.scenario_force_battlefield(payer, "Izzet Guildgate", true)
            .unwrap();
        game.create_token("token.treasure", payer, false).unwrap();
        let prism = game
            .scenario_force_battlefield(payer, "Prophetic Prism", true)
            .unwrap();
        // This creature can convoke or waterbend but produces no mana.
        game.create_token("token.ally", payer, false).unwrap();
        game.state
            .combat_mana_pool_mut(payer)
            .add(&Mana::parse("R"), None);
        game.capture_priority_mana(payer);
        let hand = game.state.zones.zone_objects(ZoneType::Hand, payer);
        let card = game.state.card_id(hand[0]).unwrap();
        // Foretell's special-action payment here consumes the combat pool.
        game.begin_mana_spend(
            payer,
            ManaCost::parse("R"),
            SpendCompletion::Foretell(card),
            EffectFrame::new(payer, Vec::new()),
        )
        .unwrap();
        // A second source arriving on the same turn increases the peak.
        game.create_token("token.treasure", payer, false).unwrap();
        game.capture_priority_mana(payer);
        let turn = &game.exposure_mut().unwrap().mana_turns[0][0];
        assert_eq!((turn.available, turn.paid), (4, 1));

        // The filter turns a pooled unit into another color at no net gain.
        game.state.mana_pool_mut(payer).add(&Mana::parse("W"), None);
        game.activate_mana_ability(payer, prism, 0).unwrap();
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
        assert_eq!(game.exposure_mut().unwrap().mana_turns[0][0].paid, 1);
    }

    #[test]
    fn combat_counts_partners_and_committed_deaths_in_a_terminal_combat() {
        use crate::flow::turn::PhaseKind;
        let (active, defender) = (PlayerId(0), PlayerId(1));
        let mut game = Game::construct(
            &configs_with_card("Mountain").into(),
            17,
            true,
            Some(CaptureMode::Exposure),
            false,
        );
        game.keep_opening_hands();
        game.state.turn.current_phase = PhaseKind::Combat as usize;
        game.state.turn.current_step = 3; // combat damage
        game.state.turn.step_initialized = false;
        game.state.combat = Some(Default::default());
        let attackers: Vec<_> = (0..4)
            .map(|_| game.create_token("token.ally", active, false).unwrap())
            .collect();
        let blockers: Vec<_> = (0..4)
            .map(|_| game.create_token("token.ally", defender, false).unwrap())
            .collect();
        game.mark_permanents_not_summoning_sick(active);
        for &attacker in &attackers {
            game.declare_attacker(attacker, true).unwrap();
        }
        for (&blocker, attacker) in blockers.iter().zip([0, 0, 1, 2]) {
            game.declare_blocker(blocker, Some(attackers[attacker]))
                .unwrap();
        }
        // A trade with one of two blockers, a sacrificed blocker, and a removed
        // attacker. The unblocked fourth attacker deals lethal player damage.
        for id in [attackers[0], blockers[1], blockers[2], attackers[2]] {
            game.move_object(game.permanent_object(id).unwrap(), ZoneType::Graveyard);
        }
        // Ending the turn during combat also settles the ledger.
        let mut surviving = game.rollback_snapshot();
        surviving.run_frame(EffectFrame::new(active, vec![Effect::EndTheTurn]));
        let facts = surviving.exposure_mut().unwrap();
        assert_eq!(facts.attackers_lost_for_nothing, [1, 0]);
        assert_eq!(facts.blockers_lost_for_nothing, [0, 1]);

        // One more blocker receives lethal damage in the same batch as its
        // controller. Only earlier, committed deaths count on this branch.
        let extra = game.create_token("token.ally", active, false).unwrap();
        game.mark_permanents_not_summoning_sick(active);
        game.declare_attacker(extra, true).unwrap();
        let last = game.create_token("token.ally", defender, false).unwrap();
        game.declare_blocker(last, Some(extra)).unwrap();
        *game.state.life_mut(defender).unwrap() = 1;
        assert!(game.tick());
        assert!(game.state.permanents[last].is_some());
        let facts = game.exposure_mut().unwrap();
        assert_eq!(facts.attackers_lost_for_nothing, [1, 0]);
        assert_eq!(facts.blockers_lost_for_nothing, [0, 1]);
    }

    #[test]
    fn capture_preserves_rules_and_excludes_forks() {
        let plain = Game::new(configs_with_card("Mountain"), 17, true);
        let mut captured = Game::new_capturing(configs_with_card("Mountain"), 17, true);
        assert_eq!(
            plain.state.deterministic_hash(),
            captured.state.deterministic_hash()
        );
        let capture = captured.card_capture.as_ref().unwrap();
        assert_eq!(capture.opening[0].len(), 7);
        assert_eq!(capture.opening[1].len(), 7);
        assert!(captured.clone().card_capture.is_none());
        let mut fork = captured.clone();
        fork.draw_cards(PlayerId(0), 2);
        captured.draw_cards(PlayerId(0), 2);
        assert_eq!(
            fork.state.deterministic_hash(),
            captured.state.deterministic_hash()
        );
        assert_eq!(
            captured
                .card_capture
                .as_ref()
                .unwrap()
                .moves
                .iter()
                .filter(|m| m.cause == "draw")
                .count(),
            2
        );
    }

    #[test]
    fn selection_bounce_and_redraw_keep_distinct_causes() {
        let mut game = Game::new_capturing(configs_with_card("Mountain"), 18, true);
        let object = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        let object = game.move_object(object, ZoneType::Hand); // selection, not draw
        let object = game.move_object(object, ZoneType::Battlefield);
        let object = game.move_object(object, ZoneType::Hand); // return, not draw
        let object = game.move_object(object, ZoneType::Library);
        // ZoneManager appends to the library's top.
        game.draw_cards(PlayerId(0), 1);
        let moves = &game.card_capture.as_ref().unwrap().moves;
        assert_eq!(
            moves.iter().map(|m| m.cause).collect::<Vec<_>>(),
            ["selection", "move", "return", "move", "draw"]
        );
        assert_eq!(moves.last().unwrap().object.copy, object.slot());
        assert_eq!(moves.last().unwrap().object.incarnation, 5);
    }
    #[test]
    fn exposure_distinguishes_declined_countered_and_completed() {
        use crate::agent::action::{Action, ActionSpace, ActionSpaceKind};
        for outcome in ["declined", "countered", "completed"] {
            let deck = BTreeMap::from([("Mountain".into(), 20), ("Gray Ogre".into(), 4)]);
            let mut game = Game::new_capturing(
                vec![
                    PlayerConfig::new("a", deck.clone()),
                    PlayerConfig::new("b", deck),
                ],
                17,
                false,
            );
            game.keep_opening_hands();
            game.scenario_force_card_in_hand(PlayerId(0), "Gray Ogre")
                .unwrap();
            let object_id = *game
                .state
                .zones
                .zone_objects(ZoneType::Hand, PlayerId(0))
                .iter()
                .find(|object_id| game.state[**object_id].name == "Gray Ogre")
                .unwrap();
            // Synthetic free, untargeted spell isolates capture from card rules.
            game.state[object_id].mana_cost = None;
            game.state[object_id]
                .keywords
                .insert(crate::state::card::Keyword::Flash);
            let initial = snapshots(&game);
            game.statistics = Some(crate::flow::statistics::Statistics::new(&initial, true));
            game.card_capture = Some(CardCapture::new(&game, initial, CaptureMode::Exposure));
            game.publish_action_space(ActionSpace {
                player: Some(PlayerId(0)),
                kind: ActionSpaceKind::Priority,
                actions: vec![
                    Action::CastSpell {
                        profile: SpellProfile::Front,
                        player: PlayerId(0),
                        object: object_id,
                        route: None,
                    },
                    Action::PassPriority {
                        player: PlayerId(0),
                    },
                ],
                focus: Vec::new(),
            });
            let source = game.state.live(object_id).unwrap();
            game.step(usize::from(outcome == "declined")).unwrap();
            if outcome == "countered" {
                game.counter_spell(object_id, None, ZoneType::Graveyard);
            }
            if outcome == "completed" {
                game.resolve_top_of_stack();
            }
            let exposure = game
                .card_capture
                .as_ref()
                .unwrap()
                .exposure
                .as_ref()
                .unwrap();
            assert_eq!(exposure.choices.len(), 1);
            assert_eq!(exposure.choices[0].actor, 0);
            assert_eq!(
                exposure.choices[0].options[0].objects[0].copy,
                source.slot()
            );
            assert_eq!(
                exposure.choices[0].options[0].objects[0].incarnation,
                source.generation()
            );
            assert_eq!(
                exposure.choices[0].selected,
                usize::from(outcome == "declined")
            );
            assert_eq!(
                exposure.resolutions.len(),
                usize::from(outcome == "completed")
            );
            let rows = game.statistics.as_ref().unwrap().rows();
            assert_eq!(
                rows.iter().map(|r| r.resolutions.unwrap()).sum::<u64>(),
                u64::from(outcome == "completed")
            );
            assert_eq!(
                rows.iter().map(|r| r.selections.unwrap()).sum::<u64>(),
                u64::from(outcome != "declined")
            );

            if let Some(resolution) = exposure.resolutions.first() {
                // CR 400.7g: the spell is the object the hand card became.
                let spell = resolution.source.unwrap();
                assert_eq!(game.state.card_id(spell), game.state.card_id(source));
                assert_eq!(spell.generation(), source.generation() + 1);
                assert_eq!(resolution.controller, 0);
            }
        }
    }
    #[test]
    fn exposure_capture_keeps_physical_identity_across_face_changes() {
        use std::sync::Arc;
        let mut game = Game::new_capturing(configs_with_card("Mountain"), 17, true);
        let object = game.state.zones.zone_objects(ZoneType::Hand, PlayerId(0))[0];
        let before = snapshot(&game, object);
        // A controlled face switch uses two real definitions without asserting
        // that this synthetic double-faced land is a playable card.
        let content = default_content_pack();
        let back = content.definition_id("Forest").unwrap();
        let front = (
            game.state[object].definition_id,
            game.state[object].copiable_definition(),
        );
        game.state[object].show_transforming_face(crate::state::game_object::TransformingFaces {
            front,
            back: (back, Arc::clone(&content.definition(back).unwrap().printed)),
            back_up: true,
        });
        let after = snapshot(&game, object);
        assert_eq!(game.state[object].name, "Forest");
        assert_eq!(after.name, before.name);
        assert_eq!(after.definition, before.definition);
        assert_eq!(after.incarnation, before.incarnation);
        game.move_object(object, ZoneType::Battlefield);
        let moved = &game
            .card_capture
            .as_ref()
            .unwrap()
            .moves
            .last()
            .unwrap()
            .object;
        assert_eq!(moved.name, before.name);
        assert_eq!(moved.definition, before.definition);
        assert_eq!(moved.incarnation, before.incarnation + 1);
    }
}

#[cfg(test)]
mod mana_tests {
    use super::ManaUse;

    #[test]
    fn spending_from_a_mixed_pool_bounds_the_expired_float() {
        let mut mana = ManaUse::default();
        mana.float(2);
        // One other unit joins. Spend one, then let the two remaining units expire.
        mana.remove(1, 2, false);
        mana.remove(2, 0, true);
        assert_eq!(
            (mana.priority_expired_min, mana.priority_expired_max),
            (1, 2)
        );
        // No attribution carries over from an emptied pool.
        mana.remove(4, 0, true);
        assert_eq!(mana.regular_expired, 6);
        assert_eq!(
            (mana.priority_expired_min, mana.priority_expired_max),
            (1, 2)
        );
    }
}
