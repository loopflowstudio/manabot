//! Optional, revision-bound mana controls for a live human chooser.
//! An inverse earns an undo offer only by restoring the semantic state on a trial game.
use crate::{
    agent::{
        action::{Action, ActionSpace, ActionSpaceKind},
        observation::Observation,
        structured_offer::{AtomicCommand, ObjectRenderId, PublicCommitment, SubjectRef},
    },
    decision::{Command, SemanticError, SemanticTransition},
    flow::{
        decision::{Decision, ManaProduction, SpendCompletion},
        event::{EventLedger, GameEvent},
        game::{Game, PendingChoice},
        turn::StepKind,
    },
    state::{
        game_object::{PermanentId, PlayerId},
        mana::Mana,
    },
};
use serde::Serialize;
use std::collections::BTreeMap;

type Amounts = BTreeMap<&'static str, u8>;

fn color_totals(output: &Mana) -> [u8; 6] {
    let amounts = output.unrestricted_colors();
    let restricted = output.restricted_colors();
    std::array::from_fn(|i| amounts[i].saturating_add(restricted[i]))
}

fn describe_output(amounts: [u8; 6]) -> (Amounts, String) {
    let mut symbols = String::new();
    let mut mana = Amounts::new();
    for (i, (name, symbol)) in [
        ("white", "W"),
        ("blue", "U"),
        ("black", "B"),
        ("red", "R"),
        ("green", "G"),
        ("colorless", "C"),
    ]
    .into_iter()
    .enumerate()
    {
        let count = amounts[i];
        if count > 0 {
            mana.insert(name, count);
            symbols.push_str(&format!("{{{symbol}}}").repeat(usize::from(count)));
        }
    }
    (mana, symbols)
}

#[derive(Debug, Serialize)]
pub struct ControlFrame {
    pub revision: u64,
    pub offers: Vec<ControlOffer>,
    pub undo: Option<UndoStatus>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "verb", rename_all = "snake_case")]
pub enum ControlOffer {
    Float {
        id: usize,
        source: SubjectRef,
        label: String,
        output: Amounts,
        #[serde(skip)]
        permanent: PermanentId,
    },
    Undo {
        id: usize,
        label: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Activation {
    command_id: String,
    label: String,
    sources: Vec<SubjectRef>,
    mana: Amounts,
}

#[derive(Clone, Debug, Serialize)]
pub struct Reason {
    code: &'static str,
    explanation: String,
}

#[derive(Debug, Serialize)]
pub struct UndoStatus {
    activation: Activation,
    offer: Option<usize>,
    reason: Option<Reason>,
}

#[derive(Debug)]
struct ManaInverse {
    player: PlayerId,
    pool: Mana,
    combat_pool: Mana,
    consecutive_passes: usize,
    cache: [Option<Mana>; 2],
    events: EventLedger,
    pending_events: usize,
    skip_count: usize,
    taps: Vec<PermanentId>,
    // The activation was legal, so neither use limit was spent beforehand.
    used: Option<(PermanentId, usize)>,
}

impl ManaInverse {
    fn reverse(&self, game: &mut Game) {
        for &permanent in &self.taps {
            game.untap_permanent(permanent);
        }
        if let Some((source, index)) = self.used {
            let permanent = game.state.permanents[source].as_mut().unwrap();
            permanent.exhausted.retain(|i| *i != index);
            permanent.activated_this_turn.retain(|i| *i != index);
        }
        game.state.priority_mut().consecutive_passes = self.consecutive_passes;
        let (pool, combat) = game.state.mana_pools_mut(self.player);
        *pool = self.pool.clone();
        *combat = self.combat_pool.clone();
        *game.state.events_mut() = self.events.clone();
        game.state
            .pending_events_mut()
            .truncate(self.pending_events);
        *game.state.mana_cache_mut() = self.cache.clone();
        game.skip_trivial_count = self.skip_count;
        let space = game
            .pending_choice_action_space()
            .unwrap_or_else(|| ActionSpace {
                player: Some(self.player),
                kind: ActionSpaceKind::Priority,
                actions: game.compute_player_actions(self.player),
                focus: Vec::new(),
            });
        game.publish_action_space(space);
    }
}

#[derive(Debug)]
struct Entry {
    activation: Activation,
    inverse: Result<ManaInverse, Reason>,
    after: String,
}

/// Retained through the upfront and output choices of one activation. Only
/// the tap events and use limits can be reversed, even if more state changed.
#[derive(Debug)]
struct ManaActivation {
    command_id: String,
    name: String,
    source: SubjectRef,
    source_permanent: PermanentId,
    pool_before_output: [u8; 6],
    observation_taps_only: bool,
    before: String,
    inverse: ManaInverse,
    refusal: Option<Reason>,
    trigger_count: u64,
    payment: bool,
    actor: PlayerId,
    turn: u32,
    step: StepKind,
}

fn payment_pending(game: &Game) -> bool {
    matches!(game.pending_choice, Some(PendingChoice::ManaPayment { .. }))
}

fn activation_decision(game: &Game) -> bool {
    game.state
        .suspended_decision
        .as_ref()
        .is_some_and(|s| match &s.decision {
            Decision::ManaChoice { .. } | Decision::ManaUpfront { .. } => true,
            Decision::ManaSpend { completion, .. } => {
                matches!(completion.as_ref(), SpendCompletion::Upfront(_))
            }
            _ => false,
        })
}

impl Reason {
    fn new(code: &'static str, explanation: impl Into<String>) -> Self {
        Self {
            code,
            explanation: explanation.into(),
        }
    }
}

impl ManaActivation {
    fn start(game: &Game, action: &Action, command_id: &str) -> Option<Self> {
        if game.state.suspended_decision.is_some()
            || (game.pending_choice.is_some() && !payment_pending(game))
        {
            return None;
        }
        let (player, source, index) = match *action {
            Action::TapForMana { player, permanent } => (player, permanent, None),
            Action::ActivateAbility {
                player,
                permanent,
                ability_index,
            } => (player, permanent, Some(ability_index)),
            _ => return None,
        };
        let perm = game.state.permanents[source].as_ref()?;
        let name = game.state[perm.object].name.clone();
        let view = game.characteristics(source);
        let mut used = None;
        let (mut refusal, taps_source) = if let Some(index) = index {
            let ability = &view.activated_abilities[index];
            ability.mana_choices()?;
            let reason = if ability.sacrifice_source {
                Some(Reason::new(
                    "sacrificed_source",
                    format!("{name} was sacrificed for this mana."),
                ))
            } else if ability.life_cost > 0 {
                Some(Reason::new("paid_life", "Life was paid for this mana."))
            } else if ability.discard_card || ability.discard_source {
                Some(Reason::new(
                    "discarded_card",
                    "A card was discarded for this mana.",
                ))
            } else if !ability.mana_suffix().is_empty() {
                Some(Reason::new(
                    "additional_instructions",
                    "This ability did more than add mana.",
                ))
            } else {
                None
            };
            if ability.exhaust || ability.once_each_turn {
                used = Some((source, index));
            }
            (reason, ability.tap_source)
        } else {
            (None, true)
        };
        if taps_source
            && refusal.is_none()
            && !game
                .tap_bonuses(&game.mana_triggers(), player, source)
                .is_empty()
        {
            refusal = Some(Reason::new(
                "triggered_ability",
                format!("{name}'s tap triggered an ability."),
            ));
        }
        Some(Self {
            command_id: command_id.into(),
            name,
            source: SubjectRef::Object {
                id: ObjectRenderId {
                    entity: perm.id.0,
                    incarnation: perm.object.generation(),
                },
            },
            source_permanent: source,
            pool_before_output: color_totals(&game.state.players[player.0].mana_pool),
            observation_taps_only: true,
            before: game.semantic_state_digest(),
            trigger_count: game.state.trigger_enqueue_counter,
            payment: payment_pending(game),
            actor: game.action_space()?.player?,
            turn: game.state.turn.turn_number,
            step: game.state.turn.current_step_kind(),
            refusal,
            inverse: ManaInverse {
                player,
                pool: game.state.players[player.0].mana_pool.clone(),
                combat_pool: game.state.players[player.0].combat_mana_pool.clone(),
                consecutive_passes: game.state.priority.consecutive_passes,
                cache: game.state.mana_cache.clone(),
                events: game.state.events.clone(),
                pending_events: game.state.pending_events.len(),
                skip_count: game.skip_trivial_count,
                taps: Vec::new(),
                used,
            },
        })
    }

    fn same_decision(&self, game: &Game) -> bool {
        let Some(space) = game.action_space() else {
            return false;
        };
        game.state.suspended_decision.is_none()
            && space.player == Some(self.actor)
            && game.state.turn.turn_number == self.turn
            && game.state.turn.current_step_kind() == self.step
            && if self.payment {
                payment_pending(game)
            } else {
                space.kind == ActionSpaceKind::Priority && game.pending_choice.is_none()
            }
    }

    fn observe_choice(&mut self, game: &Game, action: &Action) {
        // The upfront fee has now been spent. Measure the outer production
        // including automatic triggered bonuses, excluding nested fee payments.
        if matches!(action, Action::ChooseMode { .. })
            && game.state.suspended_decision.as_ref().is_some_and(|s| {
                matches!(
                    s.decision,
                    Decision::ManaChoice {
                        production: ManaProduction::Output { .. },
                        resume: None,
                        ..
                    }
                )
            })
        {
            self.pool_before_output =
                color_totals(&game.state.players[self.inverse.player.0].mana_pool);
        }
    }

    fn observe_events(&mut self, game: &Game) {
        self.observation_taps_only &= game
            .state
            .observation_events
            .iter()
            .all(|e| matches!(e, GameEvent::PermanentTapped { .. }));
    }

    fn finish(mut self, game: &Game) -> Entry {
        let mut taps_only = self.observation_taps_only;
        for index in self.inverse.events.len()..game.state.events.len() {
            match game.state.events[index] {
                GameEvent::PermanentTapped { permanent, .. } => self.inverse.taps.push(permanent),
                _ => taps_only = false,
            }
        }
        let after = color_totals(&game.state.players[self.inverse.player.0].mana_pool);
        let (mana, symbols) = describe_output(std::array::from_fn(|i| {
            after[i].saturating_sub(self.pool_before_output[i])
        }));
        let verb = if self.inverse.taps.contains(&self.source_permanent) {
            "tapped"
        } else {
            "activated"
        };
        let label = format!("{} {verb} for {symbols}", self.name);
        let mut sources = vec![self.source];
        for &id in &self.inverse.taps {
            if id != self.source_permanent {
                if let Some(p) = &game.state.permanents[id] {
                    sources.push(SubjectRef::Object {
                        id: ObjectRenderId {
                            entity: p.id.0,
                            incarnation: p.object.generation(),
                        },
                    });
                }
            }
        }
        let inverse = if let Some(reason) = self.refusal {
            Err(reason)
        } else {
            // Payment holds tap events until priority. Detect triggers before
            // removing those events, including ones not yet put on the stack.
            let queued_trigger = !game.state.pending_events.is_empty() && {
                let mut trial = game.clone();
                let prefix = self
                    .inverse
                    .pending_events
                    .min(trial.state.pending_events.len());
                trial.state.pending_events_mut().drain(..prefix);
                trial.process_game_events();
                trial.state.trigger_enqueue_counter != self.trigger_count
            };
            if queued_trigger || game.state.trigger_enqueue_counter != self.trigger_count {
                Err(Reason::new(
                    "triggered_ability",
                    format!("{}'s tap triggered an ability.", self.name),
                ))
            } else {
                let mut trial = game.clone();
                if taps_only {
                    self.inverse.reverse(&mut trial);
                }
                if taps_only && trial.semantic_state_digest() == self.before {
                    Ok(self.inverse)
                } else {
                    Err(Reason::new(
                        "other_consequences",
                        "This activation had other consequences.",
                    ))
                }
            }
        };
        Entry {
            activation: Activation {
                command_id: self.command_id,
                label,
                sources,
                mana,
            },
            inverse,
            after: game.semantic_state_digest(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ManaControls {
    pub(crate) viewer: Option<PlayerId>,
    revision: Option<u64>,
    entries: Vec<Entry>,
    pending: Option<ManaActivation>,
}

impl ManaControls {
    pub(crate) fn reset(&mut self) {
        *self = Self {
            viewer: self.viewer,
            ..Self::default()
        };
    }

    pub(crate) fn permits(&self, actor: PlayerId) -> bool {
        self.viewer.is_none_or(|viewer| viewer == actor)
    }

    pub(crate) fn semantic_command(
        &mut self,
        game: &mut Game,
        command: &Command,
        actor: PlayerId,
        commitment: Option<PublicCommitment>,
        atomic: &AtomicCommand,
    ) -> Result<(SemanticTransition, Observation, bool), SemanticError> {
        let AtomicCommand::SearchAction { action, .. } = atomic else {
            unreachable!("action-aligned semantic Command")
        };
        if self.revision != Some(game.decision_epoch) {
            self.reset();
        }
        let mut activation = self
            .pending
            .take()
            .or_else(|| ManaActivation::start(game, action, &command.command_id));
        let skip = game.skip_trivial;
        if let Some(active) = &mut activation {
            active.observe_choice(game, action);
            // Keep the completed activation's undo decision even when passing
            // would otherwise be forced. Internal forced choices still commit.
            game.skip_trivial = false;
        }
        let result = game.apply_atomic_command(atomic);
        game.skip_trivial = skip;
        let mut done = result.map_err(|error| SemanticError::IllegalCommand(error.to_string()))?;
        if let Some(active) = &mut activation {
            while skip && !done && game.action_space().is_some_and(|s| s.actions.len() == 1) {
                if !activation_decision(game) {
                    if !active.payment && active.same_decision(game) {
                        break;
                    }
                    done = game.tick();
                    break;
                }
                active.observe_choice(game, &game.action_space().unwrap().actions[0]);
                game.skip_trivial_count += 1;
                game.skip_trivial = false;
                let next = game.step(0);
                game.skip_trivial = skip;
                done = next.map_err(|e| SemanticError::IllegalCommand(e.to_string()))?;
            }
            active.observe_events(game);
        }
        let transition = game.command_transition(command, actor, commitment, done)?;
        match activation {
            Some(active) if activation_decision(game) && !done => self.pending = Some(active),
            Some(active) if active.same_decision(game) => self.entries.push(active.finish(game)),
            _ => self.entries.clear(),
        }
        self.revision = Some(game.decision_epoch);
        Ok(transition)
    }

    pub(crate) fn frame(&self, game: &Game, viewer: PlayerId) -> ControlFrame {
        let mut frame = ControlFrame {
            revision: game.decision_epoch,
            offers: Vec::new(),
            undo: None,
        };
        let Some(space) = game.action_space() else {
            return frame;
        };
        if !self.permits(viewer)
            || space.player != Some(viewer)
            || (self.pending.is_some() && self.revision == Some(game.decision_epoch))
        {
            return frame;
        }
        if space.kind == ActionSpaceKind::Priority
            && game.state.suspended_decision.is_none()
            && game.pending_choice.is_none()
        {
            for source in game.payment_sources(game.state.priority.holder) {
                let Some(output) = source.fixed.filter(|_| source.untapped) else {
                    continue;
                };
                let id = game.state.permanents[source.permanent].as_ref().unwrap().id;
                let (mana, symbols) = describe_output(color_totals(&output));
                frame.offers.push(ControlOffer::Float {
                    id: space.actions.len() + frame.offers.len(),
                    permanent: source.permanent,
                    source: SubjectRef::Object {
                        id: ObjectRenderId {
                            entity: id.0,
                            incarnation: source.object.generation(),
                        },
                    },
                    label: format!("Tap for mana: {symbols}"),
                    output: mana,
                });
            }
        }
        if let Some(top) = self
            .entries
            .last()
            .filter(|_| self.revision == Some(game.decision_epoch))
        {
            let offer = top.inverse.as_ref().ok().map(|_| {
                let id = space.actions.len() + frame.offers.len();
                frame.offers.push(ControlOffer::Undo {
                    id,
                    label: format!("Undo: {}", top.activation.label),
                });
                id
            });
            frame.undo = Some(UndoStatus {
                activation: top.activation.clone(),
                offer,
                reason: top.inverse.as_ref().err().cloned(),
            });
        }
        frame
    }

    pub(crate) fn execute(
        &mut self,
        game: &mut Game,
        command: &Command,
    ) -> Result<(SemanticTransition, Observation), SemanticError> {
        if game.is_game_over() {
            return Err(SemanticError::GameOver);
        }
        if game.decision_epoch != command.expected_revision {
            return Err(SemanticError::StaleRevision {
                expected: command.expected_revision,
                current: game.decision_epoch,
            });
        }
        let space = game.action_space().ok_or(SemanticError::NoActiveDecision)?;
        let actor = space.player.ok_or(SemanticError::NoActiveDecision)?;
        let index = (command.offer_id as usize)
            .checked_sub(space.actions.len())
            .ok_or(SemanticError::UnknownOffer(command.offer_id))?;
        let offer = self
            .frame(game, actor)
            .offers
            .into_iter()
            .nth(index)
            .ok_or(SemanticError::UnknownOffer(command.offer_id))?;
        if !command.answers.is_empty() {
            return Err(SemanticError::IllegalCommand(
                "mana controls accept no answers".into(),
            ));
        }

        game.validate_object_preconditions(command)?;
        let checkpoint = game.rollback_snapshot();
        let mut entry = None;
        let done = match offer {
            ControlOffer::Float { permanent, .. } => {
                let player = game.state.priority.holder;
                entry = Some(
                    ManaActivation::start(
                        game,
                        &Action::TapForMana { player, permanent },
                        &command.command_id,
                    )
                    .expect("offered fixed source"),
                );
                game.tap_fixed_mana_source(player, permanent);
                game.state.priority_mut().on_non_pass_action(player);
                let skip = std::mem::replace(&mut game.skip_trivial, false);
                let done = game.finish_action_step();
                game.skip_trivial = skip;
                done
            }
            ControlOffer::Undo { .. } => {
                let top = self.entries.last().expect("offered undo");
                if top.after != game.semantic_state_digest() {
                    return Err(SemanticError::IllegalCommand(
                        "activation state changed".into(),
                    ));
                }
                top.inverse.as_ref().expect("proved undo").reverse(game);
                false
            }
        };
        if let Some(active) = &mut entry {
            active.observe_events(game);
        }
        let (transition, observation, _) = match game.command_transition(command, actor, None, done)
        {
            Ok(result) => result,
            Err(error) => {
                *game = checkpoint;
                return Err(error);
            }
        };
        if self.revision != Some(command.expected_revision) {
            self.entries.clear();
        }
        self.pending = None;
        match entry {
            Some(active) if active.same_decision(game) => self.entries.push(active.finish(game)),
            Some(_) => self.entries.clear(),
            None => {
                self.entries.pop();
            }
        }
        self.revision = Some(game.decision_epoch);
        Ok((transition, observation))
    }
}
