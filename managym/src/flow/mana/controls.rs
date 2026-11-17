//! Optional, revision-bound mana controls for a live human chooser.
//! An inverse earns an undo offer only by restoring the semantic state on a trial game.
use crate::{
    agent::{
        action::{ActionSpace, ActionSpaceKind},
        observation::Observation,
        structured_offer::{ObjectRenderId, SubjectRef},
    },
    decision::{Command, SemanticError, SemanticTransition},
    flow::{
        event::{EventLedger, GameEvent},
        game::Game,
    },
    state::{
        game_object::{PermanentId, PlayerId},
        mana::Mana,
    },
};
use serde::Serialize;
use std::collections::BTreeMap;

type Amounts = BTreeMap<&'static str, u8>;

fn describe_output(output: &Mana) -> (Amounts, String) {
    let amounts = output.unrestricted_colors();
    let restricted = output.restricted_colors();
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
        let count = amounts[i].saturating_add(restricted[i]);
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
        #[serde(skip)]
        symbols: String,
    },
    Undo {
        id: usize,
        label: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Activation {
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
struct FixedTap {
    source: PermanentId,
    player: PlayerId,
    pool: Mana,
    consecutive_passes: usize,
    cache: [Option<Mana>; 2],
    // Sealed history is immutable and shared; the owned tail is bounded.
    events: EventLedger,
}

impl FixedTap {
    fn restores(&self, game: &Game, before: &str) -> bool {
        let mut trial = game.clone();
        self.reverse(&mut trial);
        trial.semantic_state_digest() == before
    }

    fn reverse(&self, game: &mut Game) {
        game.untap_permanent(self.source);
        game.state.priority_mut().consecutive_passes = self.consecutive_passes;
        *game.state.mana_pool_mut(self.player) = self.pool.clone();
        *game.state.events_mut() = self.events.clone();
        *game.state.mana_cache_mut() = self.cache.clone();
        let actions = game.compute_player_actions(self.player);
        game.publish_action_space(ActionSpace {
            player: Some(self.player),
            kind: ActionSpaceKind::Priority,
            actions,
            focus: Vec::new(),
        });
    }
}

#[derive(Debug)]
struct Entry {
    activation: Activation,
    inverse: Result<FixedTap, Reason>,
    after: String,
}

#[derive(Debug, Default)]
pub(crate) struct ManaControls {
    revision: Option<u64>,
    entries: Vec<Entry>,
}

impl ManaControls {
    pub(crate) fn frame(&self, game: &Game, viewer: PlayerId) -> ControlFrame {
        let mut frame = ControlFrame {
            revision: game.decision_epoch,
            offers: Vec::new(),
            undo: None,
        };
        let Some(space) = game.action_space() else {
            return frame;
        };
        if space.player != Some(viewer)
            || space.kind != ActionSpaceKind::Priority
            || game.state.suspended_decision.is_some()
            || game.pending_choice.is_some()
        {
            return frame;
        }
        for source in game.payment_sources(game.state.priority.holder) {
            let Some(output) = source.fixed.filter(|_| source.untapped) else {
                continue;
            };
            let id = game.state.permanents[source.permanent].as_ref().unwrap().id;
            let (mana, symbols) = describe_output(&output);
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
                symbols,
            });
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
        let before = game.semantic_state_digest();
        let mut entry = None;
        let done = match offer {
            ControlOffer::Float {
                permanent,
                source,
                output,
                symbols,
                ..
            } => {
                let player = game.state.priority.holder;
                let name = game.state[game.permanent_object(permanent).unwrap()]
                    .name
                    .clone();
                let inverse = FixedTap {
                    source: permanent,
                    player,
                    pool: game.state.players[player.0].mana_pool.clone(),
                    consecutive_passes: game.state.priority.consecutive_passes,
                    cache: game.state.mana_cache.clone(),
                    events: game.state.events.clone(),
                };
                let activation = Activation {
                    sources: vec![source],
                    mana: output,
                    label: format!("{name} tapped for {symbols}"),
                };
                game.tap_fixed_mana_source(player, permanent);
                game.state.priority_mut().on_non_pass_action(player);
                let skip = std::mem::replace(&mut game.skip_trivial, false);
                let done = game.finish_action_step();
                game.skip_trivial = skip;
                // The digest covers rules state. A revealed card could be visible even
                // when it leaves no durable change, so admit only the tap increment.
                let just_tapped = |event: &GameEvent| matches!(event, GameEvent::PermanentTapped { permanent: source, for_mana: true } if *source == permanent);
                let end = game.state.events.len();
                let tap_only = end == checkpoint.state.events.len() + 1
                    && just_tapped(&game.state.events[end - 1])
                    && matches!(game.state.observation_events.as_slice(), [event] if just_tapped(event));
                entry = Some((activation, inverse, name, tap_only));
                done
            }
            ControlOffer::Undo { .. } => {
                let top = self.entries.last().expect("offered undo");
                if top.after != before {
                    return Err(SemanticError::IllegalCommand(
                        "activation state changed".into(),
                    ));
                }
                top.inverse.as_ref().expect("proved undo").reverse(game);
                false
            }
        };
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
        if let Some((activation, inverse, name, tap_only)) = entry {
            let triggered =
                game.state.trigger_enqueue_counter != checkpoint.state.trigger_enqueue_counter;
            let inverse = if triggered {
                Err(Reason {
                    code: "triggered_ability",
                    explanation: format!("{name}'s tap triggered an ability."),
                })
            } else if tap_only && inverse.restores(game, &before) {
                Ok(inverse)
            } else {
                Err(Reason {
                    code: "other_consequences",
                    explanation: "This activation had other consequences.".into(),
                })
            };
            self.entries.push(Entry {
                activation,
                inverse,
                after: game.semantic_state_digest(),
            });
        } else {
            self.entries.pop();
        }
        self.revision = Some(game.decision_epoch);
        Ok((transition, observation))
    }
}
