//! Academy's presentation sink: the committed facts of one native step, in
//! order, each with the earlier fact it follows from and the rule that did
//! it when no player chose it.
//!
//! The sink is optional, private and unhashed, like `statistics`: search
//! clones omit it, rollback snapshots keep it, and nothing a policy reads
//! comes from it. It mirrors every [`GameEvent`] `emit` commits and adds the
//! notes no event carries (an activation reaching the stack, a stack object
//! beginning to resolve). [`Game::presentation_facts`] projects the facts
//! through the observation's own event filter, so a viewer is told exactly
//! what `recent_events` would tell it, plus the notes, causes and rules.
//!
//! Causes: while a stack object resolves, and through the stabilization after
//! it, every fact follows from that resolution; a trigger follows from the
//! event that triggered it; anything else is a root. The resolution cause
//! ends when stabilization reaches priority (`sba_done`).

use serde::Serialize;

use crate::{
    agent::observation::Observation,
    flow::{
        event::{GameEvent, ObjectEventRef},
        game::Game,
        trigger::TriggerId,
    },
    state::{
        game_object::{ObjectId, PlayerId, PublicId},
        stack_object::StackObject,
        zone::ZoneType,
    },
};

/// A fact's place in the game's told order; never reused, so a cause can
/// name a fact taken in an earlier step.
pub type FactId = u32;

/// A state-based action (CR 704.5) that moved a permanent, in words a
/// person reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// CR 704.5g.
    LethalDamage,
    /// CR 704.5f.
    ZeroToughness,
    /// CR 704.5i.
    ZeroLoyalty,
    /// CR 704.5v.
    ZeroDefense,
    /// CR 704.5j.
    LegendRule,
    /// CR 704.5m.
    UnattachedAura,
}

#[derive(Clone, Debug)]
enum Note {
    Event(GameEvent),
    /// An activated ability was put on the stack (CR 602.2).
    Activated {
        source: ObjectId,
        controller: PlayerId,
        ability: PublicId,
    },
    /// The top stack object began to resolve (CR 608.2) with legal targets.
    Resolving {
        stack: PublicId,
        source: ObjectId,
        controller: PlayerId,
        spell: bool,
    },
}

#[derive(Clone, Debug)]
struct Fact {
    id: FactId,
    cause: Option<FactId>,
    rule: Option<Rule>,
    note: Note,
}

#[derive(Clone, Debug, Default)]
pub struct PresentationSink {
    facts: Vec<Fact>,
    next_id: FactId,
    /// The resolution every fact follows from until stabilization ends.
    resolution: Option<FactId>,
    /// The cause of the next fact, set by a trigger reaching the stack.
    next_cause: Option<FactId>,
    /// Triggers waiting to be put on the stack, by the event that made them.
    trigger_causes: Vec<(TriggerId, FactId)>,
    /// State-based reasons for the batch being performed.
    rules: Vec<(ObjectEventRef, Rule)>,
}

impl PresentationSink {
    fn push(&mut self, rule: Option<Rule>, note: Note) -> FactId {
        let id = self.next_id;
        self.next_id += 1;
        let cause = self.next_cause.take().or(self.resolution);
        self.facts.push(Fact {
            id,
            cause,
            rule,
            note,
        });
        id
    }

    fn event_rule(&self, event: &GameEvent) -> Option<Rule> {
        let GameEvent::PermanentsDied { objects } = event else {
            return None;
        };
        objects.iter().find_map(|object| {
            self.rules
                .iter()
                .find(|(died, _)| died == object)
                .map(|(_, rule)| *rule)
        })
    }
}

/// One viewer-safe fact: the observation's own event rows for a mirrored
/// event, or a note no event carries.
#[derive(Clone, Debug, Serialize)]
pub struct PresentationFact {
    pub id: FactId,
    pub cause: Option<FactId>,
    pub rule: Option<Rule>,
    #[serde(serialize_with = "serialize_rows")]
    pub events: Vec<crate::agent::observation::EventData>,
    pub note: Option<PresentationNote>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PresentationNote {
    Activated {
        source: Option<ObjectEventRef>,
        controller: usize,
        stack: u32,
    },
    Triggered {
        source: Option<ObjectEventRef>,
        controller: usize,
        stack: u32,
    },
    Resolving {
        stack: u32,
        source: Option<ObjectEventRef>,
        controller: usize,
        spell: bool,
    },
    Counters {
        object: ObjectEventRef,
        counter: crate::state::permanent::CounterKind,
        count: i32,
        total: i32,
    },
}

fn serialize_rows<S: serde::Serializer>(
    rows: &[crate::agent::observation::EventData],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    crate::agent::observation_json::EventsJson(rows).serialize(serializer)
}

impl Game {
    /// Start telling this game's facts; Academy alone turns it on.
    pub fn enable_presentation(&mut self) {
        self.presentation = Some(Box::default());
    }

    pub(crate) fn presentation_event(&mut self, event: &GameEvent) {
        if let Some(sink) = self.presentation.as_deref_mut() {
            let rule = sink.event_rule(event);
            sink.push(rule, Note::Event(event.clone()));
        }
    }

    pub(crate) fn presentation_activated(&mut self, ability: &StackObject) {
        let Some(sink) = self.presentation.as_deref_mut() else {
            return;
        };
        if let StackObject::ActivatedAbility(ability) = ability {
            sink.push(
                None,
                Note::Activated {
                    source: ability.source_object,
                    controller: ability.controller,
                    ability: ability.id,
                },
            );
        }
    }

    /// The top of the stack begins resolving; facts until priority follow
    /// from it.
    pub(crate) fn presentation_resolving(&mut self, stacked: &StackObject) {
        let Some(sink) = self.presentation.as_deref_mut() else {
            return;
        };
        sink.resolution = None;
        let id = sink.push(
            None,
            Note::Resolving {
                stack: stacked.id(),
                source: stacked.source_object(),
                controller: stacked.controller(),
                spell: matches!(stacked, StackObject::Spell(_)),
            },
        );
        sink.resolution = Some(id);
    }

    /// Stabilization reached priority: the resolution's consequences end.
    pub(crate) fn presentation_settled(&mut self) {
        if let Some(sink) = self.presentation.as_deref_mut() {
            sink.resolution = None;
            sink.rules.clear();
        }
    }

    pub(crate) fn presentation_rule(&mut self, object: ObjectId, rule: Rule) {
        if self.presentation.is_none() {
            return;
        }
        let Some(reference) = self.object_event_ref(object) else {
            return;
        };
        if let Some(sink) = self.presentation.as_deref_mut() {
            sink.rules.push((reference, rule));
        }
    }

    /// Bind the triggers `event` added from index `from` of the pending list
    /// to the fact that told `event`. A permanent leaving the battlefield for
    /// a graveyard is told as the death that names it ("dies", CR 700.4).
    pub(crate) fn presentation_triggers(&mut self, event: &GameEvent, from: usize) {
        if self.presentation.is_none() || self.state.pending_triggers.len() <= from {
            return;
        }
        let died = match event {
            GameEvent::CardMoved {
                departed: Some(departed),
                from: Some(ZoneType::Battlefield),
                to: ZoneType::Graveyard,
                ..
            } => self.object_event_ref(*departed),
            _ => None,
        };
        let Some(sink) = self.presentation.as_deref_mut() else {
            return;
        };
        let death = |fact: &&Fact| {
            matches!(&fact.note, Note::Event(GameEvent::PermanentsDied { objects })
                if objects.iter().any(|object| Some(*object) == died))
        };
        let same = |fact: &&Fact| matches!(&fact.note, Note::Event(told) if told == event);
        let found = match died {
            Some(_) => sink.facts.iter().rev().find(death),
            None => None,
        };
        let Some(fact) = found
            .or_else(|| sink.facts.iter().rev().find(same))
            .map(|fact| fact.id)
        else {
            return;
        };
        for trigger in self.state.pending_triggers.iter().skip(from) {
            sink.trigger_causes.push((trigger.id, fact));
        }
    }

    /// The next fact is a trigger `id` reaching the stack.
    pub(crate) fn presentation_placing(&mut self, id: TriggerId) {
        if let Some(sink) = self.presentation.as_deref_mut() {
            if let Some(index) = sink.trigger_causes.iter().position(|(t, _)| *t == id) {
                sink.next_cause = Some(sink.trigger_causes.swap_remove(index).1);
            }
        }
    }

    /// The facts since the last take as any viewer may read them, then
    /// cleared. A mirrored event the observation does not tell keeps its id
    /// with no rows, so causes through it still resolve.
    pub fn take_presentation_facts(&mut self) -> Vec<PresentationFact> {
        let Some(sink) = self.presentation.as_deref_mut() else {
            return Vec::new();
        };
        let facts = std::mem::take(&mut sink.facts);
        facts
            .into_iter()
            .map(|fact| {
                let (events, note) = match &fact.note {
                    Note::Event(event) => {
                        (Observation::event_data(self, event), self.event_note(event))
                    }
                    Note::Activated {
                        source,
                        controller,
                        ability,
                    } => (
                        Vec::new(),
                        Some(PresentationNote::Activated {
                            source: self.object_event_ref(*source),
                            controller: controller.0,
                            stack: ability.0,
                        }),
                    ),
                    Note::Resolving {
                        stack,
                        source,
                        controller,
                        spell,
                    } => (
                        // A spell's resolution reads as the observation
                        // names a resolved spell, before its effects.
                        if *spell {
                            Observation::event_data(
                                self,
                                &GameEvent::SpellResolved { object_id: *source },
                            )
                        } else {
                            Vec::new()
                        },
                        Some(PresentationNote::Resolving {
                            stack: stack.0,
                            source: self.object_event_ref(*source),
                            controller: controller.0,
                            spell: *spell,
                        }),
                    ),
                };
                PresentationFact {
                    id: fact.id,
                    cause: fact.cause,
                    rule: fact.rule,
                    events,
                    note,
                }
            })
            .collect()
    }

    fn event_note(&self, event: &GameEvent) -> Option<PresentationNote> {
        match event {
            GameEvent::AbilityTriggered {
                source_object,
                controller,
                ability,
                ..
            } => Some(PresentationNote::Triggered {
                source: self.object_event_ref(*source_object),
                controller: controller.0,
                stack: ability.0,
            }),
            GameEvent::CountersPut {
                permanent,
                kind,
                count,
                total,
                ..
            } => Some(PresentationNote::Counters {
                object: self.permanent_event_ref(*permanent)?,
                counter: *kind,
                count: *count,
                total: *total,
            }),
            _ => None,
        }
    }
}
