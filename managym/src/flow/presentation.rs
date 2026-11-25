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
//! it, every fact follows from that resolution; outside a resolution, a death
//! to lethal damage follows from the last damage dealt to that permanent
//! (combat damage, say); a trigger follows from the event that triggered it;
//! anything else is a root. The resolution cause and the damage record end
//! when stabilization reaches priority (`sba_done`).
//!
//! Positions: the sink keeps a copy of the game (a *position*) each time the
//! game reaches a whole state after at least one new fact: an action finished
//! (and before the next one starts), a stack object finished resolving,
//! stabilization began (after combat damage or a turn-based action), a
//! state-based batch was performed, a round of triggers reached the stack.
//! Each fact names the position cut after it, so a person can see the board
//! as it was after each told line; facts after the last cut are on the board
//! the step ends with. A position has no presentation sink and no published
//! decision, so it shows a board and never a stale choice. A priority pass
//! notes no fact and so never cuts one.

use serde::Serialize;

use crate::{
    experience::{PresentationKind, PresentationLabels},
    flow::{
        event::{DamageTarget, GameEvent, ObjectEventRef},
        game::Game,
        trigger::TriggerId,
    },
    state::{
        game_object::{ObjectId, PlayerId},
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
    DrawStep,
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
    /// Combat damage (CR 510.2): the rules' `DamageDealt` and the
    /// `CombatDamageDealt` the observation tells it by, one fact.
    CombatDamage {
        dealt: GameEvent,
        told: GameEvent,
    },
    Stack,
}

#[derive(Clone, Debug)]
struct Fact {
    id: FactId,
    cause: Option<FactId>,
    rule: Option<Rule>,
    note: Note,
    lines: [Vec<PresentationLine>; 2],
    /// Index of the position cut after this fact, once one is.
    position: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct PresentationSink {
    facts: Vec<Fact>,
    pub(super) land_play: Option<ObjectId>,
    next_id: FactId,
    /// The resolution every fact follows from until stabilization ends.
    resolution: Option<FactId>,
    /// The cause of the next fact, set by a trigger reaching the stack.
    next_cause: Option<FactId>,
    /// Triggers waiting to be put on the stack, by the event that made them.
    trigger_causes: Vec<(TriggerId, FactId)>,
    /// State-based reasons for the batch being performed.
    rules: Vec<(ObjectEventRef, Rule)>,
    /// The last damage fact dealt to each permanent since priority.
    damaged: Vec<(ObjectEventRef, FactId)>,
    /// Positions cut since the last take, oldest first.
    positions: Vec<Game>,
    /// Some fact has no position yet.
    uncut: bool,
}

impl PresentationSink {
    /// Push a fact whose cause, absent a trigger's or a resolution's, is
    /// `fallback`.
    fn push_caused(
        &mut self,
        fallback: Option<FactId>,
        rule: Option<Rule>,
        note: Note,
        lines: [Vec<PresentationLine>; 2],
    ) -> FactId {
        let id = self.next_id;
        self.next_id += 1;
        let cause = self.next_cause.take().or(self.resolution).or(fallback);
        self.facts.push(Fact {
            id,
            cause,
            rule,
            note,
            lines,
            position: None,
        });
        self.uncut = true;
        id
    }

    /// The rule behind a state-based death, and for lethal damage the
    /// damage fact that dealt it.
    fn event_rule(&self, event: &GameEvent) -> (Option<Rule>, Option<FactId>) {
        let GameEvent::PermanentsDied { objects } = event else {
            return (None, None);
        };
        let Some((object, rule)) = objects.iter().find_map(|object| {
            self.rules
                .iter()
                .find(|(died, _)| died == object)
                .map(|(_, rule)| (object, *rule))
        }) else {
            return (None, None);
        };
        let damage = (rule == Rule::LethalDamage)
            .then(|| {
                self.damaged
                    .iter()
                    .rev()
                    .find(|(damaged, _)| damaged == object)
                    .map(|(_, fact)| *fact)
            })
            .flatten();
        (Some(rule), damage)
    }
}

/// Committed lines for one viewer. An untold fact still binds its cause.
#[derive(Clone, Debug, Serialize)]
pub struct PresentationFact {
    pub id: FactId,
    pub cause: Option<FactId>,
    pub rule: Option<Rule>,
    pub lines: Vec<PresentationLine>,
    pub position: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PresentationLine {
    pub kind: PresentationKind,
    pub labels: PresentationLabels,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub combat: bool,
}

impl Game {
    /// Start telling this game's facts; Academy alone turns it on.
    pub fn enable_presentation(&mut self) {
        self.presentation = Some(Box::default());
    }

    /// Keep the board as it is now if a fact was told since the last cut;
    /// called only where the game is a whole state (see the module docs).
    pub(crate) fn presentation_cut(&mut self) {
        if !self.presentation.as_deref().is_some_and(|sink| sink.uncut) {
            return;
        }
        // `Game::clone` omits the sink, as for search.
        let mut position = self.clone();
        position.current_action_space = None;
        let sink = self.presentation.as_deref_mut().expect("checked above");
        let index = sink.positions.len() as u32;
        sink.positions.push(position);
        for fact in sink.facts.iter_mut().rev() {
            if fact.position.is_some() {
                break;
            }
            fact.position = Some(index);
        }
        sink.uncut = false;
    }

    pub(crate) fn presentation_event(&mut self, event: &GameEvent) {
        if self.presentation.is_none() {
            return;
        }
        let damaged = match event {
            GameEvent::DamageDealt {
                target: DamageTarget::Permanent(permanent),
                ..
            } => self.permanent_event_ref(*permanent),
            _ => None,
        };
        let lines = [PlayerId(0), PlayerId(1)].map(|viewer| self.event_lines(event, viewer));
        let sink = self.presentation.as_deref_mut().expect("checked above");
        // Combat damage is one fact: the observation tells it by the
        // `CombatDamageDealt` emitted right after its `DamageDealt`.
        if let GameEvent::CombatDamageDealt { .. } = event {
            if let Some(last) = sink.facts.last_mut() {
                if let Note::Event(dealt @ GameEvent::DamageDealt { combat: true, .. }) = &last.note
                {
                    last.note = Note::CombatDamage {
                        dealt: dealt.clone(),
                        told: event.clone(),
                    };
                    last.lines = lines;
                    return;
                }
            }
        }
        let (rule, damage) = sink.event_rule(event);
        if matches!(
            event,
            GameEvent::CardMoved {
                departed: Some(departed),
                to: ZoneType::Battlefield,
                ..
            } if sink.land_play == Some(*departed)
        ) {
            sink.land_play = None;
        }
        let id = sink.push_caused(damage, rule, Note::Event(event.clone()), lines);
        if let Some(object) = damaged {
            sink.damaged.push((object, id));
        }
    }

    pub(crate) fn presentation_activated(&mut self, ability: &StackObject) {
        if self.presentation.is_none() {
            return;
        }
        let lines =
            [PlayerId(0), PlayerId(1)].map(|viewer| self.stack_line(ability, viewer, false));
        self.presentation
            .as_deref_mut()
            .unwrap()
            .push_caused(None, None, Note::Stack, lines);
    }

    pub(crate) fn presentation_resolving(&mut self, stacked: &StackObject) {
        if self.presentation.is_none() {
            return;
        }
        let lines = [PlayerId(0), PlayerId(1)].map(|viewer| self.stack_line(stacked, viewer, true));
        let sink = self.presentation.as_deref_mut().unwrap();
        sink.resolution = None;
        let id = sink.push_caused(None, None, Note::Stack, lines);
        sink.resolution = Some(id);
    }

    /// Attach the hand appearance to the CardDrawn fact before another effect can move it.
    pub(super) fn presentation_drawn(&mut self, player: PlayerId, card: ObjectId, draw_step: bool) {
        if self.presentation.is_none() {
            return;
        }
        let lines = [PlayerId(0), PlayerId(1)].map(|viewer| self.draw_line(player, card, viewer));
        let fact = self
            .presentation
            .as_deref_mut()
            .unwrap()
            .facts
            .last_mut()
            .unwrap();
        debug_assert!(matches!(
            fact.note,
            Note::Event(GameEvent::CardDrawn { .. })
        ));
        fact.lines = lines;
        fact.rule = draw_step.then_some(Rule::DrawStep);
    }

    /// Stabilization reached priority: the resolution's consequences end.
    pub(crate) fn presentation_settled(&mut self) {
        if let Some(sink) = self.presentation.as_deref_mut() {
            sink.resolution = None;
            sink.rules.clear();
            sink.damaged.clear();
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
        let same = |fact: &&Fact| match &fact.note {
            Note::Event(told) => told == event,
            Note::CombatDamage { dealt, told } => dealt == event || told == event,
            _ => false,
        };
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

    pub fn presentation_facts(&self, viewer: PlayerId) -> Vec<PresentationFact> {
        let Some(sink) = self.presentation.as_deref() else {
            return Vec::new();
        };
        sink.facts
            .iter()
            .map(|fact| PresentationFact {
                id: fact.id,
                cause: fact.cause,
                rule: fact.rule,
                lines: fact.lines[viewer.0].clone(),
                position: fact.position,
            })
            .collect()
    }

    /// Release one step after all viewers have read their tracks.
    pub fn take_presentation_positions(&mut self) -> Vec<Game> {
        let Some(sink) = self.presentation.as_deref_mut() else {
            return Vec::new();
        };
        sink.facts.clear();
        sink.land_play = None;
        sink.uncut = false;
        std::mem::take(&mut sink.positions)
    }
}
