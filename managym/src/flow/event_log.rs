//! Logical event sequence owned by one match state.
//!
//! A dense vector: clones copy it, and search forks by cloning the whole match.

use std::{fmt, ops::Deref, slice};

use serde::Serialize;

use crate::flow::event::GameEvent;

#[derive(Clone, Default, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct EventLog(Vec<GameEvent>);

impl EventLog {
    pub fn push(&mut self, event: GameEvent) {
        self.0.push(event);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub(crate) fn take_vec(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.0)
    }
}

impl fmt::Debug for EventLog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(&self.0).finish()
    }
}

impl Deref for EventLog {
    type Target = [GameEvent];

    fn deref(&self) -> &[GameEvent] {
        &self.0
    }
}

impl PartialEq<Vec<GameEvent>> for EventLog {
    fn eq(&self, other: &Vec<GameEvent>) -> bool {
        &self.0 == other
    }
}

impl<'a> IntoIterator for &'a EventLog {
    type Item = &'a GameEvent;
    type IntoIter = slice::Iter<'a, GameEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for EventLog {
    type Item = GameEvent;
    type IntoIter = std::vec::IntoIter<GameEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
