use std::{
    fmt,
    mem::size_of,
    slice,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::flow::event::{GameEvent, ObjectEventRef};

pub(crate) const EVENT_PAGE_BYTES: usize = 4 * 1024;

fn page_capacity() -> usize {
    (EVENT_PAGE_BYTES / size_of::<GameEvent>()).max(1)
}

fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

fn event_owned_bytes(event: &GameEvent) -> usize {
    match event {
        GameEvent::SpellCast { targets, .. } => vec_bytes(targets),
        GameEvent::AttackersDeclared { attackers, .. } => vec_bytes(attackers),
        GameEvent::CombatAttackersDeclared { attackers, .. }
        | GameEvent::PermanentsDied { objects: attackers } => vec_bytes(attackers),
        GameEvent::BlockersDeclared { assignments } => {
            vec_bytes(assignments)
                + assignments
                    .iter()
                    .map(|(_, blockers): &(ObjectEventRef, Vec<ObjectEventRef>)| {
                        vec_bytes(blockers)
                    })
                    .sum::<usize>()
        }
        _ => 0,
    }
}

#[derive(Debug, Default)]
pub(crate) struct CowStats {
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

impl CowStats {
    fn allocate(&self, bytes: usize) {
        let current = self
            .current_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed)
            + bytes as u64;
        self.peak_bytes.fetch_max(current, Ordering::Relaxed);
    }

    fn release(&self, bytes: usize) {
        self.current_bytes
            .fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn peak_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }
}

struct CowAllocation {
    stats: Arc<CowStats>,
    bytes: usize,
}

impl CowAllocation {
    fn new(stats: Arc<CowStats>, bytes: usize) -> Self {
        stats.allocate(bytes);
        Self { stats, bytes }
    }
}

impl Drop for CowAllocation {
    fn drop(&mut self) {
        self.stats.release(self.bytes);
    }
}

#[doc(hidden)]
pub struct EventPage {
    events: Vec<GameEvent>,
    _allocation: Option<CowAllocation>,
}

impl EventPage {
    fn clean(events: Vec<GameEvent>) -> Self {
        Self {
            events,
            _allocation: None,
        }
    }

    fn copied(events: &[GameEvent], stats: Arc<CowStats>) -> Self {
        let mut copied = Vec::with_capacity(page_capacity());
        copied.extend_from_slice(events);
        let bytes = vec_bytes(&copied) + copied.iter().map(event_owned_bytes).sum::<usize>();
        Self {
            events: copied,
            _allocation: Some(CowAllocation::new(stats, bytes)),
        }
    }
}

#[doc(hidden)]
pub struct PagedEventLog {
    pages: Vec<Arc<EventPage>>,
    len: usize,
    stats: Arc<CowStats>,
}

impl PagedEventLog {
    fn from_events(events: Vec<GameEvent>, stats: Arc<CowStats>) -> Self {
        let capacity = page_capacity();
        let mut pages = Vec::with_capacity(events.len().div_ceil(capacity));
        let mut source = events.into_iter();
        loop {
            let mut page = Vec::with_capacity(capacity);
            page.extend(source.by_ref().take(capacity));
            if page.is_empty() {
                break;
            }
            pages.push(Arc::new(EventPage::clean(page)));
        }
        let len = pages.iter().map(|page| page.events.len()).sum();
        Self { pages, len, stats }
    }

    fn fork_shared(&self, stats: Arc<CowStats>) -> Self {
        Self {
            pages: self.pages.clone(),
            len: self.len,
            stats,
        }
    }

    fn ensure_unique(&mut self, index: usize) -> &mut EventPage {
        if Arc::get_mut(&mut self.pages[index]).is_none() {
            let copied = EventPage::copied(&self.pages[index].events, self.stats.clone());
            self.pages[index] = Arc::new(copied);
        }
        Arc::get_mut(&mut self.pages[index]).expect("page was made unique")
    }

    fn push(&mut self, event: GameEvent) {
        let needs_page = self
            .pages
            .last()
            .is_none_or(|page| page.events.len() == page_capacity());
        if needs_page {
            let mut events = Vec::with_capacity(page_capacity());
            events.push(event);
            self.pages.push(Arc::new(EventPage::clean(events)));
        } else {
            let last = self.pages.len() - 1;
            self.ensure_unique(last).events.push(event);
        }
        self.len += 1;
    }

    fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        if len == 0 {
            self.pages.clear();
            self.len = 0;
            return;
        }
        let capacity = page_capacity();
        let retained_pages = len.div_ceil(capacity);
        self.pages.truncate(retained_pages);
        let tail_len = len - (retained_pages - 1) * capacity;
        let tail = retained_pages - 1;
        if self.pages[tail].events.len() != tail_len {
            self.ensure_unique(tail).events.truncate(tail_len);
        }
        self.len = len;
    }
}

/// Logical event sequence with an opt-in fixed-page COW representation.
///
/// Ordinary engine states stay dense. `Clone` always produces a deep dense
/// sequence; only the page-COW driver calls `fork_shared`.
pub enum EventLog {
    Dense(Vec<GameEvent>),
    Paged(PagedEventLog),
}

impl Default for EventLog {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl Clone for EventLog {
    fn clone(&self) -> Self {
        Self::Dense(self.iter().cloned().collect())
    }
}

impl fmt::Debug for EventLog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

impl EventLog {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(events) => events.len(),
            Self::Paged(events) => events.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, event: GameEvent) {
        match self {
            Self::Dense(events) => events.push(event),
            Self::Paged(events) => events.push(event),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Dense(events) => events.clear(),
            Self::Paged(events) => {
                events.pages.clear();
                events.len = 0;
            }
        }
    }

    pub fn truncate(&mut self, len: usize) {
        match self {
            Self::Dense(events) => events.truncate(len),
            Self::Paged(events) => events.truncate(len),
        }
    }

    pub fn iter(&self) -> EventLogIter<'_> {
        match self {
            Self::Dense(events) => EventLogIter::Dense(events.iter()),
            Self::Paged(events) => EventLogIter::Paged {
                pages: events.pages.iter(),
                current: None,
                remaining: events.len,
            },
        }
    }

    pub(crate) fn into_vec(self) -> Vec<GameEvent> {
        match self {
            Self::Dense(events) => events,
            Self::Paged(events) => events
                .pages
                .into_iter()
                .flat_map(|page| page.events.clone())
                .collect(),
        }
    }

    pub(crate) fn take_vec(&mut self) -> Vec<GameEvent> {
        std::mem::take(self).into_vec()
    }

    pub(crate) fn make_paged_root(&mut self) {
        if matches!(self, Self::Paged(_)) {
            return;
        }
        let dense = std::mem::take(self).into_vec();
        *self = Self::Paged(PagedEventLog::from_events(
            dense,
            Arc::new(CowStats::default()),
        ));
    }

    pub(crate) fn fork_shared(&self, stats: Arc<CowStats>) -> Self {
        match self {
            Self::Dense(events) => Self::Paged(PagedEventLog::from_events(events.clone(), stats)),
            Self::Paged(events) => Self::Paged(events.fork_shared(stats)),
        }
    }

    pub(crate) fn journal_snapshot(&self) -> Self {
        match self {
            Self::Dense(events) => Self::Dense(events.clone()),
            Self::Paged(events) => Self::Paged(events.fork_shared(events.stats.clone())),
        }
    }

    pub(crate) fn owned_bytes(&self) -> usize {
        match self {
            Self::Dense(events) => {
                vec_bytes(events) + events.iter().map(event_owned_bytes).sum::<usize>()
            }
            Self::Paged(events) => events.pages.capacity() * size_of::<Arc<EventPage>>(),
        }
    }
}

impl Serialize for EventLog {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for event in self {
            sequence.serialize_element(event)?;
        }
        sequence.end()
    }
}

impl PartialEq for EventLog {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for EventLog {}

impl PartialEq<Vec<GameEvent>> for EventLog {
    fn eq(&self, other: &Vec<GameEvent>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

pub enum EventLogIter<'a> {
    Dense(slice::Iter<'a, GameEvent>),
    Paged {
        pages: slice::Iter<'a, Arc<EventPage>>,
        current: Option<slice::Iter<'a, GameEvent>>,
        remaining: usize,
    },
}

impl<'a> Iterator for EventLogIter<'a> {
    type Item = &'a GameEvent;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dense(events) => events.next(),
            Self::Paged {
                pages,
                current,
                remaining,
            } => loop {
                if let Some(event) = current.as_mut().and_then(Iterator::next) {
                    *remaining -= 1;
                    return Some(event);
                }
                let page = pages.next()?;
                *current = Some(page.events.iter());
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Dense(events) => events.size_hint(),
            Self::Paged { remaining, .. } => (*remaining, Some(*remaining)),
        }
    }
}

impl ExactSizeIterator for EventLogIter<'_> {}

impl<'a> IntoIterator for &'a EventLog {
    type Item = &'a GameEvent;
    type IntoIter = EventLogIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for EventLog {
    type Item = GameEvent;
    type IntoIter = std::vec::IntoIter<GameEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flow::turn::StepKind, state::game_object::PlayerId};

    fn event(index: u32) -> GameEvent {
        GameEvent::TurnStarted {
            player: PlayerId((index as usize) % 2),
            turn_number: index,
        }
    }

    #[test]
    fn paged_sequence_serializes_and_compares_like_dense() {
        let source: Vec<_> = (0..(page_capacity() as u32 + 7)).map(event).collect();
        let dense = EventLog::Dense(source.clone());
        let paged = EventLog::Paged(PagedEventLog::from_events(
            source.clone(),
            Arc::new(CowStats::default()),
        ));
        assert_eq!(dense, paged);
        assert_eq!(
            serde_json::to_vec(&dense).unwrap(),
            serde_json::to_vec(&source).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&paged).unwrap(),
            serde_json::to_vec(&source).unwrap()
        );
    }

    #[test]
    fn fork_shares_clean_pages_and_detaches_only_the_written_tail() {
        let source: Vec<_> = (0..(page_capacity() as u32 + 1)).map(event).collect();
        let mut root = EventLog::Paged(PagedEventLog::from_events(
            source.clone(),
            Arc::new(CowStats::default()),
        ));
        let stats = Arc::new(CowStats::default());
        let mut branch = root.fork_shared(stats.clone());

        let (root_pages, branch_pages) = match (&root, &branch) {
            (EventLog::Paged(root), EventLog::Paged(branch)) => (&root.pages, &branch.pages),
            _ => unreachable!(),
        };
        assert!(Arc::ptr_eq(&root_pages[0], &branch_pages[0]));
        assert!(Arc::ptr_eq(&root_pages[1], &branch_pages[1]));

        branch.push(GameEvent::StepStarted {
            step: StepKind::Upkeep,
        });
        let (root_pages, branch_pages) = match (&root, &branch) {
            (EventLog::Paged(root), EventLog::Paged(branch)) => (&root.pages, &branch.pages),
            _ => unreachable!(),
        };
        assert!(Arc::ptr_eq(&root_pages[0], &branch_pages[0]));
        assert!(!Arc::ptr_eq(&root_pages[1], &branch_pages[1]));
        assert_eq!(root, source);
        assert!(stats.peak_bytes() >= (EVENT_PAGE_BYTES - size_of::<GameEvent>()) as u64);

        root.push(event(999));
        assert_ne!(root, branch);
    }
}
