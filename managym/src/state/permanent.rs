use super::{
    card::{ActivatedAbilityDefinition, Card, Keywords},
    game_object::{CardId, ObjectId, ObjectRef, PlayerId},
};

/// Counter kinds the engine knows (CR 122). A new kind is one variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    PlusOnePlusOne,
    /// CR 122.1d: replaces the next untap.
    Stun,
    /// No rules meaning of its own; cards count it.
    Fire,
    /// CR 122.1b keyword counter.
    DoubleStrike,
}

impl CounterKind {
    /// The serialized name (`stun`, `double_strike`, ...).
    pub fn name(self) -> &'static str {
        match self {
            CounterKind::PlusOnePlusOne => "plus_one_plus_one",
            CounterKind::Stun => "stun",
            CounterKind::Fire => "fire",
            CounterKind::DoubleStrike => "double_strike",
        }
    }

    pub const ALL: [CounterKind; 4] = [
        CounterKind::PlusOnePlusOne,
        CounterKind::Stun,
        CounterKind::Fire,
        CounterKind::DoubleStrike,
    ];
}

/// Counters on one object by kind (CR 122.1), +1/+1 included. Serialized as
/// a map of the kinds present, in `CounterKind` order, so hashes do not see
/// absent kinds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters([i32; CounterKind::ALL.len()]);

impl Counters {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn get(&self, kind: CounterKind) -> i32 {
        self.0[kind as usize]
    }

    /// Add `delta` counters of `kind`, clamped at zero. Returns whether the
    /// count changed.
    pub fn change(&mut self, kind: CounterKind, delta: i32) -> bool {
        let count = &mut self.0[kind as usize];
        let old = *count;
        *count = old.saturating_add(delta).max(0);
        *count != old
    }

    /// The kinds present, with their counts, in `CounterKind` order.
    pub fn iter(&self) -> impl Iterator<Item = (CounterKind, i32)> + '_ {
        CounterKind::ALL
            .into_iter()
            .map(|kind| (kind, self.get(kind)))
            .filter(|(_, count)| *count > 0)
    }
}

impl serde::Serialize for Counters {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Permanent {
    pub id: ObjectId,
    pub card: CardId,
    pub controller: PlayerId,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: i32,
    pub deathtouch_damage: bool,
    /// Dealt damage this turn ("if it was dealt damage this turn"), whether
    /// or not that damage is still marked; cleared during cleanup.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Counters of every kind (CR 122.1). Any permanent can carry them,
    /// including lands; +1/+1 counters change P/T only through
    /// `Game::effective_pt`.
    #[serde(skip_serializing_if = "Counters::is_empty")]
    pub counters: Counters,
    /// Its timestamp (CR 613.7d/e): when it entered, renewed when an Aura or
    /// Equipment becomes attached. Static abilities apply in this order among
    /// continuous-effect records.
    pub timestamp: ObjectId,
    /// Earthbend animation: this permanent is a 0/0 creature with haste in
    /// addition to its printed types (still a land). No duration — it lasts
    /// as long as the permanent does.
    pub animated: bool,
    pub attacking: bool,
    /// True only for this incarnation entering as a resolving cast spell.
    pub was_cast: bool,
    /// The exact object this Equipment is attached to (CR 301.5). State-based
    /// actions clear it once that object is gone or no longer a creature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<ObjectRef>,
    /// Exhaust abilities (indices into the card's activated abilities) this
    /// object has activated; a new object starts fresh (CR 400.7).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exhausted: Vec<usize>,
    /// "Once each turn" abilities this object has activated this turn,
    /// cleared at cleanup.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub activated_this_turn: Vec<usize>,
}

impl Permanent {
    /// Whether this object has used up the ability: exhaust once ever
    /// (CR 702.177a), "once each turn" already this turn (CR 602.5b).
    pub fn activation_used(&self, index: usize, ability: &ActivatedAbilityDefinition) -> bool {
        (ability.exhaust && self.exhausted.contains(&index))
            || (ability.once_each_turn && self.activated_this_turn.contains(&index))
    }

    /// Record an activation against the ability's use limits.
    pub fn record_activation(&mut self, index: usize, ability: &ActivatedAbilityDefinition) {
        if ability.exhaust {
            self.exhausted.push(index);
        }
        if ability.once_each_turn {
            self.activated_this_turn.push(index);
        }
    }

    pub fn new(id: ObjectId, card_id: CardId, card: &Card) -> Self {
        Self {
            id,
            card: card_id,
            controller: card.owner,
            tapped: false,
            // CR 302.6 applies to any permanent that is a creature; only
            // creatures and Vehicles can attack or pay {T} as creatures here.
            summoning_sick: (card.types.is_creature() || card.has_subtype("Vehicle"))
                && !card.keywords.haste,
            damage: 0,
            deathtouch_damage: false,
            dealt_damage_this_turn: false,
            timestamp: id,
            counters: Counters::default(),
            animated: false,
            attacking: false,
            was_cast: false,
            attached_to: None,
            exhausted: Vec::new(),
            activated_this_turn: Vec::new(),
        }
    }

    /// Whether the permanent itself makes it a creature: earthbend (a land
    /// that's also a 0/0 creature) or its own counter-threshold static (War
    /// Balloon). `Game::permanent_is_creature` adds type-changing effects.
    pub fn animated_creature(&self, card: &Card) -> bool {
        self.animated
            || card
                .creature_with_counters
                .is_some_and(|threshold| self.counter_count(threshold.kind) >= threshold.count)
    }

    /// Printed keywords plus animation (earthbent lands have haste) and
    /// keyword counters. An input to `Game::effective_keywords`, which
    /// applies continuous effects and static grants in layer 6.
    pub fn intrinsic_keywords(&self, card: &Card) -> Keywords {
        let mut keywords = card.keywords;
        if self.animated {
            keywords.haste = true;
        }
        // CR 122.1b: a keyword counter grants its keyword while it remains.
        if self.counters.get(CounterKind::DoubleStrike) > 0 {
            keywords.double_strike = true;
        }
        keywords
    }

    /// CR 122.1d: a tapped permanent with a stun counter that would become
    /// untapped stays tapped and loses one stun counter instead. Every untap
    /// path (untap step, effects, costs) goes through here.
    pub fn untap(&mut self) {
        if self.tapped && self.counters.get(CounterKind::Stun) > 0 {
            self.counters.change(CounterKind::Stun, -1);
        } else {
            self.tapped = false;
        }
    }

    pub fn counter_count(&self, kind: CounterKind) -> i32 {
        self.counters.get(kind)
    }

    /// Add `delta` counters of `kind`, clamped at zero. Returns whether the
    /// count changed.
    pub fn change_counters(&mut self, kind: CounterKind, delta: i32) -> bool {
        self.counters.change(kind, delta)
    }

    /// "With a counter on it" / "with no counters on it" read every kind.
    pub fn has_counters(&self) -> bool {
        !self.counters.is_empty()
    }

    pub fn tap(&mut self) {
        self.tapped = true;
    }

    pub fn clear_damage(&mut self) {
        self.damage = 0;
        self.deathtouch_damage = false;
    }

    pub fn take_damage(&mut self, amount: i32) {
        self.damage += amount;
        if amount > 0 {
            self.dealt_damage_this_turn = true;
        }
    }

    /// "Once each turn" limits and "dealt damage this turn" reset in the
    /// cleanup step.
    pub fn clear_turn_history(&mut self) {
        self.dealt_damage_this_turn = false;
        self.activated_this_turn.clear();
    }
}
