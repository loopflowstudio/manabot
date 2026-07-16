use std::array;

use rand::seq::SliceRandom;

use super::game_object::{CardId, PlayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[repr(i32)]
pub enum ZoneType {
    Library = 0,
    Hand = 1,
    Battlefield = 2,
    Graveyard = 3,
    Stack = 4,
    Exile = 5,
    Command = 6,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ZoneManager {
    library: [Vec<CardId>; 2],
    hand: [Vec<CardId>; 2],
    battlefield: [Vec<CardId>; 2],
    graveyard: [Vec<CardId>; 2],
    stack: [Vec<CardId>; 2],
    exile: [Vec<CardId>; 2],
    command: [Vec<CardId>; 2],
    card_zones: Vec<Option<ZoneType>>,
}

impl Default for ZoneManager {
    fn default() -> Self {
        Self {
            library: array::from_fn(|_| Vec::new()),
            hand: array::from_fn(|_| Vec::new()),
            battlefield: array::from_fn(|_| Vec::new()),
            graveyard: array::from_fn(|_| Vec::new()),
            stack: array::from_fn(|_| Vec::new()),
            exile: array::from_fn(|_| Vec::new()),
            command: array::from_fn(|_| Vec::new()),
            card_zones: Vec::new(),
        }
    }
}

impl ZoneManager {
    pub(crate) fn allocated_bytes(&self) -> usize {
        use std::mem::size_of;

        let ordered = self
            .library
            .iter()
            .chain(self.hand.iter())
            .chain(self.battlefield.iter())
            .chain(self.graveyard.iter())
            .chain(self.stack.iter())
            .chain(self.exile.iter())
            .chain(self.command.iter())
            .map(|cards| cards.capacity().saturating_mul(size_of::<CardId>()))
            .sum::<usize>();
        ordered
            + self
                .card_zones
                .capacity()
                .saturating_mul(size_of::<Option<ZoneType>>())
    }

    pub(crate) fn card_zones_len(&self) -> usize {
        self.card_zones.len()
    }

    /// Where `card` currently sits, precisely enough to put it back: zone plus
    /// its index within that zone's order. `None` means it is in no zone.
    pub(crate) fn locate(&self, card: CardId, owner: PlayerId) -> Option<(ZoneType, usize)> {
        let zone = self.zone_of(card)?;
        let index = self
            .zone_cards(zone, owner)
            .iter()
            .position(|held| *held == card)?;
        Some((zone, index))
    }

    /// Inverse of a single [`Self::move_card`] or [`Self::remove_card`].
    ///
    /// Restores zone order exactly by reinserting at the recorded index rather
    /// than pushing, and rewinds the `card_zones` reverse index to the length
    /// it had before the move, undoing any slot the move appended for a token.
    pub(crate) fn undo_move(
        &mut self,
        card: CardId,
        owner: PlayerId,
        from: Option<(ZoneType, usize)>,
        card_zones_len: usize,
    ) {
        if let Some(zone) = self.zone_of(card) {
            self.remove_from_zone(card, owner, zone);
        }
        match from {
            Some((zone, index)) => {
                let cards = self.zone_cards_mut(zone, owner);
                let index = index.min(cards.len());
                cards.insert(index, card);
                self.card_zones[card.0] = Some(zone);
            }
            None => {
                if let Some(slot) = self.card_zones.get_mut(card.0) {
                    *slot = None;
                }
            }
        }
        self.card_zones.truncate(card_zones_len);
    }

    fn ensure_slot(&mut self, card: CardId) {
        if self.card_zones.len() <= card.0 {
            self.card_zones.resize(card.0 + 1, None);
        }
    }

    pub fn zone_of(&self, card: CardId) -> Option<ZoneType> {
        self.card_zones.get(card.0).copied().flatten()
    }

    fn remove_card_in_place(cards: &mut Vec<CardId>, card: CardId) {
        cards.retain(|c| *c != card);
    }

    fn remove_from_zone(&mut self, card: CardId, owner: PlayerId, zone: ZoneType) {
        let idx = owner.0;
        match zone {
            ZoneType::Library => Self::remove_card_in_place(&mut self.library[idx], card),
            ZoneType::Hand => Self::remove_card_in_place(&mut self.hand[idx], card),
            ZoneType::Battlefield => Self::remove_card_in_place(&mut self.battlefield[idx], card),
            ZoneType::Graveyard => Self::remove_card_in_place(&mut self.graveyard[idx], card),
            ZoneType::Stack => Self::remove_card_in_place(&mut self.stack[idx], card),
            ZoneType::Exile => Self::remove_card_in_place(&mut self.exile[idx], card),
            ZoneType::Command => Self::remove_card_in_place(&mut self.command[idx], card),
        }
    }

    fn insert_to_zone(&mut self, card: CardId, owner: PlayerId, zone: ZoneType) {
        let idx = owner.0;
        match zone {
            ZoneType::Library => self.library[idx].push(card),
            ZoneType::Hand => self.hand[idx].push(card),
            ZoneType::Battlefield => self.battlefield[idx].push(card),
            ZoneType::Graveyard => self.graveyard[idx].push(card),
            ZoneType::Stack => self.stack[idx].push(card),
            ZoneType::Exile => self.exile[idx].push(card),
            ZoneType::Command => self.command[idx].push(card),
        }
    }

    /// Remove a card from all zones entirely (CR 704.5d — tokens cease to
    /// exist). The card object remains in `GameState::cards` but is no
    /// longer in any zone.
    pub fn remove_card(&mut self, card: CardId, owner: PlayerId) {
        if let Some(zone) = self.zone_of(card) {
            self.remove_from_zone(card, owner, zone);
            self.card_zones[card.0] = None;
        }
    }

    pub fn move_card(&mut self, card: CardId, owner: PlayerId, to_zone: ZoneType) {
        self.ensure_slot(card);
        if let Some(old_zone) = self.zone_of(card) {
            self.remove_from_zone(card, owner, old_zone);
        }
        self.insert_to_zone(card, owner, to_zone);
        self.card_zones[card.0] = Some(to_zone);
    }

    pub fn top(&self, zone: ZoneType, player: PlayerId) -> Option<CardId> {
        let cards = self.zone_cards(zone, player);
        cards.last().copied()
    }

    pub fn contains(&self, card: CardId, zone: ZoneType, player: PlayerId) -> bool {
        self.zone_cards(zone, player).contains(&card)
    }

    pub fn size(&self, zone: ZoneType, player: PlayerId) -> usize {
        self.zone_cards(zone, player).len()
    }

    pub fn total_size(&self, zone: ZoneType) -> usize {
        self.size(zone, PlayerId(0)) + self.size(zone, PlayerId(1))
    }

    pub fn zone_cards(&self, zone: ZoneType, player: PlayerId) -> &Vec<CardId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &self.library[idx],
            ZoneType::Hand => &self.hand[idx],
            ZoneType::Battlefield => &self.battlefield[idx],
            ZoneType::Graveyard => &self.graveyard[idx],
            ZoneType::Stack => &self.stack[idx],
            ZoneType::Exile => &self.exile[idx],
            ZoneType::Command => &self.command[idx],
        }
    }

    pub fn zone_cards_mut(&mut self, zone: ZoneType, player: PlayerId) -> &mut Vec<CardId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &mut self.library[idx],
            ZoneType::Hand => &mut self.hand[idx],
            ZoneType::Battlefield => &mut self.battlefield[idx],
            ZoneType::Graveyard => &mut self.graveyard[idx],
            ZoneType::Stack => &mut self.stack[idx],
            ZoneType::Exile => &mut self.exile[idx],
            ZoneType::Command => &mut self.command[idx],
        }
    }

    pub fn shuffle<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        self.zone_cards_mut(zone, player).shuffle(rng);
    }

    /// Resample `player`'s hidden zones (hand + library) as one pool.
    ///
    /// The union of hand and library is shuffled, then split back so the hand
    /// keeps its original size. Used by determinization: from the opponent's
    /// perspective these cards are exchangeable, so a uniform re-deal over the
    /// pool is exactly a sample from the belief state (public zones untouched).
    pub fn resample_hidden<R: rand::Rng + ?Sized>(&mut self, player: PlayerId, rng: &mut R) {
        let idx = player.0;
        let hand_size = self.hand[idx].len();
        let mut pool: Vec<CardId> = Vec::with_capacity(hand_size + self.library[idx].len());
        pool.append(&mut self.hand[idx]);
        pool.append(&mut self.library[idx]);
        pool.shuffle(rng);
        let library = pool.split_off(hand_size);
        for card in &pool {
            self.card_zones[card.0] = Some(ZoneType::Hand);
        }
        for card in &library {
            self.card_zones[card.0] = Some(ZoneType::Library);
        }
        self.hand[idx] = pool;
        self.library[idx] = library;
    }
}
