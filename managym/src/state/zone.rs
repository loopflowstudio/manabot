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
    /// The player whose `zone` holds `card`.
    fn holder_in(&self, card: CardId, zone: ZoneType) -> Option<PlayerId> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .find(|player| self.zone_cards(zone, *player).contains(&card))
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

    fn remove_from_zone(&mut self, card: CardId, zone: ZoneType) {
        let Some(holder) = self.holder_in(card, zone) else {
            return;
        };
        let idx = holder.0;
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

    fn insert_to_zone(&mut self, card: CardId, holder: PlayerId, zone: ZoneType) {
        let idx = holder.0;
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
    pub fn remove_card(&mut self, card: CardId) {
        if let Some(zone) = self.zone_of(card) {
            self.remove_from_zone(card, zone);
            self.card_zones[card.0] = None;
        }
    }

    /// Move `card` from wherever it is to the end of `holder`'s `to_zone`.
    /// The holder is the card's owner (CR 108.3), except on the
    /// battlefield, where it is the permanent's controller (CR 110.2): each
    /// player's battlefield lists the permanents they control.
    pub fn move_card(&mut self, card: CardId, holder: PlayerId, to_zone: ZoneType) {
        self.ensure_slot(card);
        if let Some(old_zone) = self.zone_of(card) {
            self.remove_from_zone(card, old_zone);
        }
        self.insert_to_zone(card, holder, to_zone);
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

    /// Shuffle an unordered hidden zone from a canonical starting order.
    ///
    /// Belief-state sampling must depend on the viewer-visible card multiset
    /// and seed, never on the authority's current private ordering.
    pub fn shuffle_canonical<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        let cards = self.zone_cards_mut(zone, player);
        cards.sort_unstable_by_key(|card| card.0);
        cards.shuffle(rng);
    }

    /// Put hidden card `new` where `player`'s face-down exiled card `old`
    /// lies. `old` is left zoneless until the caller reassigns it to the
    /// hand or library with [`ZoneManager::reassign_hidden`].
    pub(crate) fn replace_face_down_exile(&mut self, player: PlayerId, old: CardId, new: CardId) {
        if let Some(slot) = self.exile[player.0].iter_mut().find(|card| **card == old) {
            *slot = new;
        }
        self.ensure_slot(new);
        self.card_zones[new.0] = Some(ZoneType::Exile);
        self.card_zones[old.0] = None;
    }

    /// Reassign `player`'s hand and library to exactly `new_hand` and
    /// `new_library`, updating `card_zones` for every card. Used by exact
    /// possible-world materialization: the caller chooses which physical
    /// cards land in which hidden zone, so the resulting branch is a valid
    /// authoritative state rather than a uniform sample. The union of
    /// `new_hand` and `new_library` must be exactly the prior hand ∪ library.
    pub(crate) fn reassign_hidden(
        &mut self,
        player: PlayerId,
        new_hand: Vec<CardId>,
        new_library: Vec<CardId>,
    ) {
        let idx = player.0;
        for &card in &new_hand {
            self.ensure_slot(card);
            self.card_zones[card.0] = Some(ZoneType::Hand);
        }
        for &card in &new_library {
            self.ensure_slot(card);
            self.card_zones[card.0] = Some(ZoneType::Library);
        }
        self.hand[idx] = new_hand;
        self.library[idx] = new_library;
    }
}
