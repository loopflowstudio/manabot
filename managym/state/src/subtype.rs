//! Subtype categories (CR 205.3): each subtype belongs to exactly one card
//! type's list, so "Cave" is a land type, "Shrine" an enchantment type and
//! "Lesson" a spell type wherever it appears. Definitions, modifications and
//! predicates keep subtypes as their printed names; this table is the one
//! authority that says which list a name is on. Rules that read creature
//! types (CR 205.3m: a creature's creature types, a creature-type replacement
//! that keeps other subtypes) ask `is_creature_type`, never infer it by
//! excluding a partial list of other categories.
//!
//! The table holds every subtype the catalog authors (the cards crate's
//! definition checks fail on an unlisted name) and the complete short CR
//! lists of land, artifact, enchantment and spell types.

/// The card type whose subtype list contains a name (CR 205.3g–m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtypeCategory {
    Artifact,
    Creature,
    Enchantment,
    Land,
    Spell,
}

use SubtypeCategory::{Artifact, Creature, Enchantment, Land, Spell};

/// Sorted by name for binary search; `subtype_table_is_sorted` checks it.
const SUBTYPES: &[(&str, SubtypeCategory)] = &[
    ("Adventure", Spell),
    ("Advisor", Creature),
    ("Ally", Creature),
    ("Angel", Creature),
    ("Ape", Creature),
    ("Arcane", Spell),
    ("Archer", Creature),
    ("Armadillo", Creature),
    ("Artificer", Creature),
    ("Assassin", Creature),
    ("Attraction", Artifact),
    ("Aura", Enchantment),
    ("Avatar", Creature),
    ("Background", Enchantment),
    ("Badger", Creature),
    ("Bard", Creature),
    ("Bat", Creature),
    ("Bear", Creature),
    ("Beast", Creature),
    ("Berserker", Creature),
    ("Bird", Creature),
    ("Bison", Creature),
    ("Blood", Artifact),
    ("Boar", Creature),
    ("Bobblehead", Artifact),
    ("Cartouche", Enchantment),
    ("Case", Enchantment),
    ("Cat", Creature),
    ("Cave", Land),
    ("Chorus", Spell),
    ("Citizen", Creature),
    ("Class", Enchantment),
    ("Cleric", Creature),
    ("Cloud", Land),
    ("Clue", Artifact),
    ("Construct", Creature),
    ("Contraption", Artifact),
    ("Coward", Creature),
    ("Crocodile", Creature),
    ("Curse", Enchantment),
    ("Demon", Creature),
    ("Desert", Land),
    ("Detective", Creature),
    ("Devil", Creature),
    ("Dinosaur", Creature),
    ("Djinn", Creature),
    ("Dog", Creature),
    ("Dragon", Creature),
    ("Drake", Creature),
    ("Druid", Creature),
    ("Elder", Creature),
    ("Eldrazi", Creature),
    ("Elemental", Creature),
    ("Elephant", Creature),
    ("Elf", Creature),
    ("Elk", Creature),
    ("Equipment", Artifact),
    ("Eye", Creature),
    ("Faerie", Creature),
    ("Fish", Creature),
    ("Food", Artifact),
    ("Forest", Land),
    ("Fortification", Artifact),
    ("Fox", Creature),
    ("Frog", Creature),
    ("Gate", Land),
    ("Giant", Creature),
    ("Goat", Creature),
    ("Goblin", Creature),
    ("Gold", Artifact),
    ("Hero", Creature),
    ("Hippo", Creature),
    ("Homunculus", Creature),
    ("Horse", Creature),
    ("Human", Creature),
    ("Incubator", Artifact),
    ("Infinity", Artifact),
    ("Insect", Creature),
    ("Island", Land),
    ("Jellyfish", Creature),
    ("Junk", Artifact),
    ("Kangaroo", Creature),
    ("Knight", Creature),
    ("Kor", Creature),
    ("Lair", Land),
    ("Lander", Artifact),
    ("Lemur", Creature),
    ("Lesson", Spell),
    ("Lizard", Creature),
    ("Locus", Land),
    ("Map", Artifact),
    ("Mercenary", Creature),
    ("Mine", Land),
    ("Mole", Creature),
    ("Mongoose", Creature),
    ("Monk", Creature),
    ("Monkey", Creature),
    ("Mount", Creature),
    ("Mountain", Land),
    ("Mutagen", Artifact),
    ("Mutant", Creature),
    ("Noble", Creature),
    ("Ogre", Creature),
    ("Omen", Spell),
    ("Otter", Creature),
    ("Ox", Creature),
    ("Peasant", Creature),
    ("Performer", Creature),
    ("Pilot", Creature),
    ("Pirate", Creature),
    ("Plains", Land),
    ("Plan", Enchantment),
    ("Plant", Creature),
    ("Platypus", Creature),
    ("Porcupine", Creature),
    ("Power-Plant", Land),
    ("Powerstone", Artifact),
    ("Rabbit", Creature),
    ("Ranger", Creature),
    ("Rat", Creature),
    ("Rebel", Creature),
    ("Rhino", Creature),
    ("Robot", Creature),
    ("Rogue", Creature),
    ("Role", Enchantment),
    ("Room", Enchantment),
    ("Rune", Enchantment),
    ("Saga", Enchantment),
    ("Scarecrow", Creature),
    ("Scout", Creature),
    ("Seal", Creature),
    ("Serpent", Creature),
    ("Shaman", Creature),
    ("Shapeshifter", Creature),
    ("Shard", Enchantment),
    ("Sheep", Creature),
    ("Shrine", Enchantment),
    ("Skeleton", Creature),
    ("Sloth", Creature),
    ("Snake", Creature),
    ("Soldier", Creature),
    ("Spacecraft", Artifact),
    ("Sphere", Land),
    ("Spider", Creature),
    ("Spirit", Creature),
    ("Squirrel", Creature),
    ("Stone", Artifact),
    ("Swamp", Land),
    ("Terminus", Artifact),
    ("Tower", Land),
    ("Town", Land),
    ("Trap", Spell),
    ("Treasure", Artifact),
    ("Treefolk", Creature),
    ("Turtle", Creature),
    ("Urza's", Land),
    ("Vampire", Creature),
    ("Vehicle", Artifact),
    ("Villain", Creature),
    ("Wall", Creature),
    ("Warlock", Creature),
    ("Warrior", Creature),
    ("Whale", Creature),
    ("Wizard", Creature),
    ("Wolf", Creature),
    ("Wurm", Creature),
    ("Zombie", Creature),
];

/// Every subtype in the table, in name order.
pub fn subtypes() -> impl Iterator<Item = &'static str> {
    SUBTYPES.iter().map(|(name, _)| *name)
}

/// The list `subtype` is on, or `None` for a name outside the table.
pub fn subtype_category(subtype: &str) -> Option<SubtypeCategory> {
    SUBTYPES
        .binary_search_by(|(name, _)| (*name).cmp(subtype))
        .ok()
        .map(|index| SUBTYPES[index].1)
}

/// Whether `subtype` is on the list of a card type the object has
/// (CR 205.3d): kindred uses creature types (CR 205.3m), and instants and
/// sorceries share spell types (CR 205.3k). Planeswalker and battle types are
/// not in the table, so nothing corresponds to them.
pub fn subtype_corresponds_to(subtype: &str, card_type: crate::card::CardType) -> bool {
    use crate::card::CardType as T;
    let wanted = match card_type {
        T::Creature | T::Kindred => Creature,
        T::Land => Land,
        T::Artifact => Artifact,
        T::Enchantment => Enchantment,
        T::Instant | T::Sorcery => Spell,
        T::Planeswalker | T::Battle => return false,
    };
    subtype_category(subtype) == Some(wanted)
}

/// CR 205.3m. An unlisted name is not a creature type; debug builds reject it,
/// so a new subtype must be classified before any rule reads it.
pub fn is_creature_type(subtype: &str) -> bool {
    let category = subtype_category(subtype);
    debug_assert!(category.is_some(), "unclassified subtype {subtype:?}");
    category == Some(Creature)
}

/// An object's creature types (CR 205.3m): the ones it lists, or every
/// creature type at once ("all creature types", Soulstone Sanctuary). "All"
/// is one value rather than an enumeration of the creature-type list, so it
/// also covers types no card in the catalog names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CreatureTypes {
    /// Sorted and distinct.
    Listed(Vec<String>),
    All,
}

impl Default for CreatureTypes {
    fn default() -> Self {
        Self::Listed(Vec::new())
    }
}

impl CreatureTypes {
    /// The distinct creature types among `subtypes`.
    pub fn among<'a>(subtypes: impl IntoIterator<Item = &'a String>) -> Self {
        let mut listed: Vec<String> = subtypes
            .into_iter()
            .filter(|subtype| is_creature_type(subtype))
            .cloned()
            .collect();
        listed.sort();
        listed.dedup();
        Self::Listed(listed)
    }

    /// The listed creature types; none for all creature types.
    pub fn listed(&self) -> &[String] {
        match self {
            Self::Listed(listed) => listed,
            Self::All => &[],
        }
    }

    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Listed(listed) if listed.is_empty())
    }

    pub fn contains(&self, creature_type: &str) -> bool {
        match self {
            Self::Listed(listed) => listed.iter().any(|t| t == creature_type),
            Self::All => is_creature_type(creature_type),
        }
    }

    /// Whether the two have a creature type in common ("that share a
    /// creature type"); all creature types shares with any nonempty list.
    pub fn shares(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All, other) | (other, Self::All) => !other.is_empty(),
            (Self::Listed(a), Self::Listed(b)) => a.iter().any(|t| b.contains(t)),
        }
    }

    /// The creature types both have.
    pub fn intersection(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All, other) | (other, Self::All) => other.clone(),
            (Self::Listed(a), Self::Listed(b)) => {
                Self::Listed(a.iter().filter(|t| b.contains(t)).cloned().collect())
            }
        }
    }
}

/// A listed set serializes as its names, so retained records keep their
/// bytes; all creature types is the string `"all"`.
impl serde::Serialize for CreatureTypes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Listed(listed) => listed.serialize(serializer),
            Self::All => serializer.serialize_str("all"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtype_table_is_sorted() {
        assert!(SUBTYPES.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn non_creature_subtypes_are_not_creature_types() {
        for name in ["Cave", "Shrine", "Lesson", "Saga", "Equipment", "Forest"] {
            assert!(!is_creature_type(name), "{name}");
        }
        assert_eq!(subtype_category("Shrine"), Some(Enchantment));
        assert_eq!(subtype_category("Cave"), Some(Land));
        assert!(is_creature_type("Citizen"));
        assert_eq!(subtype_category("Not A Subtype"), None);
    }
}
