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
    ("Desert", Land),
    ("Detective", Creature),
    ("Devil", Creature),
    ("Dinosaur", Creature),
    ("Djinn", Creature),
    ("Dragon", Creature),
    ("Drake", Creature),
    ("Druid", Creature),
    ("Elder", Creature),
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
    ("Gate", Land),
    ("Giant", Creature),
    ("Goat", Creature),
    ("Goblin", Creature),
    ("Gold", Artifact),
    ("Hero", Creature),
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
    ("Mountain", Land),
    ("Mutagen", Artifact),
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
    ("Plant", Creature),
    ("Platypus", Creature),
    ("Porcupine", Creature),
    ("Power-Plant", Land),
    ("Powerstone", Artifact),
    ("Rabbit", Creature),
    ("Ranger", Creature),
    ("Rat", Creature),
    ("Rebel", Creature),
    ("Rogue", Creature),
    ("Role", Enchantment),
    ("Room", Enchantment),
    ("Rune", Enchantment),
    ("Saga", Enchantment),
    ("Scarecrow", Creature),
    ("Scout", Creature),
    ("Seal", Creature),
    ("Serpent", Creature),
    ("Shapeshifter", Creature),
    ("Shard", Enchantment),
    ("Shrine", Enchantment),
    ("Snake", Creature),
    ("Soldier", Creature),
    ("Spacecraft", Artifact),
    ("Sphere", Land),
    ("Spider", Creature),
    ("Spirit", Creature),
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
    ("Vehicle", Artifact),
    ("Wall", Creature),
    ("Warlock", Creature),
    ("Warrior", Creature),
    ("Wizard", Creature),
    ("Wolf", Creature),
    ("Wurm", Creature),
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

/// CR 205.3m. An unlisted name is not a creature type; debug builds reject it,
/// so a new subtype must be classified before any rule reads it.
pub fn is_creature_type(subtype: &str) -> bool {
    let category = subtype_category(subtype);
    debug_assert!(category.is_some(), "unclassified subtype {subtype:?}");
    category == Some(Creature)
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
