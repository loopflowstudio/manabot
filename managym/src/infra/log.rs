#[derive(Clone, Copy, Debug)]
pub enum LogCat {
    Agent,
    Combat,
    Priority,
    Rules,
    State,
    Turn,
    Test,
}

#[allow(unused_variables)]
pub fn debug(_cat: LogCat, _message: &str) {}

#[allow(unused_variables)]
pub fn info(_cat: LogCat, _message: &str) {}

#[allow(unused_variables)]
pub fn error(_cat: LogCat, _message: &str) {}
