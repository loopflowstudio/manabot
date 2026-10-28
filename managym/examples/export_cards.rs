//! Export the typed Rust set definitions for the offline semantic compiler.
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer_pretty(
        io::stdout().lock(),
        &managym::cardsets::authoring::catalog(),
    )?;
    Ok(())
}
