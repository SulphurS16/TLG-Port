//! Settings shared by the tlg runtime and its launcher: where a build keeps
//! its files (`paths`), display, frame-rate and game-file settings
//! (`settings`, settings.txt) and keyboard/mouse bindings (`bindings`,
//! input.txt). Both files are `name = value` lines with `#` comments (`kv`);
//! rewriting a value keeps the other lines as they are.

pub mod bindings;
pub mod carry;
pub mod kv;
pub mod paths;
pub mod saves;
pub mod settings;

/// The release version (workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
