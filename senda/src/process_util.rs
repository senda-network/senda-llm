//! Cross-platform process-spawn helpers.
//!
//! Every `Command::new(...)` inside the senda runtime ends up spawning
//! a console-subsystem child process — `rpc-server.exe`, `llama-server.exe`,
//! `taskkill.exe`, `powershell.exe`, `tar.exe`, `nvidia-smi.exe`, … On
//! Windows, when the parent is itself a console-subsystem app launched
//! interactively, those children inherit the parent's console. Fine. But
//! when the runtime is invoked from a GUI parent — the desktop sidecar
//! (`windows_subsystem = "windows"`) or the user's Scheduled Task with
//! `-LogonType Interactive` — the parent has no console of its own, so
//! `CreateProcess` allocates a *brand-new console window* for every child.
//! Short-lived helpers flash a window; long-lived ones (rpc-server,
//! llama-server) leave a persistent black box on the user's screen for
//! the entire session, multiplied by the number of model shards loaded.
//!
//! Users on Windows reported the cascade as "the app started opening
//! apps and terminals like fucking crazy until it crashed the whole
//! computer". That was literally what was happening — every model load
//! plus every probe spawned its own visible window, every Task Scheduler
//! restart re-stacked them, and Windows Defender scanning every newly
//! created process compounded the load until the box ran out of
//! responsiveness.
//!
//! `CREATE_NO_WINDOW` (0x0800_0000) on `CreateProcess` tells Windows not
//! to allocate a console for the child. Combined with the stdout/stderr
//! redirection we already do in `launch_rpc_server` / `launch_llama_server`,
//! the children run completely headless. No effect on macOS / Linux —
//! the calls are compiled away.
//!
//! Use the [`HideConsole`] extension trait on every spawn site:
//!
//! ```ignore
//! use crate::process_util::HideConsole;
//! Command::new("rpc-server").args(args).hide_console().spawn()
//! ```

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Extension trait that suppresses the spawned child's console window
/// on Windows. Implemented for both `std::process::Command` and
/// `tokio::process::Command` so it can be applied at every spawn site
/// regardless of which flavor that site happens to use.
pub(crate) trait HideConsole {
    fn hide_console(&mut self) -> &mut Self;
}

impl HideConsole for std::process::Command {
    #[cfg(windows)]
    fn hide_console(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        self.creation_flags(CREATE_NO_WINDOW)
    }
    #[cfg(not(windows))]
    fn hide_console(&mut self) -> &mut Self {
        self
    }
}

impl HideConsole for tokio::process::Command {
    #[cfg(windows)]
    fn hide_console(&mut self) -> &mut Self {
        // `tokio::process::Command` re-exports `creation_flags` via its
        // own Windows-only inherent method (no extension trait needed),
        // so we go straight through.
        self.creation_flags(CREATE_NO_WINDOW)
    }
    #[cfg(not(windows))]
    fn hide_console(&mut self) -> &mut Self {
        self
    }
}
