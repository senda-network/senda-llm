use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub(crate) enum RuntimeCommand {
    /// Show local model status on a running senda instance.
    Status {
        /// Console/API port of the running senda instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Load a local model into a running senda instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running senda instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local model from a running senda instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running senda instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
}
