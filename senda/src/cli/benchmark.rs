use clap::{Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub(crate) enum BenchmarkCommand {
    /// Compare MoE ranking sources without launching senda runtime mode.
    #[command(name = "moe-ranking")]
    MoeRanking {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        #[arg(long)]
        model: String,
        /// Number of nodes to compute assignments for.
        #[arg(long, default_value = "2")]
        nodes: usize,
        /// Shared-core overlap factor (1 = no extra redundancy).
        #[arg(long, default_value = "1")]
        overlap: usize,
        /// Minimum experts per node. Defaults to catalog value or 50% fallback.
        #[arg(long)]
        min_experts: Option<u32>,
        /// Ranking sources to compare.
        #[arg(long, value_delimiter = ',', default_value = "analyze")]
        variants: Vec<MoeRankingVariant>,
        /// Optional explicit moe-analyze CSV path.
        #[arg(long)]
        analyze_ranking: Option<PathBuf>,
        /// Optional local JSONL prompt corpus to validate and summarize.
        #[arg(long)]
        prompts: Option<PathBuf>,
        /// Where to write the JSON report. Prints to stdout when omitted.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Import a prompt corpus from a supported online source into local JSONL.
    #[command(name = "import-prompts")]
    ImportPrompts {
        /// Online source to import.
        #[arg(long, value_enum)]
        source: PromptImportSource,
        /// Maximum number of prompts to import.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Optional per-prompt decode budget hint written into the corpus.
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
    },
    /// Benchmark short llama-moe-analyze passes against a full analyze ranking.
    #[command(name = "moe-micro-analyze")]
    MoeMicroAnalyze {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        #[arg(long)]
        model: String,
        /// Minimum experts per node used for recall@N metrics.
        #[arg(long)]
        min_experts: Option<u32>,
        /// Optional explicit full moe-analyze CSV path.
        #[arg(long)]
        analyze_ranking: Option<PathBuf>,
        /// Optional local JSONL prompt corpus used for micro runs.
        #[arg(long)]
        prompts: Option<PathBuf>,
        /// Where to write the JSON report. Prints to stdout when omitted.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare expert grouping strategies using full analyze masses.
    #[command(name = "moe-grouping")]
    MoeGrouping {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        #[arg(long)]
        model: String,
        /// Number of nodes to compute assignments for.
        #[arg(long, default_value = "2")]
        nodes: usize,
        /// Shared-core overlap factor for current senda assignment mode.
        #[arg(long, default_value = "1")]
        overlap: usize,
        /// Minimum experts per node. Defaults to catalog value or 50% fallback.
        #[arg(long)]
        min_experts: Option<u32>,
        /// Optional explicit full moe-analyze CSV path.
        #[arg(long)]
        analyze_ranking: Option<PathBuf>,
        /// Where to write the JSON report. Prints to stdout when omitted.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Capture an auditor *reference* logit fingerprint (verification v1) by
    /// probing a local llama-server, and write it to
    /// `~/.senda/reference-fingerprints.json`.
    #[command(name = "capture-reference")]
    CaptureReference {
        /// Model name as served by the target llama-server (e.g. Qwen3-8B-Q4_K_M).
        #[arg(long)]
        model: String,
        /// Local llama-server HTTP port to probe (e.g. the port from the
        /// running `llama-server` process).
        #[arg(long)]
        port: u16,
    },
    /// Capture an auditor *reference battery* (Layer 1 first-token top-k) by
    /// probing a local llama-server's native `/completion`, and write it to
    /// `~/.senda/reference-batteries.json`.
    #[command(name = "capture-reference-battery")]
    CaptureReferenceBattery {
        /// Model name as served by the target llama-server (e.g. Qwen3-8B-Q4_K_M).
        #[arg(long)]
        model: String,
        /// Local llama-server HTTP port to probe.
        #[arg(long)]
        port: u16,
    },
    /// Run the full offline MoE benchmark suite across several models.
    #[command(name = "moe-model-matrix")]
    MoeModelMatrix {
        /// Model specs: local paths, catalog names, HF exact refs, or HF URLs.
        #[arg(long, required = true)]
        model: Vec<String>,
        /// Number of nodes to compute assignments for.
        #[arg(long, default_value = "2")]
        nodes: usize,
        /// Shared-core overlap factor for current senda assignment mode.
        #[arg(long, default_value = "1")]
        overlap: usize,
        /// Minimum experts per node. Defaults per model to catalog value or 50% fallback.
        #[arg(long)]
        min_experts: Option<u32>,
        /// Optional local JSONL prompt corpus used for micro-analyze runs.
        #[arg(long)]
        prompts: Option<PathBuf>,
        /// Directory containing explicit full moe-analyze CSVs named after model stem.
        #[arg(long)]
        analyze_ranking_dir: Option<PathBuf>,
        /// Where to write the JSON report. Prints to stdout when omitted.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MoeRankingVariant {
    Analyze,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PromptImportSource {
    MtBench,
    Gsm8k,
    Humaneval,
}
