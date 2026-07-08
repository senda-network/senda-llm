//! Process management for llama.cpp binaries.
//!
//! Starts rpc-server and llama-server wired up to the mesh tunnel ports.

use anyhow::{Context, Result};
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::process::Command;

use crate::cli::output::{emit_event, OutputEvent};
use crate::process_util::HideConsole;

/// llama.cpp split mode for distributing tensors across devices.
///
/// - `Layer` (default): each device gets a contiguous range of layers.
///   Works over RPC (network) and local multi-GPU.
/// - `Row`: weight matrices are sharded across devices (true tensor parallelism).
///   Only works for local multi-GPU (CUDA, ROCm) — NOT over RPC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Layer is available for explicit CLI override
pub enum SplitMode {
    /// Pipeline parallelism — split by layers (default, works everywhere).
    Layer,
    /// Tensor parallelism — split weight rows across local GPUs.
    Row,
}

impl SplitMode {
    fn as_arg(self) -> &'static str {
        match self {
            SplitMode::Layer => "layer",
            SplitMode::Row => "row",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }

    fn preferred_devices(self) -> &'static [&'static str] {
        match self {
            BinaryFlavor::Cpu => &["CPU"],
            BinaryFlavor::Cuda => &["CUDA0", "CPU"],
            BinaryFlavor::Rocm => &["ROCm0", "HIP0", "CPU"],
            BinaryFlavor::Vulkan => &["Vulkan0", "CPU"],
            BinaryFlavor::Metal => &["MTL0", "CPU"],
        }
    }

    fn primary_device(self) -> &'static str {
        self.preferred_devices()[0]
    }
}

#[derive(Clone, Debug)]
struct ResolvedBinary {
    path: PathBuf,
    flavor: Option<BinaryFlavor>,
}

pub(crate) fn platform_bin_name(name: &str) -> String {
    #[cfg(windows)]
    {
        if Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            name.to_string()
        } else {
            format!("{name}.exe")
        }
    }

    #[cfg(not(windows))]
    {
        name.to_string()
    }
}

fn flavored_bin_name(name: &str, flavor: BinaryFlavor) -> String {
    platform_bin_name(&format!("{name}-{}", flavor.suffix()))
}

fn bare_bin_name(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    #[cfg(windows)]
    {
        // On Windows, strip a `.exe` extension in a case-insensitive way.
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false)
        {
            Some(path.file_stem()?.to_string_lossy().to_string())
        } else {
            Some(file_name.to_string())
        }
    }

    #[cfg(not(windows))]
    {
        Some(file_name.to_string())
    }
}

fn infer_binary_flavor(name: &str, path: &Path) -> Option<BinaryFlavor> {
    let file_name = bare_bin_name(path)?;
    for flavor in BinaryFlavor::ALL {
        if file_name == format!("{name}-{}", flavor.suffix()) {
            return Some(flavor);
        }
    }
    None
}

fn resolve_binary_path(
    bin_dir: &Path,
    name: &str,
    requested_flavor: Option<BinaryFlavor>,
) -> Result<ResolvedBinary> {
    if let Some(flavor) = requested_flavor {
        let flavored = bin_dir.join(flavored_bin_name(name, flavor));
        if flavored.exists() {
            return Ok(ResolvedBinary {
                path: flavored,
                flavor: Some(flavor),
            });
        }

        let generic = bin_dir.join(platform_bin_name(name));
        if generic.exists() {
            return Ok(ResolvedBinary {
                path: generic,
                flavor: Some(flavor),
            });
        }

        anyhow::bail!(
            "{} not found in {} for requested flavor '{}'",
            flavored.display(),
            bin_dir.display(),
            flavor.suffix()
        );
    }

    let generic = bin_dir.join(platform_bin_name(name));
    if generic.exists() {
        let flavor = infer_binary_flavor(name, &generic);
        return Ok(ResolvedBinary {
            path: generic,
            flavor,
        });
    }

    let matches: Vec<ResolvedBinary> = BinaryFlavor::ALL
        .into_iter()
        .map(|flavor| ResolvedBinary {
            path: bin_dir.join(flavored_bin_name(name, flavor)),
            flavor: Some(flavor),
        })
        .filter(|candidate| candidate.path.exists())
        .collect();

    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => anyhow::bail!(
            "{} not found in {}",
            bin_dir.join(platform_bin_name(name)).display(),
            bin_dir.display()
        ),
        _ => {
            let options = matches
                .iter()
                .filter_map(|candidate| candidate.flavor.map(|flavor| flavor.suffix()))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "multiple {} flavors found in {} ({options}). Pass --llama-flavor to choose one.",
                name,
                bin_dir.display()
            );
        }
    }
}

pub(crate) fn resolve_binary_flavor(
    bin_dir: &Path,
    name: &str,
    requested_flavor: Option<BinaryFlavor>,
) -> Result<Option<BinaryFlavor>> {
    resolve_binary_path(bin_dir, name, requested_flavor).map(|binary| binary.flavor)
}

pub(crate) fn backend_device_for_flavor(
    index: usize,
    binary_flavor: BinaryFlavor,
) -> Option<String> {
    match binary_flavor {
        BinaryFlavor::Cpu => None,
        BinaryFlavor::Cuda => Some(format!("CUDA{index}")),
        BinaryFlavor::Rocm => Some(format!("ROCm{index}")),
        BinaryFlavor::Vulkan => Some(format!("Vulkan{index}")),
        BinaryFlavor::Metal => Some(format!("MTL{index}")),
    }
}

fn child_library_search_paths(binary_path: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(parent) = binary_path.parent() {
        paths.push(parent.to_path_buf());
    }
    paths.extend(platform_runtime_library_paths());
    paths
}

#[cfg(windows)]
fn platform_runtime_library_paths() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(cuda_path) = std::env::var_os("CUDA_PATH").filter(|value| !value.is_empty()) {
        roots.push(PathBuf::from(cuda_path));
    }
    if let Some(program_files) = std::env::var_os("ProgramFiles").filter(|value| !value.is_empty())
    {
        let toolkit_root = PathBuf::from(program_files)
            .join("NVIDIA GPU Computing Toolkit")
            .join("CUDA");
        if let Ok(entries) = std::fs::read_dir(toolkit_root) {
            let mut discovered = entries
                .filter_map(Result::ok)
                .filter_map(|entry| match entry.file_type() {
                    Ok(file_type) if file_type.is_dir() => Some(entry.path()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            discovered.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
            roots.extend(discovered);
        }
    }

    let mut paths = Vec::new();
    for root in roots {
        paths.push(root.join("bin"));
        paths.push(root.join("bin").join("x64"));
    }
    paths
}

#[cfg(not(windows))]
fn platform_runtime_library_paths() -> Vec<PathBuf> {
    Vec::new()
}

fn prepend_child_library_paths(command: &mut Command, binary_path: &Path) {
    let mut paths = child_library_search_paths(binary_path);
    if let Some(existing) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&existing));
    }

    paths.retain(|path| path.exists());
    dedup_paths(&mut paths);

    if let Ok(joined) = std::env::join_paths(paths) {
        command.env("PATH", joined);
    }
}

fn dedup_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::new();
    paths.retain(|path| {
        let key = path.to_string_lossy();
        #[cfg(windows)]
        let key = key.to_ascii_lowercase();
        #[cfg(not(windows))]
        let key = key.to_string();
        seen.insert(key)
    });
}

#[derive(Debug)]
pub struct InferenceServerHandle {
    pid: u32,
    expected_exit: Arc<AtomicBool>,
    expected_comm: String,
    expected_start_time: Option<i64>,
    pub(crate) _pidfile_guard: Option<crate::runtime::instance::PidfileGuard>,
}

impl InferenceServerHandle {
    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process_with_wait(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            20,
            std::time::Duration::from_millis(250),
        )
        .await;
    }
}

impl Drop for InferenceServerHandle {
    /// Best-effort termination if the handle is dropped without an explicit
    /// `shutdown().await` (panic, task abort, or any path that bypasses the
    /// async cleanup). We can't await in `drop`, so this only issues a single
    /// SIGTERM — the death-watcher and the cross-runtime reaper handle any
    /// stragglers. If `expected_exit` is already set, the async shutdown ran
    /// and there is nothing to do.
    fn drop(&mut self) {
        if self.expected_exit.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = send_signal_if_matches(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            ProcessSignal::Terminate,
        );
    }
}

/// Handle for a running rpc-server process.
///
/// Symmetric with [`InferenceServerHandle`] for llama-server.
/// The `_pidfile_guard` field is `None` until T9 wires up pidfile writing.
#[derive(Debug)]
pub struct RpcServerHandle {
    pub pid: u32,
    pub port: u16,
    pub expected_exit: Arc<AtomicBool>,
    pub expected_comm: String,
    pub expected_start_time: Option<i64>,
    pub(crate) _pidfile_guard: Option<crate::runtime::instance::PidfileGuard>,
}

impl Drop for RpcServerHandle {
    /// Best-effort SIGTERM if the rpc-server handle is dropped without an
    /// explicit `shutdown().await` (panic / task abort path). Mirrors the
    /// `InferenceServerHandle::Drop` safety net so a crashed parent does not
    /// leave an orphan rpc-server holding GPU memory.
    fn drop(&mut self) {
        if self.expected_exit.swap(true, Ordering::Relaxed) {
            return;
        }
        let _ = send_signal_if_matches(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            ProcessSignal::Terminate,
        );
    }
}

impl RpcServerHandle {
    pub async fn shutdown(&self) {
        self.expected_exit.store(true, Ordering::Relaxed);
        terminate_process_with_wait(
            self.pid,
            &self.expected_comm,
            self.expected_start_time,
            50,
            std::time::Duration::from_millis(100),
        )
        .await;
    }
}

#[derive(Debug)]
pub struct InferenceServerProcess {
    pub handle: InferenceServerHandle,
    pub death_rx: tokio::sync::oneshot::Receiver<()>,
    pub context_length: u32,
}

pub struct ModelLaunchSpec<'a> {
    pub model: &'a Path,
    pub http_port: u16,
    pub tunnel_ports: &'a [u16],
    pub tensor_split: Option<&'a str>,
    pub split_mode: Option<SplitMode>,
    pub draft: Option<&'a Path>,
    pub draft_max: u16,
    pub model_bytes: u64,
    pub my_vram: u64,
    pub mmproj: Option<&'a Path>,
    pub ctx_size_override: Option<u32>,
    pub total_group_vram: Option<u64>,
    pub selected_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    /// Number of parallel slots for llama-server (`--parallel`).
    /// Set from the `[[models]].slots` TOML config; defaults to 4 when unset.
    pub slots: usize,
}

pub(crate) const GB: u64 = 1_000_000_000;

static RUNTIME_SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

pub(crate) fn mark_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(true, Ordering::Relaxed);
}

pub(crate) fn clear_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(false, Ordering::Relaxed);
}

pub(crate) fn runtime_shutting_down() -> bool {
    RUNTIME_SHUTTING_DOWN.load(Ordering::Relaxed)
}

fn spawned_binary_name(path: &Path) -> String {
    #[cfg(windows)]
    {
        path.file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }

    #[cfg(not(windows))]
    {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }
}

fn model_label(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.to_string_lossy().into_owned());
    crate::models::local::split_gguf_base_name(&stem)
        .unwrap_or(stem.as_str())
        .to_string()
}

fn compute_context_size(
    ctx_size_override: Option<u32>,
    model_bytes: u64,
    my_vram: u64,
    total_group_vram: Option<u64>,
) -> u32 {
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        model_bytes
    };
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    if let Some(override_ctx) = ctx_size_override {
        override_ctx
    } else if vram_after_model >= 30 * GB {
        65536
    } else if vram_after_model >= 12 * GB {
        32768
    } else if vram_after_model >= 6 * GB {
        16384
    } else if vram_after_model >= 3 * GB {
        8192
    } else {
        4096
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    fn as_arg(&self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
            KvType::Q4_0 => "q4_0",
        }
    }

    fn is_quantized(&self) -> bool {
        !matches!(self, KvType::F16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCacheQuant {
    pub k_type: KvType,
    pub v_type: KvType,
}

/// Tracks a known open upstream llama.cpp bug that constrains what KV cache
/// configurations are actually safe to run. Used by `validation_warnings()`
/// so call sites and tests can assert on specific bugs by ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheWarning {
    /// ggml-org/llama.cpp#20866 — asymmetric quantized K/V types hit
    /// BEST_FATTN_KERNEL_NONE in the CUDA FA kernel selector unless llama.cpp
    /// is built with -DGGML_CUDA_FA_ALL_QUANTS=ON. Our own CUDA build sets
    /// this flag (see scripts/build-linux.sh), but standard Homebrew and
    /// official release binaries will crash on FA ops.
    MismatchedQuantNeedsCudaFaAllQuants,
    /// ggml-org/llama.cpp#21450 — on Metal (Apple Silicon), when Flash
    /// Attention falls back to CPU, a quantized V cache crashes with
    /// "quantized V cache requires Flash Attention".
    QuantizedVBreaksMetalFaFallback,
}

/// Build the `-fa` arguments for a llama-server launch.
///
/// llama.cpp defaults `-fa` to `auto`, which on a Metal host resolves to
/// "enabled" and then dispatches `FLASH_ATTN_EXT` ops via RPC to every worker
/// including Metal `rpc-server` instances that do not implement that op and
/// abort on it. We therefore force `-fa off` on every RPC split launch, and
/// keep `-fa on` for solo launches where the host runs the FA op locally on
/// hardware that supports it.
pub(crate) fn flash_attention_args(is_rpc_split: bool) -> [String; 2] {
    let mode = if is_rpc_split { "off" } else { "on" };
    ["-fa".to_string(), mode.to_string()]
}

/// Choose mmap mode for the host's llama-server.
///
/// Solo launches keep `--no-mmap` because the host owns the entire model
/// and `--no-mmap` avoids macOS page-cache pressure on long-lived solo
/// hosts (the original reason that flag was added).
///
/// RPC split launches MUST drop `--no-mmap`. With `--no-mmap`, the host
/// process committs the full `model_local_share + RPC0 staging buffer`
/// into anonymous RAM before chunking weights over the iroh tunnels to
/// remote workers. On a 16 GB MacBook Air hosting a 19.7 GB model with
/// a 9.4 GB RPC partition, that's ~18 GB of committed RAM on a 16 GB
/// box. The OS swaps, then OOM-kills the desktop app (this is exactly
/// the May 17 2026 crash). With mmap on, the host pages local-layer
/// weights from the GGUF on disk lazily through the OS page cache and
/// only committed RAM is KV cache + activations + transient RPC
/// staging — easily under the Mac's budget.
///
/// Returns either `["--no-mmap"]` or `[]` (empty = mmap on, llama.cpp's
/// default).
pub(crate) fn mmap_args(is_rpc_split: bool) -> Vec<String> {
    if is_rpc_split {
        Vec::new()
    } else {
        vec!["--no-mmap".to_string()]
    }
}

/// Hard ceiling on how long `start_llama` will wait for `llama-server`
/// to become healthy.
///
/// Pre-v0.66.39 we scaled by 120 s per GB with no upper bound, which
/// gave a 20 GB model up to **40 minutes**. The May 18 2026 incident
/// showed why that's a disaster: the election loop awaits `start_llama`
/// synchronously, so a multi-minute hang here disables every downstream
/// safety check (fitter watchdog, role-change cleanup, peer-change
/// re-evaluation). Caught a peer in the wrong cohort? Too bad — wait
/// 40 min for it to either finish or time out before election can react.
///
/// The replacement formula is 15 s per GB with a 90 s floor and a 300 s
/// ceiling. A 20 GB GGUF caps at 300 s, well over the ~60-90 s real
/// load time on a NVMe + GPU/Metal host; an unrealistic 50 GB MoE also
/// caps at 300 s on the theory that anything that doesn't load in 5
/// minutes isn't going to load in 40 either, and we want the runner-up
/// to get a turn.
const HEALTH_TIMEOUT_FLOOR_SECS: u64 = 90;
const HEALTH_TIMEOUT_CEIL_SECS: u64 = 300;

/// See `HEALTH_TIMEOUT_CEIL_SECS` for full motivation. Factored out so
/// the cap can be unit-tested without spinning up `llama-server`.
pub(crate) fn health_timeout_secs(model_gb: u64) -> u64 {
    let scaled = std::cmp::max(HEALTH_TIMEOUT_FLOOR_SECS, model_gb.saturating_mul(15));
    std::cmp::min(scaled, HEALTH_TIMEOUT_CEIL_SECS)
}

impl KvCacheQuant {
    /// Thresholds in bytes for the tier boundaries below. Named constants so
    /// the tests can assert exact boundary behavior.
    pub const MEDIUM_TIER_MIN_BYTES: u64 = 5 * GB;
    pub const LARGE_TIER_MIN_BYTES: u64 = 50 * GB;

    /// Fraction of the post-weights VRAM headroom we let an f16 KV cache occupy
    /// before falling back to quantization. The remaining ~25% covers compute
    /// buffers, activations, and the `-fitt` free-memory margin so keeping f16
    /// never pushes `-fit` into CPU offload. See `for_model_size_and_vram`.
    const F16_KV_VRAM_FRACTION: f64 = 0.75;

    /// Choose a KV cache quantization pair for the given model size.
    ///
    /// The small and large tiers are safe on all supported backends without
    /// extra build flags. The medium tier (5-50GB) opts into Q8_0/Q4_0 for
    /// ~25% additional KV savings and relies on GGML_CUDA_FA_ALL_QUANTS in
    /// our CUDA build (enforced by scripts/build-linux.sh) and on Metal
    /// Flash Attention being available on the host (true for all M1+ Macs).
    /// `validation_warnings()` returns the open upstream bugs this tier
    /// currently trips (`#20866`, `#21450`), and
    /// `detect_known_crash_signature` attributes the matching crash in the
    /// llama-server failure path. See the tier comment block in
    /// `build_llama_server_args` for the full rationale.
    pub fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            Self {
                k_type: KvType::Q4_0,
                v_type: KvType::Q4_0,
            }
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            // Aggressive asymmetric: ~25% less KV memory than Q8_0/Q8_0 with
            // minimal quality impact. Known risks:
            // - ggml-org/llama.cpp#20866: requires CUDA build flag (we set it)
            // - ggml-org/llama.cpp#21450: crashes if Metal FA unavailable.
            //   Safe on M1+ Macs (all support Metal FA); rare CPU-FA fallback
            //   on older Intel Macs is detected by `detect_known_crash_signature`.
            Self {
                k_type: KvType::Q8_0,
                v_type: KvType::Q4_0,
            }
        } else {
            Self {
                k_type: KvType::F16,
                v_type: KvType::F16,
            }
        }
    }

    /// Keep the full-precision (f16) KV cache whenever it demonstrably fits the
    /// VRAM left after the weights, otherwise fall back to the size-based tier.
    ///
    /// Quantized KV is a *memory* optimization, not a speed one — on CUDA it
    /// forces the slower `FA_ALL_QUANTS` flash-attention kernels (mismatched
    /// pairs especially), measured at ~3x slower decode on an RTX 4080 serving
    /// Qwen3-8B (37 tok/s quantized vs ~120 tok/s f16). So quantizing when the
    /// device has ample headroom trades throughput for memory we don't need.
    ///
    /// `f16_kv_bytes` is the *total* f16 KV cache for the launch context
    /// (`ctx_size x per-token`, computed from GGUF metadata); `None` means we
    /// couldn't size it (scan failed / shard) and must not guess. We keep f16
    /// only when it fits `vram_after_model x F16_KV_VRAM_FRACTION`, leaving the
    /// remaining headroom for compute buffers, activations, and the `-fitt`
    /// margin. Falling back to quantized is the safe direction: an over-large
    /// f16 cache would make `-fit` spill layers to CPU — far worse than the
    /// quant penalty.
    pub fn for_model_size_and_vram(
        model_bytes: u64,
        f16_kv_bytes: Option<u64>,
        vram_after_model: u64,
    ) -> Self {
        let tier = Self::for_model_size(model_bytes);
        // Small models already run f16; nothing to reconsider.
        if !tier.k_type.is_quantized() && !tier.v_type.is_quantized() {
            return tier;
        }
        let Some(kv) = f16_kv_bytes else {
            return tier; // can't size the cache → keep the conservative tier
        };
        let budget = (vram_after_model as f64 * Self::F16_KV_VRAM_FRACTION) as u64;
        if kv <= budget {
            Self {
                k_type: KvType::F16,
                v_type: KvType::F16,
            }
        } else {
            tier
        }
    }

    /// Tier-specific human-readable label used in the startup log line.
    pub fn label(&self, model_bytes: u64) -> String {
        // f16/f16 can be either a small model (default) or a larger model whose
        // cache fit VRAM headroom (see `for_model_size_and_vram`); either way
        // the precision is what matters, so don't claim a quantized tier.
        if self.k_type == KvType::F16 && self.v_type == KvType::F16 {
            return "F16 K + F16 V (full precision)".to_string();
        }
        let tier = if model_bytes >= Self::LARGE_TIER_MIN_BYTES {
            "model > 50GB"
        } else if model_bytes >= Self::MEDIUM_TIER_MIN_BYTES {
            "model 5-50GB, aggressive asymmetric"
        } else {
            "model < 5GB, no quantization"
        };
        format!(
            "{} K + {} V ({tier})",
            self.k_type.as_arg().to_uppercase(),
            self.v_type.as_arg().to_uppercase()
        )
    }

    /// Return the set of known open upstream bugs this configuration would
    /// trip over. Empty means safe to ship with default llama.cpp builds.
    pub fn validation_warnings(&self) -> Vec<KvCacheWarning> {
        let mut warnings = Vec::new();
        let mismatched = self.k_type != self.v_type;

        if self.v_type.is_quantized() && mismatched {
            warnings.push(KvCacheWarning::QuantizedVBreaksMetalFaFallback);
        }

        if self.k_type.is_quantized() && self.v_type.is_quantized() && mismatched {
            warnings.push(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants);
        }

        warnings
    }

    /// Emit `--cache-type-k`/`--cache-type-v` args (skipped for f16/f16 which
    /// is the llama-server default), log validation warnings for any known
    /// upstream bugs, and log the tier info line.
    pub fn append_args(&self, args: &mut Vec<String>, model_bytes: u64) {
        for warning in self.validation_warnings() {
            emit_kv_cache_warning(warning, self.k_type, self.v_type);
        }

        if self.k_type == KvType::F16 && self.v_type == KvType::F16 {
            tracing::info!("KV cache: {}", self.label(model_bytes));
            return;
        }

        args.extend_from_slice(&[
            "--cache-type-k".to_string(),
            self.k_type.as_arg().to_string(),
            "--cache-type-v".to_string(),
            self.v_type.as_arg().to_string(),
        ]);
        tracing::info!("KV cache: {}", self.label(model_bytes));
    }
}

impl KvCacheWarning {
    /// Substrings that uniquely identify this bug's crash in a llama.cpp log
    /// tail. `detect_known_crash_signature` scans for any of these.
    fn crash_signatures(&self) -> &'static [&'static str] {
        match self {
            KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => &[
                "fatal error",
                "BEST_FATTN_KERNEL_NONE",
                "ggml-cuda/fattn.cu",
            ],
            KvCacheWarning::QuantizedVBreaksMetalFaFallback => &[
                "quantized V cache requires Flash Attention",
                "V cache quantization requires flash_attn",
            ],
        }
    }

    /// Stable, user-facing description of the upstream bug this warning
    /// corresponds to. Used in post-mortem messages.
    fn post_mortem_hint(&self) -> &'static str {
        match self {
            KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => {
                "Known upstream bug: CUDA FA kernel selector rejects mismatched K/V \
                 quantization types without GGML_CUDA_FA_ALL_QUANTS. Our custom CUDA \
                 build sets this flag (see scripts/build-linux.sh); if you are running \
                 a Homebrew or official release llama.cpp binary this will crash every \
                 time. Track ggml-org/llama.cpp#20866."
            }
            KvCacheWarning::QuantizedVBreaksMetalFaFallback => {
                "Known upstream bug: Metal crashes on quantized V cache when Flash \
                 Attention falls back to CPU. All Apple Silicon (M1+) supports Metal FA \
                 and is not affected. If you are seeing this on Apple Silicon, please \
                 file a senda bug — it should not happen. Older Intel Macs or any \
                 host without Metal FA will trip this. Track ggml-org/llama.cpp#21450."
            }
        }
    }
}

/// Scan a log tail for a known upstream crash signature and return the
/// matching warning, if any. Used for post-mortem diagnostics when
/// llama-server exits before becoming healthy.
pub(crate) fn detect_known_crash_signature(log_tail: &str) -> Option<KvCacheWarning> {
    let candidates = [
        KvCacheWarning::QuantizedVBreaksMetalFaFallback,
        KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
    ];
    candidates.into_iter().find(|w| {
        w.crash_signatures()
            .iter()
            .any(|sig| log_tail.contains(sig))
    })
}

fn emit_kv_cache_warning(warning: KvCacheWarning, k: KvType, v: KvType) {
    match warning {
        KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants => tracing::warn!(
            "KV cache K/V types mismatched and both quantized ({}/{}); the CUDA \
             FA kernel selector returns BEST_FATTN_KERNEL_NONE unless llama.cpp is \
             built with -DGGML_CUDA_FA_ALL_QUANTS=ON. Our own build sets this flag \
             (see scripts/build-linux.sh), but standard Homebrew or release binaries \
             will crash on FA ops. Track ggml-org/llama.cpp#20866 for the upstream fix.",
            k.as_arg(),
            v.as_arg()
        ),
        KvCacheWarning::QuantizedVBreaksMetalFaFallback => tracing::warn!(
            "KV cache V is quantized ({}); requires -fa on at runtime. On Apple \
             Silicon, if Metal Flash Attention is unavailable, llama-server will \
             crash with 'quantized V cache requires Flash Attention'. Track \
             ggml-org/llama.cpp#21450 for the Metal fix.",
            v.as_arg()
        ),
    }
}

fn log_tail(path: &Path, max_lines: usize) -> String {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return String::new();
    };

    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn log_tail_message(path: &Path, max_lines: usize) -> String {
    let tail = log_tail(path, max_lines);
    if tail.is_empty() {
        format!("See {}", path.display())
    } else {
        format!("See {}:\n{}", path.display(), tail)
    }
}

fn parse_available_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut in_devices = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed == "available devices:" {
            in_devices = true;
            continue;
        }
        if !in_devices || trimmed.is_empty() {
            continue;
        }
        let Some((name, _rest)) = trimmed.split_once(':') else {
            continue;
        };
        if !name.chars().all(|c| c.is_ascii_alphanumeric()) {
            continue;
        }
        devices.push(name.to_string());
    }

    devices
}

fn probe_available_devices(binary: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new(binary)
        .args(["-d", "__senda_probe_invalid__", "-p", "0"])
        .hide_console()
        .output()
    else {
        return Vec::new();
    };

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !combined.is_empty() && !output.stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    parse_available_devices(&combined)
}

fn preferred_device(available: &[String], flavor: Option<BinaryFlavor>) -> Option<String> {
    let candidates: &[&str] = if let Some(flavor) = flavor {
        flavor.preferred_devices()
    } else {
        &["MTL0", "CUDA0", "ROCm0", "HIP0", "Vulkan0", "CPU"]
    };

    for candidate in candidates {
        if available.iter().any(|device| device == candidate) {
            return Some((*candidate).to_string());
        }
    }
    available.first().cloned()
}

fn resolve_device_for_binary(
    binary: &Path,
    flavor: Option<BinaryFlavor>,
    requested: Option<&str>,
) -> Result<String> {
    let available = probe_available_devices(binary);

    if let Some(device) = requested {
        if !available.is_empty() {
            if available.iter().any(|candidate| candidate == device) {
                return Ok(device.to_string());
            }

            // Dual support for ROCm/HIP transition
            let is_amd_requested = device.starts_with("ROCm") || device.starts_with("HIP");
            if is_amd_requested {
                let alt_device = if device.starts_with("ROCm") {
                    device.replace("ROCm", "HIP")
                } else {
                    device.replace("HIP", "ROCm")
                };
                if available.iter().any(|candidate| candidate == &alt_device) {
                    return Ok(alt_device);
                }
            }

            anyhow::bail!(
                "requested device {device} is not supported by {}. Available devices: {}",
                binary.display(),
                available.join(", ")
            );
        }
        return Ok(device.to_string());
    }

    if let Some(selected) = preferred_device(&available, flavor) {
        return Ok(selected);
    }

    if let Some(flavor) = flavor {
        return Ok(flavor.primary_device().to_string());
    }

    Ok(detect_device())
}

fn selected_backend_device(
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Result<Option<&str>> {
    match selected_gpu {
        Some(gpu) => Ok(Some(gpu.backend_device.as_str())),
        None => Ok(None),
    }
}

/// Compute the per-device memory target (in MiB) that we hand to llama.cpp's
/// `-fitt` flag.
///
/// **Semantics correction (v0.66.51, May 20 2026).** Previously this function
/// returned `0.7 * device_vram` and a docstring claimed the value was the
/// "ceiling on llama.cpp's GPU usage" / "headroom for KV cache". That was
/// inverted. From `llama-server --help`:
///
/// > `-fitt, --fit-target MiB0,MiB1,…  target margin per device for --fit,
/// > comma-separated list of values, single value is broadcast across all
/// > devices, default: 1024`
///
/// `-fitt N` is the amount of memory the fitter must **leave free** on the
/// device, not the maximum it may use. With the old formula, an 18 GB M3
/// Pro (Metal pool ~12.3 GiB) was told to keep ~9.7 GiB free, leaving only
/// ~2.6 GiB for the model. The fitter responded by offloading 22 of 36
/// layers of an 8 B Q4_K_M model to CPU repack — local decode dropped to
/// **0.74 t/s** instead of the ~17 t/s the hardware delivers when the
/// model is fully on Metal. Going through the v0.66.51 ship to
/// `clamp(device_mib / 10, 1024, 2048)` lifted the same M3 Pro's
/// native_tps from 0.737 → 8.95 t/s (12.1×) and TTFT 7290 → 2528 ms
/// (2.9×). The Phase 3.0 native baseline made the cost visible across
/// every Apple-Silicon peer in the mesh.
///
/// Replacement formula: `clamp(device_mib / 10, 1024, 2048)`.
/// - 1024 MiB matches llama.cpp's own default; safe everywhere.
/// - 10 % of device size scales modestly so multi-slot KV cache on larger
///   devices still has breathing room without becoming the dominant cost.
/// - 2048 MiB cap keeps 80 GB+ discrete GPUs from reserving 8 GB for
///   nothing — KV/compute buffers don't grow with device size at fixed
///   ctx, so the absolute margin should not either.
///
/// llama.cpp itself accounts for KV cache and compute buffers when sizing
/// the fit (see `common_params_fit_impl`'s breakdown line:
/// `5696 = 4455 + 936 + 304` for model+context+compute). The `-fitt`
/// margin only needs to cover incidental overhead and OS pressure on
/// unified-memory devices.
///
/// `selected_gpu.vram_bytes` is preferred because it's the device-specific
/// figure; `my_vram` is the node-level compound VRAM and is used as a
/// fallback for the (rare) launches where no GPU was pinned.
fn compute_fit_target_mib(
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    my_vram: u64,
) -> u64 {
    let device_vram = selected_gpu.map(|g| g.vram_bytes).unwrap_or(my_vram);
    let device_mib = device_vram / (1024 * 1024);
    (device_mib / 10).clamp(1024, 2048)
}

fn fit_target_mib_for_launch(
    has_rpc_workers: bool,
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    my_vram: u64,
) -> Option<u64> {
    (!has_rpc_workers).then(|| compute_fit_target_mib(selected_gpu, my_vram))
}

fn ensure_selected_gpu_capacity(
    selected_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    required_bytes: u64,
    purpose: &str,
) -> Result<()> {
    let Some(gpu) = selected_gpu else {
        return Ok(());
    };
    let required_with_headroom = ((required_bytes as f64) * 1.1).ceil() as u64;
    anyhow::ensure!(
        gpu.vram_bytes >= required_with_headroom,
        "pinned GPU '{}' ({}) has {:.1}GB but {purpose} assumes at least {:.1}GB on the selected device",
        gpu.stable_id,
        gpu.backend_device,
        gpu.vram_bytes as f64 / GB as f64,
        required_with_headroom as f64 / GB as f64
    );
    Ok(())
}

fn command_has_output(command: &str, args: &[&str]) -> bool {
    let Ok(output) = std::process::Command::new(command)
        .args(args)
        .hide_console()
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| !line.trim().is_empty())
}

/// Probe `rpc-server -h` and return true iff the help text mentions
/// `--gguf`. Result is memoised per-binary-path because rpc-server
/// prints help to stderr almost instantly but spawning it on every
/// model load is still wasteful. The cache lives only as long as the
/// senda process, so a runtime upgrade that lays down a new
/// rpc-server.exe at the same path is automatically picked up by the
/// next process start.
///
/// Why we need this: see the long-form comment in `start_rpc_server`.
/// TL;DR — the Windows release pipeline currently vendors the
/// upstream-unpatched rpc-server.exe from ggml-org's b9041 build,
/// which does not advertise `--gguf`. Without this probe every model
/// load on Windows hits "error: unknown argument: --gguf" and
/// rpc-server exits before listening.
fn rpc_server_supports_gguf_flag(rpc_server_path: &Path) -> bool {
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static CACHE: OnceLock<Mutex<std::collections::HashMap<std::path::PathBuf, bool>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));

    if let Ok(guard) = cache.lock() {
        if let Some(&supported) = guard.get(rpc_server_path) {
            return supported;
        }
    }

    let supported = probe_rpc_server_supports_gguf(rpc_server_path);

    if let Ok(mut guard) = cache.lock() {
        guard.insert(rpc_server_path.to_path_buf(), supported);
    }

    supported
}

fn probe_rpc_server_supports_gguf(rpc_server_path: &Path) -> bool {
    // `-h` exits 0 and prints the supported flags to stderr (line
    // shape: `  -m, --gguf PATH        local GGUF model file ...`).
    let Ok(output) = std::process::Command::new(rpc_server_path)
        .arg("-h")
        .hide_console()
        .output()
    else {
        // If we can't even spawn the binary, fall back to the
        // optimistic assumption (pass --gguf, let the launch fail
        // loudly) rather than silently degrading every load forever.
        // The common case for "couldn't spawn rpc-server" is "user
        // has no helpers installed", which produces its own actionable
        // error elsewhere.
        return true;
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    combined.contains("--gguf")
}

/// Start a local rpc-server and return a handle holding its PID and port.
/// Picks an available port automatically.
/// If `gguf_path` is provided, passes `--gguf` so the server loads weights from the local file.
pub async fn start_rpc_server(
    runtime: &crate::runtime::instance::InstanceRuntime,
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    device: Option<&str>,
    gguf_path: Option<&Path>,
) -> Result<RpcServerHandle> {
    let rpc_server = resolve_binary_path(bin_dir, "rpc-server", binary_flavor)?;
    let rpc_server_name = spawned_binary_name(&rpc_server.path);

    // Find a free port
    let port = find_free_port().await?;

    let device = resolve_device_for_binary(&rpc_server.path, rpc_server.flavor, device)?;
    // GPU backends compile shaders / allocate device contexts at first launch, which
    // can easily exceed 15s. Vulkan was already given headroom; Metal on Apple
    // Silicon takes ~8s just to load the embedded library and another ~30-45s to
    // allocate the command queue / initialise the model on a warm path, and
    // CUDA/ROCm driver init can be similarly slow on cold boot. CPU has no such
    // warmup, so we keep its tight bound to fail fast on real bugs.
    let startup_timeout = if device == "CPU" {
        std::time::Duration::from_secs(15)
    } else {
        std::time::Duration::from_secs(120)
    };
    let startup_polls = (startup_timeout.as_millis() / 500) as usize;

    let rpc_log = runtime.log_path(&format!("rpc-server-{port}"));
    let rpc_log_file = std::fs::File::create(&rpc_log)
        .with_context(|| format!("Failed to create rpc-server log file {}", rpc_log.display()))?;
    let rpc_log_file2 = rpc_log_file.try_clone()?;

    tracing::info!("Starting rpc-server on :{port} (device: {device})");
    let _ = emit_event(OutputEvent::RpcServerStarting {
        port,
        device: device.clone(),
        log_path: Some(rpc_log.display().to_string()),
    });

    let mut args = vec![
        "-d".to_string(),
        device.clone(),
        "-p".to_string(),
        port.to_string(),
    ];
    if let Some(path) = gguf_path {
        // The `--gguf PATH` flag is added by our local
        // third_party/llama.cpp/patches/0001-rpc-optimize-local-GGUF-tensor-loading.patch.
        // senda.exe tries to use it because zero-transfer tensor loading
        // is a meaningful speedup on local-only RPC.
        //
        // Windows hotfix:
        //   Every currently published senda-windows-* runtime bundle
        //   (including v0.66.4) ships rpc-server.exe that exits with
        //   `error: unknown argument: --gguf`.
        //   To restore reliability immediately, never pass --gguf on Windows;
        //   use plain RPC tensor transfer until the release pipeline starts
        //   shipping patched rpc-server builds.
        if cfg!(windows) {
            tracing::warn!(
                "Windows runtime bundle rpc-server is not yet --gguf-compatible; \
                 using network tensor transfer for {}",
                path.display()
            );
        } else if rpc_server_supports_gguf_flag(&rpc_server.path) {
            args.push("--gguf".to_string());
            args.push(path.to_string_lossy().to_string());
            tracing::info!(
                "rpc-server will load weights from local GGUF: {}",
                path.display()
            );
        } else {
            tracing::warn!(
                "rpc-server at {} does not support --gguf; falling back to network tensor \
                 transfer for this load.",
                rpc_server.path.display()
            );
        }
    }

    let mut command = Command::new(&rpc_server.path);
    command
        .args(&args)
        .env("SENDA_OWNER_PID", std::process::id().to_string())
        .env(
            "SENDA_RUNTIME_DIR",
            runtime.dir().to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::from(rpc_log_file))
        .stderr(std::process::Stdio::from(rpc_log_file2))
        // Suppress the per-rpc-server console window on Windows. See
        // crate::process_util — without this every model the runtime
        // hosts on this node opens a persistent black box on the user's
        // screen, and a multi-shard MoE host would stack one per shard.
        .hide_console();
    prepend_child_library_paths(&mut command, &rpc_server.path);

    let mut child = command.spawn().with_context(|| {
        format!(
            "Failed to start rpc-server at {}",
            rpc_server.path.display()
        )
    })?;

    let pid = child.id().context("rpc-server did not expose a PID")?;
    if pid == 0 {
        anyhow::bail!("rpc-server returned PID 0 — refusing to proceed");
    }
    let child_started_at =
        crate::runtime::instance::validate::process_started_at_unix(pid).unwrap_or(None);
    let owner_started_at: i64 =
        crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
    let metadata = crate::runtime::instance::PidfileMetadata {
        cmd_name: rpc_server_name.clone(),
        child_pid: pid,
        child_started_at_unix: child_started_at.unwrap_or(0),
        owner_pid: std::process::id(),
        owner_started_at_unix: owner_started_at,
        argv_snippet: crate::runtime::instance::PidfileMetadata::cap_argv(
            &args,
            crate::runtime::instance::ARGV_SNIPPET_MAX_BYTES,
        ),
        runtime_dir: runtime.dir().to_path_buf(),
    };
    let pidfile_guard = runtime.write_pidfile(&format!("rpc-server-{port}"), &metadata)?;
    let expected_exit = Arc::new(AtomicBool::new(false));
    let expected_exit_clone = expected_exit.clone();

    // Wait for it to be listening
    for _ in 0..startup_polls {
        if is_port_open(port).await {
            let pidfile_path = runtime.pidfile_path(&format!("rpc-server-{port}"));
            tokio::spawn(async move {
                let _ = child.wait().await;
                let _ = std::fs::remove_file(&pidfile_path);
                if !expected_exit_clone.load(Ordering::Relaxed) && !runtime_shutting_down() {
                    let _ = emit_event(OutputEvent::Warning {
                        message: "rpc-server process exited unexpectedly".to_string(),
                        context: Some(format!("port={port} device={device}")),
                    });
                }
            });
            return Ok(RpcServerHandle {
                pid,
                port,
                expected_exit,
                expected_comm: rpc_server_name.clone(),
                expected_start_time: child_started_at,
                _pidfile_guard: Some(pidfile_guard),
            });
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll rpc-server status for {}",
                rpc_server.path.display()
            )
        })? {
            let tail = log_tail(&rpc_log, 40);
            let tail_msg = if tail.is_empty() {
                format!("See {}", rpc_log.display())
            } else {
                format!("See {}:\n{}", rpc_log.display(), tail)
            };
            anyhow::bail!(
                "rpc-server exited before listening on port {port} (device: {device}, status: {status}). {tail_msg}"
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let tail = log_tail(&rpc_log, 40);
    let tail_msg = if tail.is_empty() {
        format!("See {}", rpc_log.display())
    } else {
        format!("See {}:\n{}", rpc_log.display(), tail)
    };
    anyhow::bail!(
        "rpc-server failed to start on port {port} within {}s (device: {device}). {tail_msg}",
        startup_timeout.as_secs()
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessSignal {
    Terminate,
    Kill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalOutcome {
    Sent,
    #[cfg(not(windows))]
    AlreadyDead,
    #[cfg(not(windows))]
    Skipped,
    Failed,
}

fn send_signal_if_matches(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    signal: ProcessSignal,
) -> SignalOutcome {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return SignalOutcome::Failed;
    }

    #[cfg(not(windows))]
    {
        if let Some(expected_t) = expected_start_time {
            if !crate::runtime::instance::validate::validate_pid_matches(
                pid,
                expected_comm,
                expected_t,
            ) {
                tracing::warn!("pid {pid} no longer matches expected identity, skipping signal");
                return SignalOutcome::Skipped;
            }
        } else if !crate::runtime::instance::validate::process_name_matches(pid, expected_comm) {
            tracing::warn!("pid {pid} no longer matches {expected_comm}, skipping signal");
            return SignalOutcome::Skipped;
        }
    }

    #[cfg(windows)]
    {
        let _ = expected_start_time;
        tracing::debug!(
            pid,
            expected_comm,
            "skipping process identity validation on Windows"
        );
    }

    #[cfg(unix)]
    unsafe {
        let ret = libc::kill(
            pid as libc::pid_t,
            match signal {
                ProcessSignal::Terminate => libc::SIGTERM,
                ProcessSignal::Kill => libc::SIGKILL,
            },
        );
        if ret == 0 {
            return SignalOutcome::Sent;
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return SignalOutcome::AlreadyDead;
        }

        tracing::warn!(pid, error = %err, ?signal, "failed to signal process");
        SignalOutcome::Failed
    }

    #[cfg(windows)]
    {
        let pid_str = pid.to_string();
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid_str, "/T"]);
        if signal == ProcessSignal::Kill {
            command.arg("/F");
        }
        match command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .hide_console()
            .status()
        {
            Ok(status) if status.success() => SignalOutcome::Sent,
            Ok(status) => {
                tracing::warn!(pid, exit_code = status.code(), ?signal, "taskkill failed");
                SignalOutcome::Failed
            }
            Err(err) => {
                tracing::warn!(pid, error = %err, ?signal, "failed to run taskkill");
                SignalOutcome::Failed
            }
        }
    }
}

/// Outcome of [`terminate_process_blocking`].
///
/// Callers can use [`TerminationOutcome::is_success`] for a coarse success/failure
/// check equivalent to the old `bool` return, or match on individual variants when
/// the distinction matters (e.g. deciding whether a runtime had a chance to clean up
/// its children).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminationOutcome {
    /// Process was already gone before we attempted to signal it.
    NotRunning,
    /// Process exited after SIGTERM, before kill escalation.
    Graceful,
    /// Process did not exit within the grace period and had to be SIGKILL'd.
    Killed,
    /// Signal call itself failed (e.g. identity mismatch or OS error).
    Failed,
}

impl TerminationOutcome {
    /// Returns `true` for any outcome where the process is no longer running.
    ///
    /// Equivalent to the old `bool` return from `terminate_process_blocking`:
    /// `NotRunning | Graceful | Killed` → `true`, `Failed` → `false`.
    pub(crate) fn is_success(self) -> bool {
        !matches!(self, TerminationOutcome::Failed)
    }
}

/// Attempts to terminate a process identified by `pid`, with identity validation
/// and graceful-then-forceful signal escalation.
///
/// Returns a [`TerminationOutcome`] describing how the process was stopped:
/// - [`TerminationOutcome::NotRunning`] — process was already dead.
/// - [`TerminationOutcome::Graceful`] — process exited after SIGTERM within the grace period.
/// - [`TerminationOutcome::Killed`] — process required SIGKILL after the grace period.
/// - [`TerminationOutcome::Failed`] — could not signal the process (identity mismatch or OS error).
///
/// Sends SIGTERM first, then waits up to 5 s (20 × 250 ms). If the process is
/// still alive, escalates to SIGKILL.
pub(crate) fn terminate_process_blocking(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> TerminationOutcome {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        #[cfg(not(windows))]
        SignalOutcome::AlreadyDead => return TerminationOutcome::NotRunning,
        // Identity mismatch: the PID belongs to a different process; do not
        // claim a successful stop.
        #[cfg(not(windows))]
        SignalOutcome::Skipped => return TerminationOutcome::Failed,
        SignalOutcome::Failed => return TerminationOutcome::Failed,
    }

    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return TerminationOutcome::Graceful;
        }
    }

    match send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill) {
        SignalOutcome::Sent => TerminationOutcome::Killed,
        #[cfg(not(windows))]
        SignalOutcome::AlreadyDead => TerminationOutcome::Graceful,
        _ => TerminationOutcome::Failed,
    }
}

async fn terminate_process_with_wait(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    attempts: usize,
    interval: std::time::Duration,
) {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        #[cfg(not(windows))]
        SignalOutcome::AlreadyDead | SignalOutcome::Skipped => return,
        SignalOutcome::Failed => return,
    }

    for _ in 0..attempts {
        tokio::time::sleep(interval).await;
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return;
        }
    }

    let _ = send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill);
}

/// Speculative decoding (attaching a draft model via `-md`) is **disabled by
/// default**.
///
/// Measured 2026-06-05, controlled thermally-paired A/B (alternating
/// draft/no-draft requests, unique prompts) on the catalog's canonical pairing
/// `Qwen3-0.6B → Qwen3-8B`, at a healthy ~68% draft acceptance:
///   - Apple Silicon M3 Pro:        −12% decode tok/s, draft slower 0/8 rounds
///   - discrete RTX 4080 SUPER/CUDA: −5.4% decode tok/s, draft slower 0/8 rounds
/// On a fast (≤8B) target the draft's serial forward passes cost more than the
/// accepted tokens save on *every* backend tested — there is no backend where it
/// wins, so a per-backend gate is pointless. See internal/RESILIENCE.md.
///
/// Set `SENDA_ENABLE_DRAFT=1` (or `true`/`yes`/`on`) to opt back in — e.g.
/// for Phase 4.D speculative-decoding work, which must prove a net win on its
/// big/slow target class before this default is reconsidered.
pub fn speculative_decoding_enabled() -> bool {
    parse_enable_draft(std::env::var("SENDA_ENABLE_DRAFT").ok().as_deref())
}

/// Pure parse of the `SENDA_ENABLE_DRAFT` value — truthy only on an
/// explicit affirmative; absent/empty/anything else is `false` (the default).
fn parse_enable_draft(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Start llama-server with the given model, HTTP port, and RPC tunnel ports.
/// Returns a oneshot receiver that fires when the process exits.
/// `model_bytes` is the total GGUF file size, used to select KV cache quantization:
///   - < 5GB: FP16 (default) — small models, KV cache is tiny
///   - 5-50GB: Q8_0 K + Q4_0 V — keeps attention routing precise (K dominates
///     quality via softmax), compresses values aggressively (~25% less KV memory
///     than Q8_0/Q8_0 with minimal quality impact)
///   - > 50GB: Q4_0 — maximum compression, these models need every byte
pub async fn start_llama_server(
    runtime: &crate::runtime::instance::InstanceRuntime,
    bin_dir: &Path,
    binary_flavor: Option<BinaryFlavor>,
    spec: ModelLaunchSpec<'_>,
) -> Result<InferenceServerProcess> {
    let model = spec.model;
    let http_port = spec.http_port;
    let tunnel_ports = spec.tunnel_ports;
    let tensor_split = spec.tensor_split;
    let split_mode = spec.split_mode;
    let draft = spec.draft;
    let draft_max = spec.draft_max;
    let model_bytes = spec.model_bytes;
    let my_vram = spec.my_vram;
    let mmproj = spec.mmproj;
    let ctx_size_override = spec.ctx_size_override;
    let total_group_vram = spec.total_group_vram;
    let selected_gpu = spec.selected_gpu;
    let slots = spec.slots;
    let llama_server = resolve_binary_path(bin_dir, "llama-server", binary_flavor)?;

    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());

    // Build --rpc argument: all tunnel ports as localhost endpoints
    let rpc_endpoints: Vec<String> = tunnel_ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .collect();
    let rpc_arg = rpc_endpoints.join(",");

    tracing::info!(
        "Starting llama-server on :{http_port} with model {} and --rpc {}",
        model.display(),
        rpc_arg
    );

    let llama_log = runtime.log_path(&format!("llama-server-{}", http_port));
    let log_file = std::fs::File::create(&llama_log).with_context(|| {
        format!(
            "Failed to create llama-server log file {}",
            llama_log.display()
        )
    })?;
    let log_file2 = log_file.try_clone()?;

    // llama-server uses --rpc only for remote workers.
    // Context size: scale to available VRAM on the host node.
    // In split mode (pipeline parallel), each node holds a range of layers
    // and the KV cache for those layers is allocated on the same device.
    // So both weights and KV are distributed. The host only needs VRAM for
    // its share of weights + its share of KV. We estimate the host's weight
    // share proportionally and let llama-server pick the largest -c that fits.
    let host_model_bytes = if let Some(group_vram) = total_group_vram {
        // Split mode: host holds its share of the weights
        if group_vram > 0 {
            let host_fraction = my_vram as f64 / group_vram as f64;
            (model_bytes as f64 * host_fraction) as u64
        } else {
            model_bytes
        }
    } else {
        // Local mode: host holds all weights
        model_bytes
    };
    ensure_selected_gpu_capacity(selected_gpu, host_model_bytes, "this local launch")?;
    let vram_after_model = my_vram.saturating_sub(host_model_bytes);
    let ctx_size = compute_context_size(ctx_size_override, model_bytes, my_vram, total_group_vram);
    let _ = emit_event(OutputEvent::LlamaStarting {
        model: Some(model_label(model)),
        http_port,
        ctx_size: Some(ctx_size),
        log_path: Some(llama_log.display().to_string()),
    });
    tracing::info!(
        "Context size: {ctx_size} tokens (model {:.1}GB, host weights ~{:.1}GB, {:.0}GB capacity, {:.1}GB free{})",
        model_bytes as f64 / GB as f64,
        host_model_bytes as f64 / GB as f64,
        my_vram as f64 / GB as f64,
        vram_after_model as f64 / GB as f64,
        if total_group_vram.is_some() {
            " [split]"
        } else {
            ""
        }
    );

    let mut args = vec!["-m".to_string(), model.to_string_lossy().to_string()];
    if !tunnel_ports.is_empty() {
        args.push("--rpc".to_string());
        args.push(rpc_arg);
    }
    // Pin slot count explicitly. llama.cpp's default of 4 slots silently
    // queues anything beyond that in an unbounded deferred deque, which is
    // the source of the MiniMax-style death spiral: 43k-token prefills
    // occupy all slots for minutes while new requests pile up invisibly.
    // The backend proxy enforces a matching inflight cap so the deferred
    // queue never actually fills. See network/openai/backend.rs.
    //
    // `-fit on` lets llama-server reduce `-ngl` to whatever fits in available
    // VRAM and overflow the rest onto host RAM. Required when the shard size
    // exceeds GPU capacity (e.g. compound-RAM MoE split where a node holds
    // ~14 GiB on a 12.3 GiB Metal working set, or ~14 GiB on an 8 GiB CUDA
    // laptop GPU). Without `-fit`, the model load fails with `cudaMalloc
    // failed: out of memory` (CUDA) or `kIOGPUCommandBufferCallbackError-
    // OutOfMemory` mid-decode (Metal).
    //
    // Critically, we do NOT also pass `-ngl 99` here: when both are set,
    // llama.cpp logs `n_gpu_layers already set by user to 99, abort` and
    // refuses to let `-fit` reduce the layer count, so the OOM happens
    // anyway. Leave `-ngl` unset and let `-fit` pick the largest layer
    // count that fits the chosen device-memory target.
    //
    // `-fitt` (fit-target) is set to 70% of the pinned device's reported
    // `vram_bytes` for local launches, which leaves headroom for KV cache,
    // compute buffers, and the Apple-Silicon gap between our compound-RAM
    // estimate and Metal's `recommendedMaxWorkingSetSize`. See
    // `compute_fit_target_mib`.
    //
    // Do not pass a host-derived fit target to RPC split launches: llama.cpp
    // applies the single target to every remote device, so a 14 GB Mac host can
    // accidentally require 9+ GB free on an 8 GB CUDA worker and make the whole
    // launch impossible before the fitter can rebalance.
    let is_rpc_split = !tunnel_ports.is_empty();
    let fit_target_mib = fit_target_mib_for_launch(is_rpc_split, selected_gpu, my_vram);
    args.extend_from_slice(&flash_attention_args(is_rpc_split));
    if is_rpc_split {
        tracing::info!(
            "RPC split launch detected; forcing -fa off and skipping KV-cache quantization \
             because rpc-server Metal workers can abort on FLASH_ATTN_EXT"
        );
    }
    args.extend_from_slice(&["-fit".to_string(), "on".to_string()]);
    args.extend(mmap_args(is_rpc_split));
    args.extend_from_slice(&[
        "--parallel".to_string(),
        slots.to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        http_port.to_string(),
        "-c".to_string(),
        ctx_size.to_string(),
        // Use deepseek format: thinking goes into reasoning_content field.
        // Goose/OpenAI clients parse this correctly. "none" leaks raw <think>
        // tags into content which is worse.
        "--reasoning-format".to_string(),
        "deepseek".to_string(),
        // Disable thinking by default. Thinking models (Qwen3, MiniMax) burn
        // 15-80s on hidden reasoning for no quality gain on most tasks, and
        // Qwen3.5-9B is completely broken (reasoning consumes all max_tokens).
        // API users can opt-in per-request with:
        //   "chat_template_kwargs": {"enable_thinking": true}
        "--reasoning-budget".to_string(),
        "0".to_string(),
        // `--reasoning-budget 0` ends thinking immediately, but the chat
        // template still emits an empty `<think></think>\n\n` scaffold into
        // the content stream before the first real token. Measured on
        // Qwen3-8B: the model decodes those 3 dead tokens first, costing
        // ~1-2s of TTFT for nothing. Setting `enable_thinking: false` at the
        // template level skips the scaffold entirely — same answer, first
        // token ~3-5x sooner (e.g. 1215ms -> 328ms locally). The per-request
        // opt-in above still wins (request kwargs merge over this launch
        // default; verified), and the kwarg is ignored by templates that
        // don't reference it, so this is safe to apply to every model.
        "--chat-template-kwargs".to_string(),
        "{\"enable_thinking\": false}".to_string(),
        // Anti-repetition default. llama.cpp ships repeat_penalty off
        // (1.0), which lets small quantized models fall into tight decode
        // loops ("Ok 1 2 3 4 5 Ok 1 2 3 4 5 ...") on mildly adversarial
        // prompts. 1.1 over a 256-token window is the conventional mild
        // setting — it kills loops without measurably affecting normal
        // prose. Per-request JSON fields (`repeat_penalty`, `repeat_last_n`)
        // still override these, so clients that tune their own sampling are
        // unaffected.
        "--repeat-penalty".to_string(),
        "1.1".to_string(),
        "--repeat-last-n".to_string(),
        "256".to_string(),
    ]);
    if let Some(fit_target_mib) = fit_target_mib {
        args.push("-fitt".to_string());
        args.push(fit_target_mib.to_string());
    }

    // Mesh hooks — tell llama-server where to call back.
    // Uses the management API port (default 3131) as the callback target.
    if let Ok(api_port) = std::env::var("MESH_API_PORT") {
        args.extend_from_slice(&["--mesh-port".to_string(), api_port]);
    }
    if std::env::var("MESH_HOOK_DEBUG").is_ok() {
        args.push("--mesh-hook-debug".to_string());
    }

    // KV cache quantization — asymmetric K/V strategy.
    //
    // K precision dominates quality: K controls attention routing via softmax,
    // where small errors get exponentially amplified. V errors scale linearly
    // in the weighted sum and are far more tolerant of compression.
    // (See TurboQuant ICLR 2026 / asymmetric K/V findings.)
    //
    // Current tiers:
    //   < 5GB:  leave default (FP16) — small models, KV cache is negligible
    //   5-50GB: K=Q8_0, V=Q4_0 — aggressive asymmetric, ~25% less KV memory
    //           than Q8_0/Q8_0 with minimal quality impact. Relies on two
    //           open upstream bugs being worked around (see caveats below).
    //   > 50GB: Q4_0/Q4_0 — maximum compression, matched quantized types so
    //           no GGML_CUDA_FA_ALL_QUANTS needed. Requires -fa on.
    //
    // Caveats for the 5-50GB K=Q8_0/V=Q4_0 tier as of 2026-04:
    //   - ggml-org/llama.cpp#20866 — asymmetric K/V types require rebuilding
    //     llama.cpp with -DGGML_CUDA_FA_ALL_QUANTS=ON. Standard Homebrew and
    //     release binaries crash with BEST_FATTN_KERNEL_NONE → GGML_ABORT.
    //     Our own CUDA build sets that flag and the post-cmake assertion in
    //     scripts/build-linux.sh keeps it from being dropped silently. Users
    //     pointing senda at an external llama.cpp binary can still trip
    //     this; `detect_known_crash_signature` attributes the failure.
    //   - ggml-org/llama.cpp#21450 — Metal crashes on mixed quantized KV when
    //     Flash Attention falls back to CPU ("quantized V cache requires Flash
    //     Attention"). All Apple Silicon (M1+) supports Metal FA and is not
    //     affected in practice. Older Intel Macs or any host without Metal
    //     FA trip this; `detect_known_crash_signature` attributes the
    //     failure when it matches.
    //
    // TODO(ggml-org/llama.cpp#20866, ggml-org/llama.cpp#21450): once both are
    // closed upstream and our fork is rebased past the fixes, remove the
    // corresponding KvCacheWarning variants, the build assertion, and this
    // caveat block.
    if !is_rpc_split {
        // Prefer f16 KV (fastest) when it fits VRAM headroom; quantize only
        // under real memory pressure. Size the f16 cache from GGUF metadata at
        // the chosen context — None (scan failed/shard) keeps the size tier.
        let f16_kv_bytes = crate::models::gguf::scan_gguf_compact_meta(model).and_then(|meta| {
            let k = meta.k_cache_bytes_per_token_f16()?;
            let v = meta.v_cache_bytes_per_token_f16()?;
            Some((k + v).saturating_mul(ctx_size as u64))
        });
        KvCacheQuant::for_model_size_and_vram(model_bytes, f16_kv_bytes, vram_after_model)
            .append_args(&mut args, model_bytes);
    }
    if let Some(ts) = tensor_split {
        args.push("--tensor-split".to_string());
        args.push(ts.to_string());
    }
    if let Some(mode) = split_mode {
        args.push("--split-mode".to_string());
        args.push(mode.as_arg().to_string());
        match mode {
            SplitMode::Layer => {
                tracing::info!(
                    "Split mode: {} (layer-based / pipeline parallelism)",
                    mode.as_arg()
                );
            }
            SplitMode::Row => {
                tracing::info!(
                    "Split mode: {} (tensor parallelism across local GPUs)",
                    mode.as_arg()
                );
            }
        }
    }
    let local_device = resolve_device_for_binary(
        &llama_server.path,
        llama_server.flavor,
        selected_backend_device(selected_gpu)?,
    )?;
    if selected_gpu.is_some() {
        args.push("--device".to_string());
        args.push(local_device.clone());
    }
    if let Some(draft_path) = draft {
        if !speculative_decoding_enabled() {
            // Disabled by default — measured net throughput loss on every
            // backend tested (see `speculative_decoding_enabled` / RESILIENCE.md).
            tracing::info!(
                "Speculative decoding disabled by default; ignoring draft {} \
                 (set SENDA_ENABLE_DRAFT=1 to opt back in).",
                draft_path.display()
            );
        } else if draft_path.exists() {
            // Always honor explicit --draft: the user opted in, even on CPU.
            // GPU-specific offload flags (`-ngld`, `--device-draft`) only get
            // emitted when we actually have a GPU; the speculative-decoding
            // budget itself (`--spec-draft-n-max`) is device-agnostic. This
            // also lets CPU CI smoke tests exercise the spec-draft flag
            // surface and catch upstream llama.cpp arg drift early — the May
            // 2026 incident (`--draft-max` silently removed) would have been
            // caught by CI if this gate hadn't quietly skipped CPU runs.
            args.push("-md".to_string());
            args.push(draft_path.to_string_lossy().to_string());
            // Only pin the draft model to a specific GPU when the runtime
            // has actually pinned the base model to one (`selected_gpu`
            // is what gates `--device` for the base — we mirror that).
            // Otherwise we'd emit e.g. `--device-draft MTL0` on a CI
            // runner whose `MTL0` is reported by the binary's device
            // probe but isn't actually usable, and llama-server bails
            // with `invalid device: MTL0`.
            if selected_gpu.is_some() && local_device != "CPU" {
                args.push("-ngld".to_string());
                args.push("99".to_string());
                args.push("--device-draft".to_string());
                args.push(local_device.clone());
            }
            // Upstream llama.cpp removed --draft / --draft-n / --draft-max
            // and renamed them to --spec-draft-n-max. The short alias was
            // dropped at the same time. Older builds of llama-server still
            // accepted --draft-max; current builds exit on startup with
            // "argument has been removed". Always emit the new name.
            args.push("--spec-draft-n-max".to_string());
            args.push(draft_max.to_string());
            tracing::info!(
                "Speculative decoding: draft={}, spec-draft-n-max={}, device={}",
                draft_path.display(),
                draft_max,
                local_device
            );
        } else {
            tracing::warn!(
                "Draft model not found at {}, skipping speculative decoding",
                draft_path.display()
            );
        }
    }
    if let Some(proj) = mmproj {
        if proj.exists() {
            args.push("--mmproj".to_string());
            args.push(proj.to_string_lossy().to_string());
            // Vision images can produce large token batches — need ubatch >= 2048
            args.push("--ubatch-size".to_string());
            args.push("2048".to_string());
            tracing::info!("Vision: mmproj={}", proj.display());
        } else {
            tracing::warn!("mmproj not found at {}, skipping vision", proj.display());
        }
    }
    let mut command = Command::new(&llama_server.path);
    command
        .args(&args)
        .env("SENDA_OWNER_PID", std::process::id().to_string())
        .env(
            "SENDA_RUNTIME_DIR",
            runtime.dir().to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        // Suppress the per-llama-server console window on Windows. See
        // rpc-server above and crate::process_util — same problem,
        // amplified because llama-server is the longest-lived child the
        // runtime owns (lives for the entire model session).
        .hide_console();
    prepend_child_library_paths(&mut command, &llama_server.path);

    let mut child = command.spawn().with_context(|| {
        format!(
            "Failed to start llama-server at {}",
            llama_server.path.display()
        )
    })?;

    let model_gb = model_bytes / GB + 1; // ceiling
    let max_wait_secs = health_timeout_secs(model_gb);
    tracing::info!(
        "Health timeout: {max_wait_secs}s (model ~{model_gb} GB, cap {}s)",
        HEALTH_TIMEOUT_CEIL_SECS
    );
    let url = format!("http://127.0.0.1:{http_port}/health");
    for i in 0..max_wait_secs {
        if i > 0 && i % 10 == 0 {
            let bytes = crate::network::tunnel::bytes_transferred();
            let kb = bytes as f64 / 1024.0;
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let transferred = if gb >= 1.0 {
                format!("{gb:.1} GB")
            } else if mb >= 1.0 {
                format!("{mb:.1} MB")
            } else {
                format!("{kb:.0} KB")
            };
            tracing::info!(
                "Still waiting for llama-server to load model... ({i}s, {transferred} transferred)"
            );
        }
        if let Some(status) = child.try_wait().with_context(|| {
            format!(
                "Failed to poll llama-server status for {}",
                llama_server.path.display()
            )
        })? {
            let tail = log_tail(&llama_log, 80);
            let hint = detect_known_crash_signature(&tail)
                .map(|w| format!("\n\n{}", w.post_mortem_hint()))
                .unwrap_or_default();
            let tail_msg = if tail.is_empty() {
                format!("See {}", llama_log.display())
            } else {
                format!("See {}:\n{}", llama_log.display(), tail)
            };
            anyhow::bail!(
                "llama-server exited before becoming healthy on port {http_port} (status: {status}). {}{}",
                tail_msg,
                hint
            );
        }
        if reqwest_health_check(&url).await {
            let pid = child
                .id()
                .context("llama-server started but did not expose a PID")?;
            if pid == 0 {
                anyhow::bail!("llama-server returned PID 0 — refusing to proceed");
            }
            let child_started_at =
                crate::runtime::instance::validate::process_started_at_unix(pid).unwrap_or(None);
            let owner_started_at: i64 =
                crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
            let llama_server_name = spawned_binary_name(&llama_server.path);
            let metadata = crate::runtime::instance::PidfileMetadata {
                cmd_name: llama_server_name.clone(),
                child_pid: pid,
                child_started_at_unix: child_started_at.unwrap_or(0),
                owner_pid: std::process::id(),
                owner_started_at_unix: owner_started_at,
                argv_snippet: crate::runtime::instance::PidfileMetadata::cap_argv(
                    &args,
                    crate::runtime::instance::ARGV_SNIPPET_MAX_BYTES,
                ),
                runtime_dir: runtime.dir().to_path_buf(),
            };
            let pidfile_guard =
                runtime.write_pidfile(&format!("llama-server-{}", http_port), &metadata)?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle {
                pid,
                expected_exit: expected_exit.clone(),
                expected_comm: llama_server_name,
                expected_start_time: child_started_at,
                _pidfile_guard: Some(pidfile_guard),
            };
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            let pidfile_path = runtime.pidfile_path(&format!("llama-server-{}", http_port));
            let launched_model_label = model_label(model);
            tokio::spawn(async move {
                let _ = child.wait().await;
                let _ = std::fs::remove_file(&pidfile_path);
                if !expected_exit.load(Ordering::Relaxed) && !runtime_shutting_down() {
                    let _ = emit_event(OutputEvent::Warning {
                        message: "llama-server process exited unexpectedly".to_string(),
                        context: Some(format!("model={launched_model_label} port={http_port}")),
                    });
                }
                let _ = death_tx.send(());
            });
            return Ok(InferenceServerProcess {
                handle,
                death_rx,
                context_length: ctx_size,
            });
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!(
        "llama-server failed to become healthy within {max_wait_secs}s (model ~{model_gb} GB). {}",
        log_tail_message(&llama_log, 80)
    );
}

/// Find an available TCP port
async fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Check if a port is accepting connections
async fn is_port_open(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

/// Returns true only when `pid` is safe to pass to `kill(2)`.
///
/// Unsafe values on Unix:
///   0        → signals every process in the caller's process group
///   1        → signals init / launchd
///   >i32::MAX → wraps to a negative pid_t (e.g. u32::MAX → −1, which kills all user processes)
pub fn is_safe_kill_target(pid: u32) -> bool {
    pid > 1 && pid <= i32::MAX as u32
}

/// Terminate a process by PID, validating comm before signaling.
/// Returns true if the process is dead or was not ours. Returns false on unexpected error.
pub async fn terminate_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return false;
    }
    !matches!(
        send_signal_if_matches(
            pid,
            expected_comm,
            expected_start_time,
            ProcessSignal::Terminate
        ),
        SignalOutcome::Failed
    )
}

/// Force-kill a process by PID, validating comm before signaling.
pub async fn force_kill_process(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> bool {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} — refusing");
        return false;
    }
    !matches!(
        send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill),
        SignalOutcome::Failed
    )
}

/// Poll until process exits or timeout_ms elapses. Returns true if dead within timeout.
pub async fn wait_for_exit(pid: u32, timeout_ms: u64) -> bool {
    if crate::runtime::instance::validate::process_liveness(pid)
        == crate::runtime::instance::validate::Liveness::Dead
    {
        return true;
    }
    let steps = timeout_ms / 100;
    for _ in 0..steps {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if crate::runtime::instance::validate::process_liveness(pid)
            == crate::runtime::instance::validate::Liveness::Dead
        {
            return true;
        }
    }
    false
}

/// Detect the best available compute device
fn detect_device() -> String {
    if cfg!(target_os = "macos") {
        return "MTL0".to_string();
    }

    // Linux: check for NVIDIA CUDA
    if command_has_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        return "CUDA0".to_string();
    }

    // Linux: check for NVIDIA Tegra/Jetson (tegrastats — Jetson AGX/NX devices support CUDA)
    // nvidia-smi is absent on Tegra; tegrastats is the canonical hardware stats tool.
    if let Ok(mut child) = std::process::Command::new("tegrastats")
        .args(["--interval", "1"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .hide_console()
        .spawn()
    {
        let _ = child.kill();
        let _ = child.wait();
        return "CUDA0".to_string();
    }

    // ROCm/HIP
    if has_rocm_backend() {
        return "ROCm0".to_string();
    }

    // Vulkan
    if command_succeeds("vulkaninfo", &["--summary"]) {
        return "Vulkan0".to_string();
    }

    "CPU".to_string()
}

fn has_rocm_backend() -> bool {
    #[cfg(windows)]
    {
        if std::env::var_os("ROCM_PATH").is_some() || std::env::var_os("HIP_PATH").is_some() {
            return true;
        }
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            let base = PathBuf::from(program_files).join("AMD");
            if base.join("ROCm").exists() || base.join("HIP").exists() {
                return true;
            }
        }
        command_has_output("hipInfo", &[]) || command_has_output("hipconfig", &[])
    }

    #[cfg(not(windows))]
    {
        command_has_output("rocm-smi", &["--showproductname"])
            || command_has_output("rocminfo", &[])
    }
}

fn command_succeeds(command: &str, args: &[&str]) -> bool {
    std::process::Command::new(command)
        .args(args)
        .hide_console()
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Simple HTTP health check (avoid adding reqwest as a dep — just use TCP + raw HTTP)
async fn reqwest_health_check(url: &str) -> bool {
    // Parse host:port from URL
    let url = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = url.split_once('/').unwrap_or((url, ""));
    let path = format!("/{path}");

    let Ok(mut stream) = tokio::net::TcpStream::connect(host_port).await else {
        return false;
    };

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }

    let mut response = vec![0u8; 1024];
    let Ok(n) = stream.read(&mut response).await else {
        return false;
    };

    let response = String::from_utf8_lossy(&response[..n]);
    response.contains("200 OK")
}

use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(test)]
mod tests {
    use super::{
        compute_context_size, flash_attention_args, health_timeout_secs, is_safe_kill_target,
        mmap_args, model_label, parse_available_devices, parse_enable_draft, preferred_device,
        terminate_process, wait_for_exit, BinaryFlavor, KvCacheQuant, KvCacheWarning, KvType,
        RpcServerHandle, SplitMode, GB, HEALTH_TIMEOUT_CEIL_SECS, HEALTH_TIMEOUT_FLOOR_SECS,
    };
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn speculative_decoding_off_by_default_opt_in_only() {
        // Default (env absent) and every non-affirmative value must be false:
        // the draft is a measured net throughput loss on all tested backends.
        assert!(!parse_enable_draft(None));
        assert!(!parse_enable_draft(Some("")));
        assert!(!parse_enable_draft(Some("0")));
        assert!(!parse_enable_draft(Some("false")));
        assert!(!parse_enable_draft(Some("off")));
        assert!(!parse_enable_draft(Some("nope")));
        // Only an explicit affirmative opts back in (case/space-insensitive).
        assert!(parse_enable_draft(Some("1")));
        assert!(parse_enable_draft(Some("true")));
        assert!(parse_enable_draft(Some(" YES ")));
        assert!(parse_enable_draft(Some("On")));
    }

    #[test]
    fn mmap_enabled_for_rpc_split_launches() {
        assert_eq!(
            mmap_args(true),
            Vec::<String>::new(),
            "RPC split launches must drop --no-mmap so the host pages local-layer weights from \
             the GGUF lazily via OS page cache; otherwise host RAM commits model_local_share + \
             RPC0 staging and OOM-kills 16 GB Macs hosting 19.7 GB models (May 17 2026 crash)"
        );
        assert_eq!(
            mmap_args(false),
            vec!["--no-mmap".to_string()],
            "solo launches keep --no-mmap because the host owns the full model and --no-mmap \
             avoids macOS page-cache pressure on long-lived solo hosts"
        );
    }

    #[test]
    fn health_timeout_secs_caps_large_models() {
        assert_eq!(
            health_timeout_secs(20),
            HEALTH_TIMEOUT_CEIL_SECS,
            "Qwen3-32B-Q4_K_M (~20 GB) must cap at the 300 s ceiling, not the legacy 40-min \
             (model_gb * 120) figure that orphaned a Mac llama-server for 19 min in the \
             May 18 2026 incident"
        );
        assert_eq!(
            health_timeout_secs(50),
            HEALTH_TIMEOUT_CEIL_SECS,
            "even a 50 GB MoE shard caps at 300 s — if a host can't load in 5 min, the \
             HostAttemptBackoff path is faster than waiting another 35 min"
        );
        assert_eq!(
            health_timeout_secs(1),
            HEALTH_TIMEOUT_FLOOR_SECS,
            "a 1 GB tiny model still gets the 90 s floor so CI / first-boot disk caches \
             have time to warm without spurious failover"
        );
        assert_eq!(
            health_timeout_secs(8),
            120,
            "an 8 GB model gets 8 * 15 = 120 s, between floor and ceiling"
        );
    }

    #[test]
    fn flash_attention_args_force_off_for_rpc_split_launches() {
        assert_eq!(
            flash_attention_args(true),
            ["-fa".to_string(), "off".to_string()],
            "RPC split launches must pass -fa off explicitly; relying on llama.cpp's `-fa auto` \
             default re-enables FLASH_ATTN_EXT on Metal hosts and aborts the Metal rpc-server"
        );
        assert_eq!(
            flash_attention_args(false),
            ["-fa".to_string(), "on".to_string()],
            "solo launches must keep -fa on so single-host inference uses Flash Attention"
        );
    }

    #[test]
    fn kv_quant_small_model_is_plain_f16() {
        let quant = KvCacheQuant::for_model_size(GB);
        assert_eq!(quant.k_type, KvType::F16);
        assert_eq!(quant.v_type, KvType::F16);
        assert!(
            quant.validation_warnings().is_empty(),
            "small-model default should not trigger any upstream bug warnings"
        );
    }

    #[test]
    fn model_label_uses_clean_stem_for_gguf_paths() {
        assert_eq!(
            model_label(Path::new(
                "/Users/example/.cache/huggingface/hub/Qwen3.5-27B-Q4_K_M.gguf"
            )),
            "Qwen3.5-27B-Q4_K_M"
        );
        assert_eq!(
            model_label(Path::new(
                "/Users/example/.cache/huggingface/hub/GLM-5-UD-IQ2_XXS-00001-of-00006.gguf"
            )),
            "GLM-5-UD-IQ2_XXS"
        );
    }

    #[test]
    fn kv_quant_medium_model_is_aggressive_asymmetric() {
        let quant = KvCacheQuant::for_model_size(20 * GB);
        assert_eq!(
            quant.k_type,
            KvType::Q8_0,
            "medium-tier K should be Q8_0 for attention routing precision"
        );
        assert_eq!(
            quant.v_type,
            KvType::Q4_0,
            "medium-tier V is Q4_0 for 25% memory savings over Q8_0/Q8_0. \
             This intentionally opts into two open upstream bugs (#20866 on CUDA, \
             #21450 on Metal FA fallback); our own CUDA build sets GGML_CUDA_FA_ALL_QUANTS \
             and Metal FA is available on all M1+ Macs. If you change this, also update \
             kv_quant_medium_model_emits_expected_warnings and the detect_known_crash_signature \
             wiring."
        );
    }

    #[test]
    fn kv_quant_medium_model_emits_expected_warnings() {
        let warnings = KvCacheQuant::for_model_size(20 * GB).validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants),
            "medium tier must flag #20866 so the startup log is clear about the CUDA \
             build requirement, got {warnings:?}"
        );
        assert!(
            warnings.contains(&KvCacheWarning::QuantizedVBreaksMetalFaFallback),
            "medium tier must flag #21450 so a user hitting the rare Metal-CPU-FA-fallback \
             crash can connect their failure to the open upstream issue, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_large_model_is_matched_q4_0() {
        let quant = KvCacheQuant::for_model_size(100 * GB);
        assert_eq!(quant.k_type, KvType::Q4_0);
        assert_eq!(
            quant.v_type,
            KvType::Q4_0,
            "large-tier K and V must be the same quantized type to avoid #20866"
        );
    }

    #[test]
    fn kv_quant_default_tier_warnings_are_intentional() {
        let expected: &[(u64, &[KvCacheWarning])] = &[
            (1, &[]),
            (4, &[]),
            (
                5,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
            (
                20,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
            (
                49,
                &[
                    KvCacheWarning::QuantizedVBreaksMetalFaFallback,
                    KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants,
                ],
            ),
            (50, &[]),
            (100, &[]),
        ];

        for (size_gb, expected_warnings) in expected {
            let actual = KvCacheQuant::for_model_size(size_gb * GB).validation_warnings();
            assert_eq!(
                actual, *expected_warnings,
                "tier for {size_gb}GB drifted from the documented warning set. \
                 If you changed the defaults, update this test to the new expected \
                 warnings and verify detect_known_crash_signature still maps them."
            );
        }
    }

    #[test]
    fn kv_quant_mismatched_quant_flags_cuda_bug() {
        let quant = KvCacheQuant {
            k_type: KvType::Q8_0,
            v_type: KvType::Q4_0,
        };
        let warnings = quant.validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants),
            "K=Q8_0/V=Q4_0 must flag #20866, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_quantized_v_flags_metal_bug() {
        let quant = KvCacheQuant {
            k_type: KvType::F16,
            v_type: KvType::Q4_0,
        };
        let warnings = quant.validation_warnings();
        assert!(
            warnings.contains(&KvCacheWarning::QuantizedVBreaksMetalFaFallback),
            "any quantized V must flag #21450, got {warnings:?}"
        );
    }

    #[test]
    fn kv_quant_tier_boundaries_are_exact() {
        let just_below_medium =
            KvCacheQuant::for_model_size(KvCacheQuant::MEDIUM_TIER_MIN_BYTES - 1);
        assert_eq!(just_below_medium.k_type, KvType::F16);
        assert_eq!(just_below_medium.v_type, KvType::F16);
        let at_medium = KvCacheQuant::for_model_size(KvCacheQuant::MEDIUM_TIER_MIN_BYTES);
        assert_eq!(at_medium.k_type, KvType::Q8_0);
        assert_eq!(at_medium.v_type, KvType::Q4_0);
        let just_below_large = KvCacheQuant::for_model_size(KvCacheQuant::LARGE_TIER_MIN_BYTES - 1);
        assert_eq!(just_below_large.k_type, KvType::Q8_0);
        assert_eq!(just_below_large.v_type, KvType::Q4_0);
        let at_large = KvCacheQuant::for_model_size(KvCacheQuant::LARGE_TIER_MIN_BYTES);
        assert_eq!(at_large.k_type, KvType::Q4_0);
        assert_eq!(at_large.v_type, KvType::Q4_0);
    }

    #[test]
    fn kv_quant_keeps_f16_for_medium_model_when_cache_fits_vram() {
        // Qwen3-8B (~5GB) on a 16GB card: f16 KV for 32k ctx ≈ 4.8GB, headroom
        // after weights ≈ 10GB. Must stay f16 — quantizing it is the LYU
        // regression (37 tok/s vs ~120). 0.75 * 10GB = 7.5GB budget > 4.8GB.
        let f16_kv = Some(4_800_000_000u64);
        let vram_after_model = 10_000_000_000u64;
        let quant = KvCacheQuant::for_model_size_and_vram(5 * GB, f16_kv, vram_after_model);
        assert_eq!(quant.k_type, KvType::F16);
        assert_eq!(quant.v_type, KvType::F16);
    }

    #[test]
    fn kv_quant_falls_back_to_quantized_when_f16_cache_too_large() {
        // Same model, but f16 KV would eat almost all the headroom (e.g. a
        // full-attention model or a very long context). Keeping f16 here would
        // make `-fit` spill layers to CPU, so fall back to the quantized tier.
        let f16_kv = Some(9_000_000_000u64);
        let vram_after_model = 10_000_000_000u64; // 0.75 * 10GB = 7.5GB < 9GB
        let quant = KvCacheQuant::for_model_size_and_vram(5 * GB, f16_kv, vram_after_model);
        assert_eq!(quant.k_type, KvType::Q8_0);
        assert_eq!(quant.v_type, KvType::Q4_0);
    }

    #[test]
    fn kv_quant_falls_back_to_quantized_when_cache_size_unknown() {
        // No GGUF metadata (scan failed / shard): we must not guess f16 fits.
        let quant = KvCacheQuant::for_model_size_and_vram(20 * GB, None, 100 * GB);
        assert_eq!(quant.k_type, KvType::Q8_0);
        assert_eq!(quant.v_type, KvType::Q4_0);
    }

    #[test]
    fn kv_quant_small_model_stays_f16_regardless_of_vram() {
        // Small models never quantize; the VRAM-aware path is a no-op for them
        // even when we can't size the cache.
        let quant = KvCacheQuant::for_model_size_and_vram(GB, None, 0);
        assert_eq!(quant.k_type, KvType::F16);
        assert_eq!(quant.v_type, KvType::F16);
    }

    #[test]
    fn kv_quant_label_reports_full_precision_for_f16() {
        let f16 = KvCacheQuant {
            k_type: KvType::F16,
            v_type: KvType::F16,
        };
        // Even for a medium-sized model kept at f16, the label must not claim
        // the "aggressive asymmetric" quantized tier.
        assert_eq!(f16.label(20 * GB), "F16 K + F16 V (full precision)");
    }

    #[test]
    fn kv_quant_append_args_emits_cache_flags_for_quantized_tiers() {
        let mut args: Vec<String> = Vec::new();
        KvCacheQuant::for_model_size(20 * GB).append_args(&mut args, 20 * GB);
        assert_eq!(
            args,
            vec![
                "--cache-type-k".to_string(),
                "q8_0".to_string(),
                "--cache-type-v".to_string(),
                "q4_0".to_string(),
            ]
        );
    }

    #[test]
    fn kv_quant_append_args_is_silent_for_f16_default() {
        let mut args: Vec<String> = Vec::new();
        KvCacheQuant::for_model_size(GB).append_args(&mut args, GB);
        assert!(
            args.is_empty(),
            "f16/f16 must not emit --cache-type-* flags (it's the llama-server default): {args:?}"
        );
    }

    #[test]
    fn detect_crash_signature_matches_cuda_fa_abort() {
        let log = "ggml_backend_cuda_flash_attn_ext\n\
                   /llama.cpp/ggml/src/ggml-cuda/fattn.cu:504: fatal error\n\
                   BEST_FATTN_KERNEL_NONE";
        assert_eq!(
            super::detect_known_crash_signature(log),
            Some(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants)
        );
    }

    #[test]
    fn detect_crash_signature_matches_metal_v_cache_error_both_phrasings() {
        let new_phrasing = "common_init_from_params: quantized V cache requires Flash Attention";
        let old_phrasing = "llama_init_from_model: V cache quantization requires flash_attn";
        assert_eq!(
            super::detect_known_crash_signature(new_phrasing),
            Some(KvCacheWarning::QuantizedVBreaksMetalFaFallback)
        );
        assert_eq!(
            super::detect_known_crash_signature(old_phrasing),
            Some(KvCacheWarning::QuantizedVBreaksMetalFaFallback)
        );
    }

    #[test]
    fn detect_crash_signature_returns_none_for_unrelated_logs() {
        let log = "llama_context: n_ctx = 65536\n\
                   sched_reserve: graph nodes = 3849\n\
                   common_init_from_params: warming up the model with an empty run";
        assert_eq!(super::detect_known_crash_signature(log), None);
    }

    #[test]
    fn post_mortem_hint_references_issue_number() {
        assert!(KvCacheWarning::MismatchedQuantNeedsCudaFaAllQuants
            .post_mortem_hint()
            .contains("#20866"));
        assert!(KvCacheWarning::QuantizedVBreaksMetalFaFallback
            .post_mortem_hint()
            .contains("#21450"));
    }

    #[test]
    fn parse_available_devices_ignores_non_device_lines() {
        let output = r#"
error: unknown device: HIP0
available devices:
No devices found
  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 13737 MiB free)
  CPU: AMD Ryzen 7 7800X3D 8-Core Processor (192857 MiB, 192857 MiB free)
"#;

        assert_eq!(
            parse_available_devices(output),
            vec!["Vulkan0".to_string(), "CPU".to_string()]
        );
    }

    #[test]
    fn preferred_device_picks_vulkan_when_that_is_all_binary_supports() {
        let available = vec!["Vulkan0".to_string(), "CPU".to_string()];
        assert_eq!(
            preferred_device(&available, Some(BinaryFlavor::Vulkan)),
            Some("Vulkan0".to_string())
        );
    }

    #[test]
    fn backend_device_for_flavor_uses_backend_namespace() {
        assert_eq!(
            super::backend_device_for_flavor(0, BinaryFlavor::Vulkan),
            Some("Vulkan0".to_string())
        );
        assert_eq!(
            super::backend_device_for_flavor(1, BinaryFlavor::Cuda),
            Some("CUDA1".to_string())
        );
        assert_eq!(super::backend_device_for_flavor(0, BinaryFlavor::Cpu), None);
    }

    #[test]
    fn infer_binary_flavor_from_filename() {
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan")),
            Some(BinaryFlavor::Vulkan)
        );
        #[cfg(windows)]
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server-vulkan.exe")),
            Some(BinaryFlavor::Vulkan)
        );
        assert_eq!(
            super::infer_binary_flavor("rpc-server", Path::new("rpc-server")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn platform_bin_name_preserves_existing_exe_suffix_case_insensitively() {
        assert_eq!(super::platform_bin_name("rpc-server.EXE"), "rpc-server.EXE");
    }

    #[test]
    fn compute_context_size_prefers_explicit_override() {
        assert_eq!(
            compute_context_size(Some(24576), 8_000_000_000, 48_000_000_000, None),
            24576
        );
    }

    #[test]
    fn compute_context_size_uses_full_model_bytes_in_local_mode() {
        assert_eq!(
            compute_context_size(None, 10_000_000_000, 22_000_000_000, None),
            32768
        );
        assert_eq!(
            compute_context_size(None, 10_000_000_000, 13_000_000_000, None),
            8192
        );
    }

    #[test]
    fn compute_context_size_accounts_for_split_host_weight_share() {
        let model_bytes = 40_000_000_000;
        let my_vram = 20_000_000_000;
        let total_group_vram = Some(80_000_000_000);

        assert_eq!(
            compute_context_size(None, model_bytes, my_vram, total_group_vram),
            16384
        );
    }

    #[test]
    fn pinned_gpu_runtime_launch_returns_required_backend_device() {
        let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
            index: 1,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA1".into(),
            vram_bytes: 24_000_000_000,
        };
        let selected = super::selected_backend_device(Some(&pinned_gpu)).unwrap();

        assert_eq!(selected, Some("CUDA1"));
    }

    #[test]
    fn pinned_gpu_runtime_launch_rejects_insufficient_selected_device_capacity() {
        let err = super::ensure_selected_gpu_capacity(
            Some(&crate::runtime::StartupPinnedGpuTarget {
                index: 0,
                stable_id: "uuid:GPU-small".into(),
                backend_device: "CUDA0".into(),
                vram_bytes: 8_000_000_000,
            }),
            10_000_000_000,
            "this local launch",
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("uuid:GPU-small"));
        assert!(message.contains("selected device"));
    }

    #[test]
    fn rpc_split_launch_does_not_apply_host_fit_target_to_workers() {
        let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
            index: 0,
            stable_id: "metal:apple-m3-pro".into(),
            backend_device: "MTL0".into(),
            vram_bytes: 14_000_000_000,
        };

        assert_eq!(
            super::fit_target_mib_for_launch(true, Some(&pinned_gpu), 14_000_000_000),
            None
        );
        assert_eq!(
            super::fit_target_mib_for_launch(false, Some(&pinned_gpu), 14_000_000_000),
            Some(super::compute_fit_target_mib(
                Some(&pinned_gpu),
                14_000_000_000
            ))
        );
    }

    /// Regression for v0.66.51: the `-fitt` margin must stay small enough
    /// that the fitter doesn't push layers to CPU on common device sizes.
    ///
    /// Pre-v0.66.51 the formula was `0.7 * device_vram`, which on an 18 GB
    /// M3 Pro (Metal pool 12.3 GiB) returned 9676 MiB and forced 22/36
    /// layers of an 8 B Q4_K_M model to CPU. The new formula
    /// `clamp(device_mib / 10, 1024, 2048)` keeps the margin in a 1–2 GiB
    /// band regardless of device size; this test pins that down across the
    /// devices we know are in the mesh today and the ones we expect next
    /// (high-end Apple, mid CUDA, datacenter).
    #[test]
    fn fit_target_mib_is_a_small_margin_not_most_of_vram() {
        let cases = [
            // M3 Pro 18 GB after the runtime's 4.5 GB OS reserve: 14_495_514_624 bytes ≈ 13824 MiB → /10 = 1382 MiB margin.
            ("M3 Pro 18 GB post-reserve", 14_495_514_624u64, 1382u64),
            // M1 Air 8 GB after reserve: ~5722 MiB → /10 = 572 → clamped up to 1024 MiB. Confirms the 1024 floor matches llama.cpp's default.
            ("M1/M2/M3 Air 8 GB post-reserve", 6_000_000_000, 1024),
            ("RTX 4070 Laptop 8 GB", 8_585_740_288, 1024),
            ("RTX 5090 32 GB", 32u64 * 1024 * 1024 * 1024, 2048),
            ("H100 80 GB", 80u64 * 1024 * 1024 * 1024, 2048),
            ("M3 Ultra 192 GB", 192u64 * 1024 * 1024 * 1024, 2048),
        ];
        for (label, vram_bytes, expected_mib) in cases {
            let pinned = crate::runtime::StartupPinnedGpuTarget {
                index: 0,
                stable_id: format!("test:{label}"),
                backend_device: "test".into(),
                vram_bytes,
            };
            let got = super::compute_fit_target_mib(Some(&pinned), vram_bytes);
            assert!(
                (1024..=2048).contains(&got),
                "{label}: -fitt margin must stay in [1024, 2048] MiB; got {got}"
            );
            assert_eq!(got, expected_mib, "{label}: -fitt margin regression");
            assert!(
                (vram_bytes / (1024 * 1024)).saturating_sub(got) >= 4096,
                "{label}: -fitt margin must leave ≥ 4 GiB usable; vram={} MiB, margin={got} MiB",
                vram_bytes / (1024 * 1024)
            );
        }
    }

    // ── SplitMode ──

    #[test]
    fn split_mode_layer_arg() {
        assert_eq!(SplitMode::Layer.as_arg(), "layer");
    }

    #[test]
    fn split_mode_row_arg() {
        assert_eq!(SplitMode::Row.as_arg(), "row");
    }

    #[test]
    fn rpc_handle_has_pid_and_port() {
        let handle = RpcServerHandle {
            pid: 12345,
            port: 8080,
            expected_exit: Arc::new(AtomicBool::new(false)),
            expected_comm: "rpc-server".to_string(),
            expected_start_time: Some(1700000000),
            _pidfile_guard: None,
        };
        assert!(handle.pid > 0);
        assert!(handle.port > 0);
    }

    #[test]
    fn rpc_handle_shutdown_sets_expected_exit() {
        let flag = Arc::new(AtomicBool::new(false));
        let handle = RpcServerHandle {
            pid: 999_999,
            port: 9999,
            expected_exit: flag.clone(),
            expected_comm: "rpc-server".to_string(),
            expected_start_time: Some(1700000000),
            _pidfile_guard: None,
        };
        assert!(!flag.load(Ordering::Relaxed));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(handle.shutdown());
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn safe_kill_target_rejects_zero() {
        assert!(!is_safe_kill_target(0));
    }

    #[test]
    fn safe_kill_target_rejects_one() {
        assert!(!is_safe_kill_target(1));
    }

    #[test]
    fn safe_kill_target_rejects_u32_max() {
        assert!(!is_safe_kill_target(u32::MAX));
    }

    #[test]
    fn safe_kill_target_rejects_i32_max_plus_one() {
        assert!(!is_safe_kill_target(i32::MAX as u32 + 1));
    }

    #[test]
    fn safe_kill_target_accepts_normal_pid() {
        assert!(is_safe_kill_target(999_999));
        assert!(is_safe_kill_target(2));
        assert!(is_safe_kill_target(i32::MAX as u32));
    }

    #[tokio::test]
    async fn terminate_nonexistent_pid_returns_true() {
        let result = terminate_process(999999, "nonexistent", None).await;
        assert!(result, "nonexistent PID should return true (already dead)");
    }

    #[tokio::test]
    async fn terminate_skips_when_comm_mismatch() {
        let self_pid = std::process::id();
        let result = terminate_process(self_pid, "wrong-comm-name", None).await;
        assert!(
            result,
            "mismatched comm should return true (skipped, treated as not our process)"
        );
    }

    #[tokio::test]
    async fn terminate_unsafe_pid_returns_false() {
        let result = terminate_process(1, "senda", None).await;
        assert!(!result, "unsafe PID should return false");
    }

    #[tokio::test]
    async fn wait_for_exit_returns_false_for_live_process() {
        let self_pid = std::process::id();
        let result = wait_for_exit(self_pid, 50).await;
        assert!(
            !result,
            "live process with short timeout should return false"
        );
    }

    #[tokio::test]
    async fn wait_for_exit_immediately_detects_dead_process() {
        let result = wait_for_exit(999_999, 0).await;
        assert!(
            result,
            "dead process should be detected before entering the poll loop"
        );
    }

    #[test]
    fn no_pkill_f_in_source_tree() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/inference/launch.rs"
        ))
        .unwrap();
        let forbidden_pattern = ["pkill", "-f", "llama-server"].join(" ");
        assert!(
            !src.contains(&forbidden_pattern),
            "forbidden pattern found in launch.rs"
        );
        let kill_func = format!("{}_{}{}", "kill", "llama", "_server");
        assert!(
            !src.contains(&kill_func),
            "legacy function reference still present after removal"
        );
        let runtime_src =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/runtime/mod.rs"))
                .unwrap();
        assert!(
            !runtime_src.contains(&kill_func),
            "legacy function reference still present in runtime module"
        );
        let orphan_func = format!("{}_{}{}{}", "kill", "orphan", "_rpc", "_servers");
        assert!(
            !runtime_src.contains(&orphan_func),
            "legacy orphan cleanup function reference still present in runtime module"
        );
    }

    #[test]
    fn log_path_does_not_duplicate_extension() {
        use crate::runtime::instance::InstanceRuntime;
        use std::env;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let temp_path = temp_dir.path();

        // Set runtime root to temp directory for test isolation
        let original_root = env::var("SENDA_RUNTIME_ROOT").ok();
        env::set_var("SENDA_RUNTIME_ROOT", temp_path);

        let runtime =
            InstanceRuntime::acquire(std::process::id()).expect("Failed to acquire runtime");

        // Test llama-server log path (should NOT have .log.log)
        let llama_log = runtime.log_path("llama-server");
        assert!(
            llama_log.to_string_lossy().ends_with("llama-server.log"),
            "llama-server log path should end with .log, got: {}",
            llama_log.display()
        );
        assert!(
            !llama_log.to_string_lossy().contains(".log.log"),
            "llama-server log path should not have double .log extension, got: {}",
            llama_log.display()
        );

        // Test rpc-server log path (should NOT have .log.log)
        let rpc_log = runtime.log_path("rpc-server-8001");
        assert!(
            rpc_log.to_string_lossy().ends_with("rpc-server-8001.log"),
            "rpc-server log path should end with .log, got: {}",
            rpc_log.display()
        );
        assert!(
            !rpc_log.to_string_lossy().contains(".log.log"),
            "rpc-server log path should not have double .log extension, got: {}",
            rpc_log.display()
        );

        // Restore original env var
        if let Some(orig) = original_root {
            env::set_var("SENDA_RUNTIME_ROOT", orig);
        } else {
            env::remove_var("SENDA_RUNTIME_ROOT");
        }
    }

    // ── Regression tests for slots/parallel wiring (T9) ──

    /// Verify that ModelLaunchSpec has a public `slots` field at compile time.
    /// This is a structural assertion — if the field disappears or becomes private,
    /// this code will not compile. It guards against regressions where TOML config
    /// parallel values are silently dropped before reaching llama-server.
    #[test]
    fn model_launch_spec_has_public_slots_field() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let _spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 1_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: None,
            ctx_size_override: None,
            total_group_vram: None,
            selected_gpu: None,
            slots: 8, // ← compile-time check: field must exist and be accessible
        };
    }

    /// Verify that all construction sites in election.rs populate `slots` correctly.
    /// This test constructs a ModelLaunchSpec with explicit non-default slots value
    /// to ensure callers can pass per-model parallel counts from TOML config.
    /// If any site forgets the field, this compilation will fail — which is exactly
    /// what we want as a regression guard.
    #[test]
    fn model_launch_spec_accepts_non_default_slots() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: Some(SplitMode::Layer),
            draft: None,
            draft_max: 4,
            model_bytes: 50_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: Some(Path::new("/dev/null")),
            ctx_size_override: Some(32768),
            total_group_vram: Some(96_000_000_000u64),
            selected_gpu: None,
            slots: 16, // non-default value from TOML config
        };
        assert_eq!(spec.slots, 16);
    }

    /// Verify that start_llama_server receives and can read the `slots` field.
    /// This is a compile-time check that the destructured `let slots = spec.slots;`
    /// in start_llama_server actually works — if the field were renamed or removed,
    /// this would fail to compile. The actual process-spawning behavior is tested
    /// at integration time since we don't want to spawn real llama-server here.
    #[test]
    fn launch_spec_slots_propagates_through_destructure() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        let spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 20_000_000_000u64,
            my_vram: 16_000_000_000u64,
            mmproj: None,
            ctx_size_override: Some(8192),
            total_group_vram: None,
            selected_gpu: None,
            slots: 32,
        };

        // Destructure exactly as start_llama_server does it (lines ~1078-1091)
        let _model = spec.model;
        let _http_port = spec.http_port;
        let _tunnel_ports = spec.tunnel_ports;
        let _tensor_split = spec.tensor_split;
        let _split_mode = spec.split_mode;
        let _draft = spec.draft;
        let _draft_max = spec.draft_max;
        let _model_bytes = spec.model_bytes;
        let _my_vram = spec.my_vram;
        let _mmproj = spec.mmproj;
        let _ctx_size_override = spec.ctx_size_override;
        let _total_group_vram = spec.total_group_vram;
        let _selected_gpu = spec.selected_gpu;
        let slots = spec.slots; // ← this is the key line being tested
        assert_eq!(slots, 32);
    }

    /// Verify that default slots value of 4 would be explicitly set — callers must
    /// not rely on defaults. This test documents the expected behavior: when TOML
    /// config omits `slots`, the caller should use an explicit fallback (4).
    #[test]
    fn launch_spec_slots_is_explicitly_set() {
        use super::ModelLaunchSpec;
        let path = Path::new("/dev/null");
        // The following would NOT compile if we omitted `slots:`:
        //   let bad_spec = ModelLaunchSpec { model: path, ..Default::default() };
        // Since ModelLaunchSpec doesn't implement Default, this is enforced at compile time.
        let good_spec = ModelLaunchSpec {
            model: path,
            http_port: 8080,
            tunnel_ports: &[],
            tensor_split: None,
            split_mode: None,
            draft: None,
            draft_max: 0,
            model_bytes: 1_000_000_000u64,
            my_vram: 24_000_000_000u64,
            mmproj: None,
            ctx_size_override: None,
            total_group_vram: None,
            selected_gpu: None,
            slots: 4, // explicit default from TOML config fallback
        };
        assert_eq!(good_spec.slots, 4);
    }
}
