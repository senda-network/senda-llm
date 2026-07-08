//! Node capability schema for capability-aware mesh routing.
//!
//! Every mesh peer advertises a [`NodeCapability`] alongside its existing
//! `PeerAnnouncement`. The router uses this to filter live peers down to
//! the set that can actually serve a given model before applying load /
//! latency scoring.
//!
//! Schema is intentionally coarse — over-fine-grained classes just shift
//! the brittleness from model-arch checks to compute-class checks.

use serde::{Deserialize, Serialize};

use crate::proto::node as pb;

/// Acceleration backend selected at build time on this node.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }

    pub fn to_proto(self) -> i32 {
        match self {
            Backend::Metal => pb::Backend::Metal as i32,
            Backend::Cuda => pb::Backend::Cuda as i32,
            Backend::Rocm => pb::Backend::Rocm as i32,
            Backend::Vulkan => pb::Backend::Vulkan as i32,
            Backend::Cpu => pb::Backend::Cpu as i32,
        }
    }

    pub fn from_proto(value: i32) -> Option<Self> {
        match pb::Backend::try_from(value).ok()? {
            pb::Backend::Metal => Some(Backend::Metal),
            pb::Backend::Cuda => Some(Backend::Cuda),
            pb::Backend::Rocm => Some(Backend::Rocm),
            pb::Backend::Vulkan => Some(Backend::Vulkan),
            pb::Backend::Cpu => Some(Backend::Cpu),
            pb::Backend::Unspecified => None,
        }
    }
}

/// GPU vendor (or [`GpuVendor::None`] for CPU-only nodes).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    None,
    Apple,
    Nvidia,
    Amd,
    Intel,
}

impl GpuVendor {
    pub fn label(self) -> &'static str {
        match self {
            GpuVendor::None => "none",
            GpuVendor::Apple => "apple",
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
        }
    }

    pub fn to_proto(self) -> i32 {
        match self {
            GpuVendor::None => pb::GpuVendor::None as i32,
            GpuVendor::Apple => pb::GpuVendor::Apple as i32,
            GpuVendor::Nvidia => pb::GpuVendor::Nvidia as i32,
            GpuVendor::Amd => pb::GpuVendor::Amd as i32,
            GpuVendor::Intel => pb::GpuVendor::Intel as i32,
        }
    }

    pub fn from_proto(value: i32) -> Option<Self> {
        match pb::GpuVendor::try_from(value).ok()? {
            pb::GpuVendor::None => Some(GpuVendor::None),
            pb::GpuVendor::Apple => Some(GpuVendor::Apple),
            pb::GpuVendor::Nvidia => Some(GpuVendor::Nvidia),
            pb::GpuVendor::Amd => Some(GpuVendor::Amd),
            pb::GpuVendor::Intel => Some(GpuVendor::Intel),
            pb::GpuVendor::Unspecified => None,
        }
    }

    /// Best-effort vendor inference from the legacy `gpu_name` string.
    /// Used to back-fill capability for older peers that don't advertise it.
    pub fn from_gpu_name(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        if n.contains("apple")
            || n.contains(" m1")
            || n.contains(" m2")
            || n.contains(" m3")
            || n.contains(" m4")
            || n.contains(" m5")
        {
            GpuVendor::Apple
        } else if n.contains("nvidia")
            || n.contains("rtx")
            || n.contains("gtx")
            || n.contains("tesla")
            || n.contains("a100")
            || n.contains("h100")
            || n.contains("h200")
            || n.contains("l40")
            || n.contains("a40")
            || n.contains("jetson")
        {
            GpuVendor::Nvidia
        } else if n.contains("amd")
            || n.contains("radeon")
            || n.contains("instinct")
            || n.starts_with("mi")
            || n.contains("rx ")
            || n.contains("rdna")
        {
            GpuVendor::Amd
        } else if n.contains("intel")
            || n.contains("arc ")
            || n.contains(" arc")
            || n.contains("iris")
        {
            GpuVendor::Intel
        } else {
            GpuVendor::None
        }
    }
}

/// Coarse compute tier. Lo/Mid/Hi/Pro = "phone-class / laptop-iGPU /
/// consumer-discrete-GPU / datacenter-GPU".
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ComputeClass {
    Lo,
    Mid,
    Hi,
    Pro,
}

impl ComputeClass {
    pub fn label(self) -> &'static str {
        match self {
            ComputeClass::Lo => "lo",
            ComputeClass::Mid => "mid",
            ComputeClass::Hi => "hi",
            ComputeClass::Pro => "pro",
        }
    }

    pub fn to_proto(self) -> i32 {
        match self {
            ComputeClass::Lo => pb::ComputeClass::Lo as i32,
            ComputeClass::Mid => pb::ComputeClass::Mid as i32,
            ComputeClass::Hi => pb::ComputeClass::Hi as i32,
            ComputeClass::Pro => pb::ComputeClass::Pro as i32,
        }
    }

    pub fn from_proto(value: i32) -> Option<Self> {
        match pb::ComputeClass::try_from(value).ok()? {
            pb::ComputeClass::Lo => Some(ComputeClass::Lo),
            pb::ComputeClass::Mid => Some(ComputeClass::Mid),
            pb::ComputeClass::Hi => Some(ComputeClass::Hi),
            pb::ComputeClass::Pro => Some(ComputeClass::Pro),
            pb::ComputeClass::Unspecified => None,
        }
    }

    /// Heuristic compute-class derivation from VRAM + backend. Good enough as a
    /// default when no explicit per-GPU mapping is configured.
    pub fn from_vram_and_backend(vram_mb: u64, backend: Backend) -> ComputeClass {
        match backend {
            Backend::Cpu => ComputeClass::Lo,
            _ => match vram_mb {
                0..=8_191 => ComputeClass::Lo,
                8_192..=16_383 => ComputeClass::Mid,
                16_384..=49_151 => ComputeClass::Hi,
                _ => ComputeClass::Pro,
            },
        }
    }
}

/// Per-node capability advertisement. See module doc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeCapability {
    pub backend: Backend,
    pub vendor: GpuVendor,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub compute_class: ComputeClass,
    pub supported_archs: Vec<String>,
    pub supported_quants: Vec<String>,
    pub loaded_models: Vec<String>,
    /// Largest model (in GB) this node could realistically serve solo.
    /// 0 means "unknown — back-fill with vram-derived heuristic".
    pub can_serve_max_gb: u64,
}

impl Default for NodeCapability {
    /// Default capability: CPU-only, unknown VRAM. Equivalent to "we don't
    /// know much about this node — treat it as the most conservative option."
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            vendor: GpuVendor::None,
            vram_total_mb: 0,
            vram_free_mb: 0,
            compute_class: ComputeClass::Lo,
            supported_archs: Vec::new(),
            supported_quants: Vec::new(),
            loaded_models: Vec::new(),
            can_serve_max_gb: 0,
        }
    }
}

impl NodeCapability {
    pub fn to_proto(&self) -> pb::NodeCapability {
        pb::NodeCapability {
            backend: self.backend.to_proto(),
            vendor: self.vendor.to_proto(),
            vram_total_mb: self.vram_total_mb,
            vram_free_mb: self.vram_free_mb,
            compute_class: self.compute_class.to_proto(),
            supported_archs: self.supported_archs.clone(),
            supported_quants: self.supported_quants.clone(),
            loaded_models: self.loaded_models.clone(),
            can_serve_max_gb: self.can_serve_max_gb,
        }
    }

    pub fn from_proto(p: &pb::NodeCapability) -> Option<Self> {
        Some(Self {
            backend: Backend::from_proto(p.backend)?,
            vendor: GpuVendor::from_proto(p.vendor).unwrap_or(GpuVendor::None),
            vram_total_mb: p.vram_total_mb,
            vram_free_mb: p.vram_free_mb,
            compute_class: ComputeClass::from_proto(p.compute_class).unwrap_or(
                ComputeClass::from_vram_and_backend(
                    p.vram_total_mb,
                    Backend::from_proto(p.backend)?,
                ),
            ),
            supported_archs: p.supported_archs.clone(),
            supported_quants: p.supported_quants.clone(),
            loaded_models: p.loaded_models.clone(),
            can_serve_max_gb: p.can_serve_max_gb,
        })
    }

    /// True when this node can plausibly serve a model with the given
    /// requirements solo (no pipeline-parallel split).
    pub fn matches(&self, req: &CapabilityRequirements) -> bool {
        if let Some(arch) = &req.arch {
            if !self.supported_archs.is_empty()
                && !self
                    .supported_archs
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(arch))
            {
                return false;
            }
        }
        if let Some(quants) = &req.quants {
            if !self.supported_quants.is_empty()
                && !quants.iter().any(|q| {
                    self.supported_quants
                        .iter()
                        .any(|sq| sq.eq_ignore_ascii_case(q))
                })
            {
                return false;
            }
        }
        if let Some(min_vram_mb) = req.min_vram_mb {
            // VRAM unknown == 0 == "treat as CPU"; only fail when both sides know.
            if self.vram_total_mb > 0 && self.vram_total_mb < min_vram_mb {
                return false;
            }
            if self.backend == Backend::Cpu {
                return false;
            }
        }
        if let Some(max_gb) = req.min_can_serve_gb {
            if self.can_serve_max_gb > 0 && self.can_serve_max_gb < max_gb {
                return false;
            }
        }
        true
    }
}

/// Detect the *local* node's [`NodeCapability`] from the build target plus
/// runtime hardware probe results. Called once at startup and gossiped.
///
/// Inputs are the values the node already collects today: legacy
/// `gpu_name`/`gpu_vram` strings, the `is_soc` hint, the names of the models
/// this node intends to serve, and the `BinaryFlavor` of the bundled
/// llama-server (if known — this overrides the cfg-derived guess).
pub fn detect_local_capability(
    gpu_name: Option<&str>,
    gpu_vram_str: Option<&str>,
    is_soc: Option<bool>,
    serving_models: &[String],
    bundle_flavor: Option<&str>,
) -> NodeCapability {
    let mut cap = backfill_from_legacy(gpu_name, gpu_vram_str, is_soc, serving_models);
    let target_backend = backend_from_target_os(cap.vendor);
    cap.backend = match (bundle_flavor, target_backend) {
        (Some(f), _) => parse_bundle_flavor(f).unwrap_or(target_backend),
        (None, b) => b,
    };
    if cap.backend == Backend::Cpu {
        cap.vram_total_mb = 0;
        cap.vram_free_mb = 0;
    }
    cap.compute_class = ComputeClass::from_vram_and_backend(cap.vram_total_mb, cap.backend);
    cap.supported_archs = default_supported_archs()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    cap.supported_quants = default_supported_quants()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    cap.can_serve_max_gb = max_servable_model_gb(cap.backend, cap.vram_total_mb);
    cap
}

/// Compute-time approximation of the largest model (in whole GB) this node
/// could realistically serve solo. Roughly: VRAM / 1.2 for GPU backends
/// (leaving headroom for KV cache + activations + the OS); CPU nodes get a
/// fraction of system RAM as a stand-in but we don't probe sysram here, so
/// just zero — the router treats 0 as "use heuristic on vram_total_mb".
fn max_servable_model_gb(backend: Backend, vram_total_mb: u64) -> u64 {
    if backend == Backend::Cpu || vram_total_mb == 0 {
        return 0;
    }
    let usable_mb = (vram_total_mb as f64) / 1.2;
    (usable_mb / 1024.0).floor().max(0.0) as u64
}

/// llama.cpp model architectures we know we can run with the bundled
/// `llama-server`. Hard-coded against current llama.cpp; reviewed per upgrade.
/// Used by the router as the "I can plausibly run this" filter for the
/// requested model's `arch` field.
pub fn default_supported_archs() -> &'static [&'static str] {
    &[
        "llama",
        "llama2",
        "llama3",
        "mistral",
        "mixtral",
        "qwen",
        "qwen2",
        "qwen2.5",
        "qwen3",
        "phi",
        "phi2",
        "phi3",
        "phi4",
        "gemma",
        "gemma2",
        "gemma3",
        "deepseek",
        "deepseek2",
        "command-r",
        "command-r-plus",
        "falcon",
        "starcoder",
        "starcoder2",
        "bloom",
        "yi",
        "olmo",
        "stablelm",
        "minicpm",
        "internlm",
    ]
}

/// GGUF quantization formats supported by llama.cpp at runtime. New formats
/// rarely appear; the list is reviewed per llama.cpp upgrade.
pub fn default_supported_quants() -> &'static [&'static str] {
    &[
        "f32", "f16", "bf16", "q8_0", "q6_k", "q5_k_s", "q5_k_m", "q5_0", "q5_1", "q4_k_s",
        "q4_k_m", "q4_0", "q4_1", "q3_k_s", "q3_k_m", "q3_k_l", "q2_k", "iq1_m", "iq1_s",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_s", "iq3_m", "iq4_xs", "iq4_nl",
    ]
}

fn parse_bundle_flavor(raw: &str) -> Option<Backend> {
    match raw.to_ascii_lowercase().as_str() {
        "metal" => Some(Backend::Metal),
        "cuda" => Some(Backend::Cuda),
        "rocm" => Some(Backend::Rocm),
        "vulkan" => Some(Backend::Vulkan),
        "cpu" => Some(Backend::Cpu),
        _ => None,
    }
}

fn backend_from_target_os(vendor: GpuVendor) -> Backend {
    if cfg!(target_os = "macos") {
        return Backend::Metal;
    }
    if cfg!(target_os = "linux") {
        return match vendor {
            GpuVendor::Nvidia => Backend::Cuda,
            GpuVendor::Amd => Backend::Rocm,
            GpuVendor::Intel => Backend::Vulkan,
            GpuVendor::Apple | GpuVendor::None => Backend::Cpu,
        };
    }
    if cfg!(target_os = "windows") {
        // Windows ROCm story is poor in practice; prefer Vulkan for AMD/Intel.
        return match vendor {
            GpuVendor::Nvidia => Backend::Cuda,
            GpuVendor::Amd | GpuVendor::Intel => Backend::Vulkan,
            GpuVendor::Apple | GpuVendor::None => Backend::Cpu,
        };
    }
    Backend::Cpu
}

/// Best-effort capability for a peer that didn't advertise one (older
/// runtime versions). We synthesize from the legacy `gpu_name` / `gpu_vram`
/// fields plus an optional `is_soc` hint.
///
/// `gpu_vram_str` is the legacy free-form string ("16 GB", "8589934592"),
/// which we parse leniently — `0` total VRAM signals "treat as CPU".
pub fn backfill_from_legacy(
    gpu_name: Option<&str>,
    gpu_vram_str: Option<&str>,
    is_soc: Option<bool>,
    serving_models: &[String],
) -> NodeCapability {
    let vram_total_mb = parse_legacy_vram_mb(gpu_vram_str);
    let vendor = gpu_name
        .map(GpuVendor::from_gpu_name)
        .unwrap_or(GpuVendor::None);
    let backend = if vram_total_mb == 0 {
        Backend::Cpu
    } else {
        match (vendor, is_soc.unwrap_or(false)) {
            (GpuVendor::Apple, _) | (_, true) => Backend::Metal,
            (GpuVendor::Nvidia, _) => Backend::Cuda,
            (GpuVendor::Amd, _) => Backend::Rocm,
            (GpuVendor::Intel, _) => Backend::Vulkan,
            (GpuVendor::None, _) => Backend::Cpu,
        }
    };
    let compute_class = ComputeClass::from_vram_and_backend(vram_total_mb, backend);
    let can_serve_max_gb = (vram_total_mb / 1024).saturating_sub(1);
    NodeCapability {
        backend,
        vendor,
        vram_total_mb,
        vram_free_mb: vram_total_mb,
        compute_class,
        supported_archs: Vec::new(),
        supported_quants: Vec::new(),
        loaded_models: serving_models.to_vec(),
        can_serve_max_gb,
    }
}

fn parse_legacy_vram_mb(raw: Option<&str>) -> u64 {
    let Some(s) = raw else {
        return 0;
    };
    let s = s.trim();
    if s.is_empty() {
        return 0;
    }
    // Try plain bytes integer first ("8589934592").
    if let Ok(bytes) = s.parse::<u64>() {
        return bytes / 1024 / 1024;
    }
    // Try unit-suffixed forms: "16 GB", "16GB", "8 GiB", "512 MB".
    let lower = s.to_ascii_lowercase();
    let (num_part, unit_part) = lower
        .find(|c: char| c.is_ascii_alphabetic())
        .map(|i| (lower[..i].trim().to_string(), lower[i..].trim().to_string()))
        .unwrap_or((lower.clone(), String::new()));
    let num: f64 = num_part.parse().unwrap_or(0.0);
    let mb = match unit_part.as_str() {
        "kb" | "kib" => num / 1024.0,
        "mb" | "mib" | "" => num,
        "gb" | "gib" => num * 1024.0,
        "tb" | "tib" => num * 1024.0 * 1024.0,
        _ => num,
    };
    mb.max(0.0) as u64
}

/// Inputs the router resolves from a model id before filtering peers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityRequirements {
    /// e.g. "qwen2", "llama3", "mixtral". Match is case-insensitive against
    /// `NodeCapability::supported_archs`. None = don't filter on arch.
    pub arch: Option<String>,
    /// Acceptable quantizations, in preference order. None = don't filter.
    pub quants: Option<Vec<String>>,
    /// Minimum VRAM (MB) this node must report to be a candidate. Implies the
    /// node must have a non-CPU backend.
    pub min_vram_mb: Option<u64>,
    /// Minimum `can_serve_max_gb` value (in whole GB) the node must self-report.
    pub min_can_serve_gb: Option<u64>,
}

impl CapabilityRequirements {
    pub fn for_model_size_gb(size_gb: u64) -> Self {
        // Heuristic: a model fits comfortably in VRAM when total VRAM ≥ size_gb * 1.2.
        let min_vram_mb = ((size_gb as f64) * 1024.0 * 1.2) as u64;
        Self {
            min_vram_mb: Some(min_vram_mb),
            min_can_serve_gb: Some(size_gb),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(backend: Backend, vram_mb: u64) -> NodeCapability {
        NodeCapability {
            backend,
            vendor: GpuVendor::Nvidia,
            vram_total_mb: vram_mb,
            vram_free_mb: vram_mb,
            compute_class: ComputeClass::from_vram_and_backend(vram_mb, backend),
            supported_archs: vec!["llama3".into(), "qwen2".into()],
            supported_quants: vec!["q4_k_m".into()],
            loaded_models: Vec::new(),
            can_serve_max_gb: vram_mb / 1024,
        }
    }

    #[test]
    fn proto_roundtrip() {
        let original = cap(Backend::Cuda, 24_576);
        let p = original.to_proto();
        let restored = NodeCapability::from_proto(&p).expect("roundtrip");
        assert_eq!(original, restored);
    }

    #[test]
    fn matches_arch_filter() {
        let c = cap(Backend::Cuda, 24_576);
        let mut req = CapabilityRequirements {
            arch: Some("LLAMA3".into()),
            ..Default::default()
        };
        assert!(c.matches(&req));
        req.arch = Some("mixtral".into());
        assert!(!c.matches(&req));
    }

    #[test]
    fn matches_quant_filter() {
        let c = cap(Backend::Cuda, 24_576);
        let mut req = CapabilityRequirements {
            quants: Some(vec!["q4_k_m".into(), "q5_k_m".into()]),
            ..Default::default()
        };
        assert!(c.matches(&req));
        req.quants = Some(vec!["q8_0".into()]);
        assert!(!c.matches(&req));
    }

    #[test]
    fn cpu_node_excluded_when_min_vram_set() {
        let cpu_cap = cap(Backend::Cpu, 0);
        let req = CapabilityRequirements {
            min_vram_mb: Some(8_000),
            ..Default::default()
        };
        assert!(!cpu_cap.matches(&req));
    }

    #[test]
    fn vendor_inference_basic() {
        assert_eq!(GpuVendor::from_gpu_name("Apple M4 Max"), GpuVendor::Apple);
        assert_eq!(
            GpuVendor::from_gpu_name("NVIDIA RTX 4090"),
            GpuVendor::Nvidia
        );
        assert_eq!(
            GpuVendor::from_gpu_name("AMD Radeon RX 7900 XTX"),
            GpuVendor::Amd
        );
        assert_eq!(GpuVendor::from_gpu_name("Intel Arc A770"), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_gpu_name(""), GpuVendor::None);
    }

    #[test]
    fn compute_class_buckets() {
        assert_eq!(
            ComputeClass::from_vram_and_backend(0, Backend::Cpu),
            ComputeClass::Lo
        );
        assert_eq!(
            ComputeClass::from_vram_and_backend(7_000, Backend::Cuda),
            ComputeClass::Lo
        );
        assert_eq!(
            ComputeClass::from_vram_and_backend(12_288, Backend::Cuda),
            ComputeClass::Mid
        );
        assert_eq!(
            ComputeClass::from_vram_and_backend(24_576, Backend::Cuda),
            ComputeClass::Hi
        );
        assert_eq!(
            ComputeClass::from_vram_and_backend(80_000, Backend::Cuda),
            ComputeClass::Pro
        );
    }
}
