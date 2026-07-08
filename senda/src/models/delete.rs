use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::types::cache::HFCacheInfo;

use crate::models::local::{
    senda_cache_dir, gguf_metadata_cache_path, huggingface_hub_cache_dir,
    huggingface_identity_for_path, scan_hf_cache_info, split_gguf_base_name,
};
use crate::models::resolve::{
    parse_delete_model_ref, resolve_huggingface_file_from_sibling_entries, DeleteModelRef,
};
use crate::models::usage;

/// Default management/API port a senda runtime listens on. Matches
/// the default in `cli/commands/runtime.rs`. We keep this in sync rather
/// than importing it because the CLI module isn't part of the public API
/// of `models::delete`.
const DEFAULT_RUNTIME_API_PORT: u16 = 3131;

#[derive(Debug)]
pub struct DeleteResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub removed_metadata_files: usize,
    pub removed_usage_records: usize,
}

pub async fn resolve_model_identifier(identifier: &str) -> Result<Vec<PathBuf>> {
    match parse_delete_model_ref(identifier).await? {
        DeleteModelRef::LocalStem(stem) => {
            let path = crate::models::find_model_path(&stem);
            if !path.exists() {
                bail!("Model not found: {}", identifier);
            }
            let mut resolved = BTreeSet::from([normalize_path(&path)]);
            if let Some(cache_info) = scan_hf_cache_info(&huggingface_hub_cache_dir()) {
                resolved.extend(find_related_hf_cache_paths(&cache_info, &path));
            }
            Ok(resolved.into_iter().collect())
        }
        DeleteModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => resolve_cached_hf_ref(&repo, revision.as_deref(), &file)
            .await
            .with_context(|| format!("Resolve installed model ref {identifier}")),
    }
}

fn normalized_gguf_stem(stem: &str) -> &str {
    let stem = stem.strip_suffix(".gguf").unwrap_or(stem);
    split_gguf_base_name(stem).unwrap_or(stem)
}

async fn resolve_cached_hf_ref(
    repo_id: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<Vec<PathBuf>> {
    let cache_root = huggingface_hub_cache_dir();
    let Some(cache_info) = scan_hf_cache_info(&cache_root) else {
        bail!("Model not found: {repo_id}");
    };

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model || repo.repo_id != repo_id {
            continue;
        }
        for cached_revision in &repo.revisions {
            if revision.is_some_and(|requested| {
                requested != cached_revision.commit_hash
                    && !cached_revision.refs.iter().any(|r| r == requested)
            }) {
                continue;
            }
            let sibling_entries: Vec<(String, Option<u64>)> = cached_revision
                .files
                .iter()
                .map(|entry| {
                    let size = std::fs::metadata(&entry.file_path)
                        .ok()
                        .map(|meta| meta.len());
                    (entry.file_name.clone(), size)
                })
                .collect();
            let resolved_file = resolve_huggingface_file_from_sibling_entries(
                repo_id,
                revision.or_else(|| cached_revision.refs.first().map(String::as_str)),
                file,
                &sibling_entries,
            )
            .await?;
            if !resolved_file.ends_with(".gguf") {
                bail!("Delete only supports GGUF models: {repo_id}");
            }
            let expected = normalized_gguf_stem(&resolved_file);
            let mut matches: Vec<PathBuf> = cached_revision
                .files
                .iter()
                .filter(|entry| entry.file_name.ends_with(".gguf"))
                .filter(|entry| {
                    normalized_gguf_stem(&entry.file_name).eq_ignore_ascii_case(expected)
                })
                .map(|entry| entry.file_path.clone())
                .collect();
            if !matches.is_empty() {
                matches.sort();
                return Ok(matches);
            }
        }
    }

    bail!("Model not found: {repo_id}")
}

fn find_related_hf_cache_paths(cache_info: &HFCacheInfo, path: &Path) -> Vec<PathBuf> {
    let mut results = BTreeSet::new();
    let Some(identity) = huggingface_identity_for_path(path) else {
        return Vec::new();
    };
    let Some(file_name) = Path::new(&identity.file)
        .file_name()
        .and_then(|value| value.to_str())
    else {
        return Vec::new();
    };
    let expected = normalized_gguf_stem(file_name);

    for repo in &cache_info.repos {
        use hf_hub::RepoType;
        if repo.repo_type != RepoType::Model || repo.repo_id != identity.repo_id {
            continue;
        }
        for revision in &repo.revisions {
            if revision.commit_hash != identity.revision {
                continue;
            }
            for file in &revision.files {
                if !file.file_name.ends_with(".gguf") {
                    continue;
                }
                if normalized_gguf_stem(&file.file_name).eq_ignore_ascii_case(expected) {
                    results.insert(file.file_path.clone());
                }
            }
        }
    }

    results.into_iter().collect()
}

pub fn collect_delete_paths(resolved_paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut to_delete: BTreeSet<PathBuf> = BTreeSet::new();
    if resolved_paths.is_empty() {
        return Ok(Vec::new());
    }

    for path in resolved_paths {
        ensure_delete_path_allowed(path)?;
        to_delete.insert(normalize_path(path));
    }

    let primary_path = &resolved_paths[0];
    if let Some(record) = usage::load_model_usage_record_for_path(primary_path) {
        if record.mesh_managed && !record.managed_paths.is_empty() {
            for p in &record.managed_paths {
                to_delete.insert(normalize_path(p));
            }
        }
    }

    Ok(to_delete.into_iter().collect())
}

pub async fn delete_model_by_identifier(identifier: &str) -> Result<DeleteResult> {
    let resolved_paths = resolve_model_identifier(identifier).await?;

    if resolved_paths.is_empty() {
        bail!("Model not found: {}", identifier);
    }

    let all_paths = collect_delete_paths(&resolved_paths)?;

    if all_paths.is_empty() {
        bail!(
            "No GGUF files found at resolved path: {}",
            resolved_paths[0].display()
        );
    }

    // Before touching the filesystem, ask any locally running runtime to
    // drop the model from its registry. On macOS / Linux the open mmap
    // doesn't block unlink, but on Windows the GGUF stays locked
    // (`ERROR_SHARING_VIOLATION`, os error 32) until llama-server exits.
    // Doing the unload in-process here means every caller (CLI users,
    // scripts, third-party sidecars) gets the same correct behaviour —
    // the dashboard's own controller-side mitigation can then go away.
    //
    // Best-effort: a 404 ("not loaded") or a missing runtime is fine; we
    // still proceed to remove_file. The only thing we don't want is to
    // skip remove_file because the runtime happened to be unreachable.
    let unload_targets = collect_unload_candidates(identifier, &all_paths);
    if !unload_targets.is_empty() {
        try_unload_from_running_runtime(&unload_targets).await;
    }

    let mut reclaimed_bytes: u64 = 0;
    let mut removed_metadata_files: usize = 0;
    let mut removed_usage_records: usize = 0;
    let mut deleted_paths: Vec<PathBuf> = Vec::new();
    let mut removed_record_paths = BTreeSet::new();

    for path in &all_paths {
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(path) {
                reclaimed_bytes += meta.len();
            }
            if let Err(err) = std::fs::remove_file(path) {
                if is_file_in_use_error(&err) {
                    bail!(
                        "Cannot remove {}: file is still in use by a senda runtime. \
                         Stop the service (`senda service stop`) or unload the model \
                         (`senda runtime models drop {}`) and retry. (os error {})",
                        path.display(),
                        identifier,
                        err.raw_os_error().unwrap_or(0),
                    );
                }
                return Err(anyhow::Error::new(err))
                    .with_context(|| format!("Remove {}", path.display()));
            }
            deleted_paths.push(path.clone());

            if let Some(metadata_path) = gguf_metadata_cache_path(path) {
                if metadata_path.exists() {
                    std::fs::remove_file(&metadata_path).with_context(|| {
                        format!("Remove metadata cache {}", metadata_path.display())
                    })?;
                    removed_metadata_files += 1;
                }
            }

            prune_empty_ancestors(path, &huggingface_hub_cache_dir());
        }
    }

    for path in &all_paths {
        if let Some(record) = load_model_usage_record_for_path(path) {
            let usage_dir = usage::model_usage_cache_dir();
            let record_path = usage::usage_record_path(&usage_dir, &record.lookup_key);
            if removed_record_paths.insert(record_path.clone()) && record_path.exists() {
                std::fs::remove_file(&record_path)
                    .with_context(|| format!("Remove usage record {}", record_path.display()))?;
                removed_usage_records += 1;
            }
        }
    }

    Ok(DeleteResult {
        deleted_paths,
        reclaimed_bytes,
        removed_metadata_files,
        removed_usage_records,
    })
}

/// Load a model usage record for a given path.
fn load_model_usage_record_for_path(path: &std::path::Path) -> Option<usage::ModelUsageRecord> {
    usage::load_model_usage_record_for_path(path)
}

fn ensure_delete_path_allowed(path: &Path) -> Result<()> {
    let normalized = normalize_path(path);
    let hf_root = normalize_path(&huggingface_hub_cache_dir());
    let mesh_root = normalize_path(&senda_cache_dir());
    if normalized.starts_with(&hf_root) || normalized.starts_with(&mesh_root) {
        Ok(())
    } else {
        bail!(
            "Deletion target outside known model roots: {}",
            normalized.display()
        );
    }
}

/// Prune empty ancestor directories up to (but not including) stop_at.
fn prune_empty_ancestors(path: &std::path::Path, stop_at: &std::path::Path) {
    let stop_at = normalize_path(stop_at);
    let mut current = path.parent().map(normalize_path);
    while let Some(dir) = current {
        if dir == stop_at {
            break;
        }
        let Ok(mut entries) = std::fs::read_dir(&dir) else {
            break;
        };
        if entries.next().is_some() {
            break;
        }
        if std::fs::remove_dir(&dir).is_err() {
            break;
        }
        current = dir.parent().map(normalize_path);
    }
}

fn normalize_path(path: &std::path::Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Build the list of names to ask the runtime to unload before we delete.
/// The runtime's registry keys models by their loaded name, which is
/// usually the catalog stem (e.g. `Mixtral-8x7B-Instruct-v0.1-Q4_K_M`)
/// but can also be the user-supplied identifier verbatim. We try both —
/// the runtime route returns 404 for misses, which we ignore.
fn collect_unload_candidates(identifier: &str, paths: &[PathBuf]) -> Vec<String> {
    let mut names = BTreeSet::new();
    let trimmed = identifier.trim();
    if !trimmed.is_empty() {
        names.insert(trimmed.to_string());
    }
    for path in paths {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            names.insert(normalized_gguf_stem(stem).to_string());
        }
    }
    names.into_iter().collect()
}

/// Best-effort unload. Discovers any co-located runtime via the runtime
/// root and issues `DELETE /api/runtime/models/<name>` for each candidate
/// name. Failures (no runtime, transport error, 404, 5xx) are swallowed
/// so the deletion still proceeds — by design, this is a hint, not a
/// gate.
async fn try_unload_from_running_runtime(names: &[String]) {
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    if let Ok(runtime_root) = crate::runtime::instance::runtime_root() {
        if let Ok(snapshots) =
            crate::runtime::instance::scan_local_instances(&runtime_root, std::process::id()).await
        {
            for snap in &snapshots {
                if let Some(port) = snap.api_port {
                    ports.insert(port);
                }
            }
        }
    }
    if ports.is_empty() {
        // Fall back to the documented default. Common case: user runs
        // `senda models delete` while the desktop service is up but
        // no owner.json was readable for some reason (sandboxing,
        // permissions, race during startup).
        ports.insert(DEFAULT_RUNTIME_API_PORT);
    }

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return,
    };

    for port in ports {
        for name in names {
            let encoded = percent_encode_path_segment(name);
            let url = format!("http://127.0.0.1:{port}/api/runtime/models/{encoded}");
            let _ = client.delete(&url).send().await;
        }
    }

    // Windows occasionally needs a moment for the OS to release the
    // mmap handle after `llama-server` shuts down. 250 ms is empirically
    // enough on the systems we've reproduced this on; we only pay it
    // once per delete, and only when there's something to unload.
    #[cfg(windows)]
    if !names.is_empty() {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// Percent-encode a string for use as a URL path segment. Mirrors the
/// helper in `cli/commands/runtime.rs`; copied rather than re-exported
/// to keep the CLI module out of the public API surface of `models`.
fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b => {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

/// Detect Windows `ERROR_SHARING_VIOLATION` (os error 32) on `remove_file`.
/// We avoid `cfg(windows)` here so the logic is also exercised by tests on
/// non-Windows hosts via `Error::from_raw_os_error(32)`.
fn is_file_in_use_error(err: &std::io::Error) -> bool {
    err.raw_os_error() == Some(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_gguf_stem_collapses_split_shards() {
        assert_eq!(
            normalized_gguf_stem("GLM-5-UD-IQ2_XXS-00001-of-00006.gguf"),
            "GLM-5-UD-IQ2_XXS"
        );
        assert_eq!(normalized_gguf_stem("Qwen3-8B-Q4_K_M"), "Qwen3-8B-Q4_K_M");
    }

    #[test]
    fn collect_unload_candidates_includes_identifier_and_path_stems() {
        let identifier = "Mixtral-8x7B-Instruct-v0.1-Q4_K_M";
        let path = PathBuf::from("/tmp/Mixtral-8x7B-Instruct-v0.1.Q4_K_M-00001-of-00002.gguf");
        let names = collect_unload_candidates(identifier, &[path]);
        assert!(names.iter().any(|n| n == identifier));
        // The split-shard suffix should be normalised away so the
        // runtime's loaded name (sans -NNNNN-of-MMMMM) matches.
        assert!(names
            .iter()
            .any(|n| n == "Mixtral-8x7B-Instruct-v0.1.Q4_K_M"));
    }

    #[test]
    fn collect_unload_candidates_ignores_empty_identifier() {
        let path = PathBuf::from("/tmp/Llama-3.2-1B-Q4_K_M.gguf");
        let names = collect_unload_candidates("   ", &[path]);
        assert_eq!(names, vec!["Llama-3.2-1B-Q4_K_M".to_string()]);
    }

    #[test]
    fn percent_encode_keeps_unreserved_and_escapes_others() {
        assert_eq!(percent_encode_path_segment("Llama-3.2_1B"), "Llama-3.2_1B");
        // Space → %20, slash → %2F. We rely on this so the runtime
        // route's URL decoder sees exactly the original model name.
        assert_eq!(percent_encode_path_segment("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn is_file_in_use_recognises_os_error_32() {
        let err = std::io::Error::from_raw_os_error(32);
        assert!(is_file_in_use_error(&err));
        let other = std::io::Error::from_raw_os_error(2); // ENOENT
        assert!(!is_file_in_use_error(&other));
    }
}
