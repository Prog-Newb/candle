use crate::error::KernelError;
use crate::wrappers::SendSyncModule;
use rocm_rs::hip::Device;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Single unified cache for compiled kernel modules.
///
/// Combines the functionality of the old CacheManager (disk cache)
/// and KernelManager (module cache) into one simpler struct.
pub struct KernelCache {
    cache_dir: PathBuf,
    arch: String,
    rocm_version: String,
    modules: Mutex<HashMap<&'static str, Arc<SendSyncModule>>>,
}

impl KernelCache {
    /// Create a new KernelCache for the given device
    pub fn new(device: &Device) -> Result<Self, KernelError> {
        let arch = detect_gpu_arch(device)?;
        let rocm_version = detect_rocm_version()?;
        let cache_dir = get_cache_dir()?;

        // Create cache directory structure: ~/.cache/candle-rocm/{arch}-{rocm_version}/
        let arch_version = format!("{}-{}", arch, rocm_version);
        let kernel_dir = cache_dir.join(&arch_version);
        fs::create_dir_all(&kernel_dir).map_err(|e| {
            KernelError::Io(format!(
                "Failed to create cache directory {}: {}",
                kernel_dir.display(),
                e
            ))
        })?;

        Ok(Self {
            cache_dir: kernel_dir,
            arch,
            rocm_version,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// Get or compile a kernel module.
    ///
    /// This method checks the in-memory cache first, then the disk cache,
    /// and compiles from source if needed.
    pub fn get_or_load(
        &self,
        name: &'static str,
        source: &'static str,
    ) -> Result<Arc<SendSyncModule>, KernelError> {
        // Check in-memory cache first
        {
            let modules = self
                .modules
                .lock()
                .map_err(|_| KernelError::Internal("Failed to lock modules cache".to_string()))?;
            if let Some(module) = modules.get(name) {
                return Ok(module.clone());
            }
        }

        // Compute hash of source to version the cache
        let source_hash = compute_source_hash(source);
        let cache_file = self.cache_dir.join(format!("{}_{}.cso", name, source_hash));

        // Try to load from disk cache or compile
        let binary = if cache_file.exists() {
            fs::read(&cache_file).map_err(|e| {
                KernelError::Io(format!(
                    "Failed to read cached binary {}: {}",
                    cache_file.display(),
                    e
                ))
            })?
        } else {
            let binary = compile_kernel(name, source, &self.arch, &cache_file)?;
            fs::write(&cache_file, &binary).map_err(|e| {
                KernelError::Io(format!(
                    "Failed to write cache file {}: {}",
                    cache_file.display(),
                    e
                ))
            })?;
            binary
        };

        // Load module from binary
        let module = SendSyncModule::load_data(&binary).map_err(|e| {
            KernelError::Compilation(format!(
                "Failed to load module {} from compiled binary: {}",
                name, e
            ))
        })?;

        let module = Arc::new(module);

        // Store in memory cache
        {
            let mut modules = self
                .modules
                .lock()
                .map_err(|_| KernelError::Internal("Failed to lock modules cache".to_string()))?;
            modules.insert(name, module.clone());
        }

        Ok(module)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get GPU architecture
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Get ROCm version
    pub fn rocm_version(&self) -> &str {
        &self.rocm_version
    }
}

/// Detect the GPU architecture (e.g., "gfx908", "gfx90a", "gfx942")
fn detect_gpu_arch(_device: &Device) -> Result<String, KernelError> {
    // First try to get from environment variable (useful for testing/build machines)
    if let Ok(arch) = std::env::var("CANDLE_ROCM_ARCH") {
        return Ok(arch);
    }

    // The app sidecar may launch outside the build shell; preserve the architecture selected
    // for this ROCm build.
    if let Some(arch) = option_env!("CANDLE_ROCM_ARCH") {
        return Ok(arch.to_owned());
    }

    // Try to use rocminfo to detect the architecture
    match Command::new("rocminfo").arg("-a").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Look for "Name:" line with gfxXXXX
            for line in stdout.lines() {
                if line.contains("Name:") && line.contains("gfx") {
                    if let Some(start) = line.find("gfx") {
                        let arch = &line[start..];
                        // Extract just the gfxXXXX part
                        let end = arch
                            .find(|c: char| !c.is_alphanumeric())
                            .unwrap_or(arch.len());
                        return Ok(arch[..end].to_string());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to run rocminfo: {}", e);
        }
    }

    // Try hipcc to get default arch
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(_) => {
            eprintln!("Warning: Could not detect GPU architecture, defaulting to gfx908");
            Ok("gfx908".to_string())
        }
        Err(e) => Err(KernelError::Compilation(format!(
            "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_ARCH environment variable",
            e
        ))),
    }
}

/// Detect ROCm version
fn detect_rocm_version() -> Result<String, KernelError> {
    // Try to get from environment variable first
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok(version);
    }

    // Try to get from hipcc --version
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse version from output like "HIP version: 6.1.0"
            for line in stdout.lines() {
                if line.contains("HIP version:") || line.contains("HIP_VERSION:") {
                    if let Some(v) = line.split(':').nth(1) {
                        let version = v.trim().split('.').take(2).collect::<Vec<_>>().join(".");
                        return Ok(version);
                    }
                }
            }
            // If we can't parse, return a default
            Ok("6.0".to_string())
        }
        Err(e) => Err(KernelError::Compilation(format!(
            "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_VERSION environment variable",
            e
        ))),
    }
}

/// Get the base cache directory
fn get_cache_dir() -> Result<PathBuf, KernelError> {
    let home = dirs::cache_dir()
        .or_else(|| std::env::var("HOME").ok().map(PathBuf::from))
        .ok_or_else(|| KernelError::Internal("Could not determine cache directory".to_string()))?;

    Ok(home.join("candle-rocm"))
}

/// Compute a hash of the source code
fn compute_source_hash(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let result = hasher.finalize();
    // Use first 16 characters of hex as hash
    format!("{:x}", result)[..16].to_string()
}
fn extract_hip_code_object(
    bundle_file: &Path,
    output_file: &Path,
    arch: &str,
) -> Result<(), KernelError> {
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";

    let data = fs::read(bundle_file).map_err(|e| {
        KernelError::Io(format!(
            "Failed to read HIP offload bundle {}: {}",
            bundle_file.display(),
            e
        ))
    })?;

    if data.len() < 32 || &data[..MAGIC.len()] != MAGIC {
        return Err(KernelError::Compilation(format!(
            "Invalid HIP offload bundle in {}",
            bundle_file.display()
        )));
    }

    // Bundle format:
    // 0x00  24-byte magic
    // 0x18  u64 bundle count
    // Each bundle:
    //   u64 offset
    //   u64 size
    //   u64 identifier length
    //   identifier bytes
    let read_u64 = |offset: usize| -> Result<u64, KernelError> {
        let end = offset.checked_add(8).ok_or_else(|| {
            KernelError::Compilation("HIP bundle offset overflow".into())
        })?;

        if end > data.len() {
            return Err(KernelError::Compilation(
                "HIP bundle header is truncated".into(),
            ));
        }

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[offset..end]);
        Ok(u64::from_le_bytes(bytes))
    };

    let bundle_count = read_u64(24)? as usize;
    let mut cursor = 32usize;

    let wanted = format!("hipv4-amdgcn-amd-amdhsa--{}", arch);

    for _ in 0..bundle_count {
        let offset = read_u64(cursor)? as usize;
        cursor += 8;

        let size = read_u64(cursor)? as usize;
        cursor += 8;

        let id_len = read_u64(cursor)? as usize;
        cursor += 8;

        let id_end = cursor.checked_add(id_len).ok_or_else(|| {
            KernelError::Compilation("HIP bundle ID overflow".into())
        })?;

        if id_end > data.len() {
            return Err(KernelError::Compilation(
                "HIP bundle ID extends past file".into(),
            ));
        }

        let id = String::from_utf8_lossy(&data[cursor..id_end]);
        cursor = id_end;

        if id == wanted {
            let end = offset.checked_add(size).ok_or_else(|| {
                KernelError::Compilation(
                    "HIP code object offset overflow".into(),
                )
            })?;

            if end > data.len() {
                return Err(KernelError::Compilation(format!(
                    "HIP code object for {} extends past bundle",
                    wanted
                )));
            }

            let code_object = &data[offset..end];

            // AMDGPU code objects are ELF files.
            if code_object.len() < 4 || &code_object[..4] != b"\x7fELF" {
                return Err(KernelError::Compilation(format!(
                    "HIP bundle entry {} is not an ELF code object",
                    wanted
                )));
            }

            fs::write(output_file, code_object).map_err(|e| {
                KernelError::Io(format!(
                    "Failed to write HIP code object {}: {}",
                    output_file.display(),
                    e
                ))
            })?;

            return Ok(());
        }
    }

    Err(KernelError::Compilation(format!(
        "HIP code object {} not found in bundle",
        wanted
    )))
}


/// Compile a kernel using hipcc
fn compile_kernel(
    name: &str,
    source: &str,
    arch: &str,
    output_path: &Path,
) -> Result<Vec<u8>, KernelError> {
    let temp_dir = std::env::temp_dir();
    let source_hash = compute_source_hash(source);

    let source_file =
        temp_dir.join(format!("candle_{}_{}.hip", name, source_hash));
    let obj_file =
        temp_dir.join(format!("candle_{}_{}.o", name, source_hash));
    let hsaco_file =
        temp_dir.join(format!("candle_{}_{}.hsaco", name, source_hash));

    let _cleanup = TempFileCleanup {
        files: vec![
            source_file.clone(),
            obj_file.clone(),
            hsaco_file.clone(),
        ],
    };

    fs::write(&source_file, source).map_err(|e| {
        KernelError::Io(format!(
            "Failed to write source file {}: {}",
            source_file.display(),
            e
        ))
    })?;

    // Step 1: Compile HIP to a device offload bundle.
    //
    // ROCm 7.x on Windows produces a __CLANG_OFFLOAD_BUNDLE__
    // containing the host/device entries. We extract the
    // gfx-specific ELF directly in Rust below.
    let rocm_path = std::env::var("HIP_PATH")
        .ok()
        .or_else(|| option_env!("HIP_PATH").map(str::to_owned))
        .or_else(|| std::env::var("ROCM_PATH").ok())
        .or_else(|| option_env!("ROCM_PATH").map(str::to_owned))
        .ok_or_else(|| {
            KernelError::Compilation(
                "HIP_PATH or ROCM_PATH must identify the ROCm SDK root".to_owned(),
            )
        })?;
    let rocm_device_lib_path = std::env::var("HIP_PATH")
        .ok()
        .or_else(|| option_env!("HIP_PATH").map(str::to_owned))
        .or_else(|| std::env::var("ROCM_PATH").ok())
        .or_else(|| option_env!("ROCM_PATH").map(str::to_owned))
        .map(|root| {
            PathBuf::from(root)
                .join("lib")
                .join("llvm")
                .join("amdgcn")
                .join("bitcode")
                .to_string_lossy()
                .into_owned()
        })
        .ok_or_else(|| {
            KernelError::Compilation(
                "ROCM_PATH or HIP_PATH must identify the ROCm SDK root".to_owned(),
            )
        })?;

    let output = Command::new("hipcc")
        .args(&[
            "--cuda-device-only",
            &format!("--offload-arch={}", arch),
            "-O3",
            &format!("--rocm-path={rocm_path}"),
            &format!("--rocm-device-lib-path={rocm_device_lib_path}"),
            "-c",
            "-o",
            obj_file.to_str().unwrap(),
            source_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!(
                "Failed to execute hipcc: {}",
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        return Err(KernelError::Compilation(format!(
            "hipcc compilation failed for {}:\n{}",
            name, stderr
        )));
    }

    // Step 2: Extract the gfx-specific ELF directly from
    // the __CLANG_OFFLOAD_BUNDLE__ produced by hipcc.
    extract_hip_code_object(
        &obj_file,
        &hsaco_file,
        arch,
    )?;

    // Step 3: Read the extracted ELF code object.
    let binary = fs::read(&hsaco_file).map_err(|e| {
        KernelError::Io(format!(
            "Failed to read code object {}: {}",
            hsaco_file.display(),
            e
        ))
    })?;

    // Step 4: Write the final code object to the Candle cache.
    fs::write(output_path, &binary).map_err(|e| {
        KernelError::Io(format!(
            "Failed to write cache file {}: {}",
            output_path.display(),
            e
        ))
    })?;

    Ok(binary)
}

/// Find an ROCm tool using hipcc
fn find_rocm_tool(tool_name: &str) -> Result<String, KernelError> {
    if let Ok(hip_path) = std::env::var("HIP_PATH") {
        let candidate = PathBuf::from(&hip_path)
            .join("lib")
            .join("llvm")
            .join("bin")
            .join(if cfg!(windows) {
                format!("{}.exe", tool_name)
            } else {
                tool_name.to_string()
            });

        if candidate.exists() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }

    let output = Command::new("hipcc")
        .args(&["--print-prog-name", tool_name])
        .output()
        .map_err(|e| KernelError::Compilation(format!("Failed to run hipcc: {}", e)))?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() && PathBuf::from(&path).exists() {
            return Ok(path);
        }
    }

    Err(KernelError::Compilation(format!(
        "{} not found via HIP_PATH or hipcc. Is ROCm installed?",
        tool_name
    )))
}

/// Helper struct to clean up temporary files
struct TempFileCleanup {
    files: Vec<PathBuf>,
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        for file in &self.files {
            let _ = fs::remove_file(file);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_hash() {
        let source1 = "__global__ void test() {}";
        let source2 = "__global__ void test() {}";
        let source3 = "__global__ void test2() {}";

        let hash1 = compute_source_hash(source1);
        let hash2 = compute_source_hash(source2);
        let hash3 = compute_source_hash(source3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
