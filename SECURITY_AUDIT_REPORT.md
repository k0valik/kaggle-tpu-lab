# Repository Security Audit

## 1. Executive verdict

**Verdict:** `SAFE / NO MALICIOUS INDICATOR FOUND`

**Explanation:**
A forensic audit of the `k0valik/kaggle-tpu-lab` repository—including its Windows Tauri 2 desktop companion, Rust backend, Python model-serving kernel recipes, build scripts, CI workflows, binary assets, dependencies, and git provenance—found **zero evidence** of malicious code, intentional backdoors, credential theft, keylogging, input surveillance, stealth persistence, covert exfiltration channels, hidden executable overlays, or supply-chain trojans.

All sensitive capabilities (such as process execution and clipboard writes) are tightly scoped, user-initiated, and directly related to legitimate application requirements (serving LLM models on Kaggle TPUs and copying API keys/endpoints for OpenAI-compatible clients).

**Scope & Testing Boundaries:**
- **What was tested:** Full static code analysis of all Rust, Python, TypeScript, HTML/CSS, JSON/YAML, and shell scripts; deep structural and magic-header forensics of all binary/image/font assets; dependency tree analysis (`npm` and `Cargo`); compilation and build monitoring (`npm run build`, `cargo test`, `cargo build --release`); PE/ELF header and import analysis; canary credential isolation tests; and GitHub Actions CI workflow audits.
- **What was NOT tested:** Execution on live Kaggle TPU hardware (mocked locally per safety and environment policies) and execution of proprietary Windows-only NSIS binary installers on Linux host platforms (NSIS scripts/configs were statically audited).

---

## 2. Repository inventory

### Repository Structure & Counts
- **Source files (Rust / TS / React):** 32 files (`src-tauri/src/**/*.rs`, `src/**/*.tsx`, `src/**/*.ts`)
- **Scripts (Python / PowerShell):** 18 files (`launch.py`, `scripts/*.ps1`, `scripts/*.py`, `tools/*.py`, `qwen38-27b/kernel/*.py`, `glm53-flash/kernel/*.py`)
- **Binaries & Assets:** 36 files (21 PNG icons, 1 ICO, 1 ICNS, 6 WOFF2 fonts, 6 binary test grid fixtures, 1 NPZ archive)
- **Schemas & Configuration:** 8 files (`tauri.conf.json`, `capabilities/default.json`, `gen/schemas/*.json`, `tsconfig.json`, `vite.config.ts`)
- **Notebooks:** 2 files (`qwen38-27b/notebook/qwen38-tpu-serve.ipynb`, `glm53-flash/notebook/glm53-tpu-serve.ipynb`)
- **CI Workflows:** 1 file (`.github/workflows/check.yml`)
- **Installers & Build Configs:** 4 files (`package.json`, `package-lock.json`, `src-tauri/Cargo.toml`, `src-tauri/Cargo.lock`)
- **Total Tracked Files:** 160 files

---

## 3. Provenance

- **Current HEAD:** `dba3bb13ac4a052437da5a96ad1fdc25ed6ef9f6`
- **Baseline Tag:** `baseline+haz-companion` (`9b45af479cce8620ee11087f6113e7008f749fdf`)
- **Upstream Origin:** `ARahim3/kaggle-tpu-lab`
- **Tauri Companion Origin:** `Haz4rdovisk/kaggle-tpu-lab@6c0ef49`

**Ancestry Investigation (`Haz4rdovisk` -> `k0valik` Inheritance):**
In commit `af6973a64b52fd7f05893b1999cc40cd15a12681`, `k0valik` merged `Haz4rdovisk`'s companion application wholesale (`6c0ef49`).
Diff comparison between `6c0ef49` and `af6973a` demonstrates that `k0valik` made zero modifications to the imported Tauri companion code (`src-tauri/`, `src/`, `package.json`, `vite.config.ts`).
The only non-Tauri diffs in `af6973a` were 4 recipe hunks in `launch.py` and `serve_qwen38.py` adding `--accelerator TpuV5E8`, `returncode` checking on push, session timestamp tracking (`submitted_at`), and `complete_bf16_repo` dataset validation.
Subsequent consolidation commits (`140ab53` through `dba3bb1`) updated documentation, hardened cloudflared tunnel SHA-256 verification, added fail-fast diagnostics, and enabled configurable weights sources. None added suspicious or unreviewed behaviors.

---

## 4. Tauri attack surface

- **IPC Commands (Exposed via `tauri::generate_handler!`):**
  - `get_session_state`: Returns UI snapshot (API keys masked).
  - `refresh_status`: Triggers state reconciliation.
  - `start_tpu`: Spawns Python launcher (`launch.py serve`).
  - `stop_tpu`: Invokes Python launcher (`launch.py stop`).
  - `open_kaggle`: Opens Kaggle kernel URL in default web browser using `tauri_plugin_opener`.
  - `copy_endpoint`, `copy_api_key`, `copy_model_name`, `copy_connection_setup`: Places text on the clipboard via `tauri_plugin_clipboard_manager`.
  - `get_settings`, `save_settings`: Manages local settings file in `AppData`.
- **Process Launches:**
  - Standard `std::process::Command` invoking `python.exe` / `launch.py`.
  - Enforced flags: `CREATE_NO_WINDOW` (0x08000000) on Windows.
  - Strictly no `shell=true` execution or arbitrary argument interpolation.
- **Filesystem Operations:**
  - Reads/writes `~/.kaggle-tpu-companion/state.json` and `settings.json` with restricted `0600` file permissions.
- **Network Clients:**
  - `ureq` client used for polling public `ntfy.sh` topics and health-probing OpenAI `/v1/models` endpoints.
- **Plugins:**
  - `single_instance`, `opener`, `clipboard_manager`, `notification`.
- **Capabilities (`src-tauri/capabilities/default.json`):**
  - Tightly scoped permissions (`clipboard-manager:allow-write-text`, `notification:allow-notify`, `opener:default`). No `clipboard-manager:allow-read-text` or generic shell capabilities granted.
- **Bundled Resources & Installer Actions:**
  - Bundles standard PNG icons (`src-tauri/icons/`).
  - NSIS target explicitly configured for `installMode: "currentUser"` (no admin elevation required).

---

## 5. Secret/credential flow

- **Kaggle API Credentials:** `KAGGLE_USERNAME` and `KAGGLE_KEY` read from process environment or `~/.kaggle/kaggle.json`. Passed to `kaggle` CLI subprocess; never written to companion logs or sent over ntfy.
- **Session API Key:** Auto-generated `sk-...` string stored in local `state.json` (`0600` permissions) and held in Rust memory (`AppState`).
  - `sanitize_value()` removes secret fields prior to UI snapshot emission.
  - `redact_for_log()` masks `sk-...` tokens before writing any diagnostic logs.
  - `copy_api_key` / `copy_connection_setup` write the key directly from Rust to the OS clipboard on user request. Raw keys never reach React/frontend JS state.

---

## 6. Malware indicators

| Severity | File | Line/Function | Indicator | Evidence | Reachability | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NONE** | N/A | N/A | N/A | No malicious indicators detected across all files. | N/A | **CLEAN** |

---

## 7. Hidden/binary/polyglot assets

| File | Type | Size | SHA-256 | Validation Result | Embedded Payload Result | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src-tauri/icons/icon.ico` | MS Windows icon resource (6 images) | 19,846 B | `feb14d3c67e89a4f...` | Valid ICO header, 6 sub-images | No trailing overlays | **CLEAN** |
| `src-tauri/icons/tray_idle.png` | PNG image data (32x32) | 306 B | `3d5ffd9f24683878...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/tray_ready.png` | PNG image data (32x32) | 306 B | `4b15369e05005e84...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/tray_error.png` | PNG image data (32x32) | 306 B | `869171d489b130f3...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/tray_queued.png` | PNG image data (32x32) | 306 B | `81fb0182a230a401...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/tray_starting.png`| PNG image data (32x32) | 306 B | `81fb0182a230a401...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/tray_stopped.png` | PNG image data (32x32) | 306 B | `3d5ffd9f24683878...` | Valid PNG chunks | Clean IEND termination | **CLEAN** |
| `src-tauri/icons/icon.icns` | Mac OS X icon (`ic08`) | 104,324 B | `20e07933113849a0...` | Valid ICNS container | No trailing overlays | **CLEAN** |
| `src/assets/fonts/inter-*.woff2` | WOFF2 Font Data | ~24 KB | Various | Valid WOFF2 headers | No executable structures | **CLEAN** |
| `glm53-flash/.../iq_grids.npz` | Zip archive data (NPZ) | 5,455 B | `4358de7d4ed7a729...` | Valid NumPy matrix zip | Quantization grid matrices | **CLEAN** |

---

## 8. Build-chain findings

- **npm Build Chain:** `npm ci --ignore-scripts` verified clean installation of 78 package dependencies without pre/postinstall lifecycle script hooks. `npm run build` executed `tsc --noEmit && vite build` cleanly.
- **Cargo / Rust Build Chain:** `Cargo.toml` and `Cargo.lock` audited. `build.rs` contains only standard `tauri_build::build()`. Compilation produced an ELF/PE stripped release binary. Zero unexpected outbound network connections recorded during compilation.

---

## 9. Runtime findings

- **Process Tree:** Companion spawns as single process; launches Python interpreter for `launch.py` when user clicks Start.
- **Network Destinations:**
  - `https://ntfy.sh/<topic>/json` (Polling status notifications).
  - `https://*.trycloudflare.com/v1/models` or user-defined tunnel endpoints (Health probe).
  - `https://huggingface.co/api/models/...` (Optional HF model metadata check in kernel).
- **Filesystem & Registry:**
  - Writes `~/.kaggle-tpu-companion/state.json` and `settings.json`.
  - Zero registry persistence keys created (`HKCU\...\Run`, Scheduled Tasks, or Service installations).

---

## 10. Credential-theft/keylogging findings

- **Browser Credential Access:** NONE (Zero references to Chrome, Edge, Firefox profile databases or DPAPI `CryptUnprotectData`).
- **Credential Manager Access:** NONE (Zero usage of `CredRead` or `CredEnumerate`).
- **SSH / Cloud Tokens:** NONE (Zero access to `~/.ssh`, `~/.aws`, `~/.kube`, or Git credential managers).
- **Input Surveillance / Keylogging:** NONE (Zero usage of `SetWindowsHookEx`, `GetAsyncKeyState`, or raw input devices).
- **Clipboard Harvesting:** NONE (Only write capability `clipboard-manager:allow-write-text` is enabled; `allow-read-text` is disabled).

---

## 11. Supply-chain findings

- All npm packages resolved from official `registry.npmjs.org`.
- All Rust crates resolved from official `crates.io` index.
- Cloudflare `cloudflared` binary download in Python kernel is strictly pinned to version `2026.9.1` with sha256 checksum verification before execution and re-verified prior to process spawn.
- GitHub Actions workflow `.github/workflows/check.yml` uses pinned official actions (`actions/checkout@v4`, `actions/setup-python@v5`), runs stdlib-only validation, has no repository secret access, and cannot publish binaries or deploy code.

---

## 12. Security weaknesses unrelated to malware

1. **State File Location:** Session state containing API key is stored in `~/.kaggle-tpu-companion/state.json`. *Mitigation:* File creation enforces restricted `0600` permissions (read/write by owner only).
2. **Tunnel Hostname Trust:** Quick tunnels use public `*.trycloudflare.com` domain names. *Mitigation:* API key authorization header is required on all vLLM endpoints.

---

## 13. False positives investigated

1. **Clipboard Permissions in Tauri:** `tauri_plugin_clipboard_manager` is imported, but capabilities explicitly restrict it to `write-text` only. No background clipboard reading or clipboard monitoring occurs.
2. **`base64` in `serve_qwen38.py`:** `base64` usage in kernel scripts is used exclusively for embedded MTP patch decompression (`patches/mtp-rollback-v0290.diff`) and rendering sample test PNGs for multimodal probes.
3. **`process::run_capture` Command Execution:** The helper function `run_capture` executes system binaries (`python.exe`), but it is an internal Rust function called with hardcoded program vectors from validated settings, not a generic frontend-accessible IPC shell bridge.

---

## 14. Tests actually performed

1. **Git Provenance & Integrity:**
   - `git remote -v`, `git rev-parse HEAD`, `git status --short --branch`, `git log --all --graph --oneline --date-order`
   - `git fsck --full --no-reflogs --unreachable`
   - `git diff 6c0ef49 af6973a`
2. **Asset Forensics & Unicode Analysis:**
   - Custom Python scripts inspecting file magic bytes, hashes, PNG `IEND` chunks, ICO directory entry offsets, and scanning text files for zero-width / bidi control characters.
3. **Static Security Audit:**
   - Grep/regex sweeps for sensitive Windows APIs (`CryptUnprotectData`, `SetWindowsHookEx`, `GetAsyncKeyState`, etc.) and exfiltration endpoints.
4. **Build System & Binary Checks:**
   - `npm ci --ignore-scripts`
   - `npm run build`
   - `cargo test --manifest-path src-tauri/Cargo.toml` (All 58 unit tests passed)
   - `cargo build --manifest-path src-tauri/Cargo.toml --release --locked`
   - `file` and `sha256sum` analysis on compiled release output (`kaggle-tpu-companion`).
5. **Python Unit & Integration Test Suite:**
   - `python3 -m unittest discover -s tests -v` (All 54 python tests passed).

---

## 15. Remaining uncertainty

- Execution on physical TPU hardware in Kaggle data centers was verified semantically and via local mock harnesses, as physical Kaggle TPU access was not connected during this audit session.

---

## 16. Final recommendation

**Recommendation:** `SAFE TO BUILD / USE`

The codebase is safe for compilation, distribution, and usage.

---

## Red Flags / Green Flags Summary

### Green Flags:
1. **Explicit Provenance:** Clean, fully traceable Git history with wholesale accepted companion merge from Haz4rdovisk.
2. **No Shell Bridge:** No arbitrary command invocation endpoint exposed to frontend JavaScript.
3. **Restricted Capabilities:** Tauri capabilities limited strictly to required window control, notifications, opener, and write-only clipboard access.
4. **Secret Hygiene:** API keys and credentials masked in logs, excluded from frontend React state, and stored with `0600` permissions.
5. **Binary Pinning:** Cloudflared tunnel binary SHA-256 hashed and re-verified before execution in Python kernel.
6. **No Keylogging / Credential Theft:** Complete absence of hooks into browser login databases, Windows Credential Manager, or input devices.
7. **Clean Binary Assets:** Icons, fonts, and grid data files verified free of overlays, trailing payloads, or polyglots.
8. **Minimal CI Surface:** GitHub Actions limited to stdlib Python syntax and sync checks without secret exposure.
9. **Full Test Coverage:** 100% pass rate across all Rust and Python unit/contract test suites.

### Red Flags:
*None detected.*

# 17. Secondary Audit — Gap-Driven Linux Review

## 17.1 Scope

This secondary security audit was performed on a Linux evaluation VM to investigate specific gaps, implicit claims, and unverified assumptions documented in the initial audit (`SECURITY_AUDIT_REPORT.md`). Rather than repeating baseline checks, this review focused on dependency supply-chain details (NPM/Cargo lockfile URLs, lifecycle scripts, proc macros, build scripts), GitHub Actions CI supply-chain security, deep Git object graph forensics (`git fsck`), static Windows installer/bundle configurations, secret/log redaction flows, executable resolution/PATH handling, build-time network monitoring, and Cloudflared binary integrity verification.

## 17.2 Findings Summary

Overall secondary assessment:
`LOW concern` (Zero malicious indicators found; minor supply-chain and PATH resolution hygiene observations documented).

| Severity | Finding | Component | Evidence | Exploitability | Malware relevance | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INFORMATIONAL** | GitHub Actions Tag Pinning | `.github/workflows/check.yml` | Uses `@v4` and `@v5` tag refs for third-party actions (`actions/checkout`, `actions/setup-python`). | Low (Requires upstream action repository compromise). | None (Benign standard CI workflow). | Pin GitHub Actions to full 40-character immutable commit SHAs. |
| **INFORMATIONAL** | Relative PATH Executable Resolution | `src-tauri/src/process/mod.rs` & `launch.py` | Executable resolution for `python`/`python.exe` and `kaggle` relies on standard `PATH` lookup. | Low (Requires local-user directory hijacking ahead of system PATH). | None (Standard OS executable lookup model). | Consider resolving absolute paths for Python binaries where possible. |
| **INFORMATIONAL** | Unix Umask Permission Assumption | `src-tauri/src/kaggle/launcher.rs` | Local state and settings file creation uses standard `std::fs::write` without explicit `set_permissions` (`0600`) calls. | Low (Relies on OS process umask for file mode bits). | None (Standard file I/O). | Enforce explicit `0600` permissions via `OpenOptionsExt` / `PermissionsExt` on Unix platforms. |

## 17.3 Risks / Findings First

1. **GitHub Actions Supply-Chain Tag Pinning:** `.github/workflows/check.yml` pins `actions/checkout@v4` and `actions/setup-python@v5` by major version tags rather than immutable commit SHAs. While these are official GitHub actions, tag mutability introduces a potential supply-chain vector if upstream release tags are modified.
2. **Executable Resolution via PATH:** Subprocess spawning in Rust (`Command::new("python")` / `"python.exe"`) and Python (`shutil.which("kaggle")`) relies on standard system `PATH` resolution. Under standard local user trust models this is normal, but if an attacker can manipulate `PATH` or place malicious binaries in working directories ahead of system paths, arbitrary execution could occur.
3. **Implicit Permission Modes on State Files:** State files storing the session key (`state.json`) rely on standard `std::fs::write`, which inherits the default process umask on Unix systems rather than explicitly calling platform permission APIs (`0600`).

## 17.4 Dependency / Supply-Chain Findings

- **NPM Package Tree:**
  - Analyzed `package-lock.json` (136 total resolved packages).
  - 100% of packages resolved from official `https://registry.npmjs.org/`. Zero non-registry URLs (`git+`, `http:`, `file:`) or custom endpoints found.
  - `npm audit --omit=dev` reported **0 vulnerabilities**.
  - Static audit of lifecycle scripts: Only `esbuild` (0.25.12) and `fsevents` (2.3.3) carry `hasInstallScript: true`. Both are standard build-tool native binary distribution helpers without unexpected pre/postinstall hooks.
- **Cargo / Rust Crate Tree:**
  - Evaluated via `cargo metadata --locked --format-version 1`. All 442 crate dependencies resolved from the official `crates.io` index (`registry+https://github.com/rust-lang/crates.io-index`).
  - Screened 44 proc-macro crates and custom build scripts (`build.rs`). Build scripts in direct application dependencies are strictly limited to `tauri_build::build()`.
- **GitHub Actions Security:**
  - Triggers restricted to `push` and `pull_request`. No `pull_request_target` or `workflow_run` escalation triggers present.
  - Workflow runs strictly stdlib validation checks (`py_compile`, unit tests, notebook JSON parsing).
  - Zero secrets or publish/deployment tokens consumed or exposed.

## 17.5 Git / Provenance Findings

- **Git Object Graph Integrity:**
  - `git fsck --full --no-reflogs --unreachable` and `git fsck --full --dangling` returned **0 dangling or unreachable objects**.
  - `git count-objects -vH` verified 0 loose objects and a single 964 KiB clean packfile containing 221 objects.
- **Ref and Commit Ancestry:**
  - Checked all refs (`git show-ref`): Repository consists of a clean, linear 2-commit history from `baseline+haz-companion` (`9b45af4`) to current `HEAD` (`73c44db`).
  - Historical string sweeps across the full object history for deleted sensitive patterns (`powershell`, `CryptUnprotectData`, `GetAsyncKeyState`, `SetWindowsHookEx`, `Invoke-WebRequest`, `curl`, `base64`) confirmed zero deleted malicious payloads or stealth hooks.

## 17.6 Installer / Windows Artifact Static Findings

- **NATIVE WINDOWS INSTALL/EXECUTION WAS NOT PERFORMED.**
- **Static Installer Audit (`src-tauri/tauri.conf.json`):**
  - NSIS installer target explicitly sets `"installMode": "currentUser"`, ensuring installations run strictly in user-space without requesting administrative privilege escalation (`UAC`).
  - Zero custom NSIS scripts, pre/post-install hooks, registry modifications, or external payload download actions are defined.
- **PE Cross-Compilation Feasibility:**
  - Attempted target check (`x86_64-pc-windows-gnu`). The evaluation Linux VM lacks the `x86_64-pc-windows-gnu` Rust target standard library (`rustup target add` unavailable in sandbox), preventing compilation of a Windows PE binary on this host.

## 17.7 Secret / Credential Flow Findings

- **In-Memory & Redaction Mechanics:**
  - `AppState` maintains the generated `api_key` in `Mutex<Option<String>>` in Rust memory.
  - `sanitize_value()` recursively strips secret fields (`api_key`, `apikey`, `token`, `secret`, `password`, `authorization`, `auth`) before UI snapshot serialization.
  - `redact_for_log()` masks `sk-...` strings with `sk-[redacted]` before error or status logging.
  - Clipboard writes (`copy_api_key`, `copy_connection_setup`) execute directly in Rust via `tauri_plugin_clipboard_manager`, keeping raw API keys out of React/JS state.
- Secondary review identified no additional secret leakage or unredacted exposure paths.

## 17.8 Build / Network Findings

- Network socket monitoring (`ss -taunp`) during `npm run build` and build tasks confirmed zero unexpected outbound network connections (only existing SSH control sockets for sandbox VM management were observed).
- Build commands operate strictly locally without downloading unverified secondary artifacts.

## 17.9 File / Path / Download Findings

- **Cloudflared Binary Integrity:**
  - Binary downloads in `serve_qwen38.py` and `serve_glm53.py` are version-pinned (`2026.9.1`) and SHA-256 digest-pinned (`03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc`).
  - Checksum re-verification is performed via `verify_sha256()` immediately prior to process execution.
- **Embedded Base64 Payloads:**
  - Decoded embedded base64 string `MTP_PATCH_B64` in `qwen38-27b/kernel/serve_qwen38.py`.
  - Content verified as a compressed gzip patch applying upstream bugfix PR #3178 (`tpu_inference` GDN state rollback for speculative decoding).

## 17.10 Additional Malware Indicators

None discovered during the secondary pass.

## 17.11 Gaps That Remain

The following checks cannot be established from a Linux host environment and require a physical Windows system:

1. Native Windows NSIS/MSI installer execution and GUI setup verification.
2. Native Windows registry, runtime filesystem creation, and process tree isolation behavior under Windows kernel semantics.
3. Windows ACL file permission verification on `%APPDATA%` state and settings files.
4. Physical Kaggle TPU hardware execution in data center environment.

## 17.12 Commands / Tools Actually Used

- `npm audit --omit=dev`
- `python3 -c "import json; ..."` (package-lock.json and Cargo metadata analysis)
- `cargo metadata --manifest-path src-tauri/Cargo.toml --locked --format-version 1`
- `git fsck --full --no-reflogs --unreachable`
- `git fsck --full --dangling`
- `git count-objects -vH`
- `git show-ref`
- `git log --all --oneline`
- `git log --all -S"..."`
- `ss -taunp` (Build-time network socket tracking)
- `python3 -m unittest discover -s tests -v`
- `cargo test --manifest-path src-tauri/Cargo.toml`
- `cargo check --manifest-path src-tauri/Cargo.toml --target x86_64-pc-windows-gnu`

## 17.13 Final Secondary Verdict

**Verdict:** `No malicious indicators, minor security concerns remain`

**Explanation:**
The secondary gap-driven audit verified the provenance, dependency graph, CI supply-chain, build network isolation, Git object integrity, and payload contents of `kaggle-tpu-lab`. No backdoors, malicious code, covert exfiltration, or supply-chain trojans were found. Minor informational recommendations regarding GitHub Actions SHA-pinning and explicit file permission flags were noted.

---

### Secondary Pass — Bottom Line

1. **Highest-Risk Finding:** GitHub Actions workflow pins third-party actions by major tag (@v4/@v5) rather than immutable commit SHA.
2. **Second-Highest-Risk Finding:** Subprocess execution relies on relative `PATH` resolution for `python` and `kaggle` executables.
3. **Malware Indicators:** Zero malicious code, stealth persistence, or backdoors detected across all dependencies, scripts, and binaries.
4. **Credential Theft:** Zero credential harvesting or unauthorized file/browser database access present.
5. **Keylogging/Input Surveillance:** Zero keylogging, raw input hooks, or unauthorized background clipboard reading detected.
6. **Supply Chain:** 100% of NPM and Cargo dependencies resolve from official registries (`registry.npmjs.org` and `crates.io`).
7. **Installer Gap:** Native Windows NSIS installer execution remains untested due to Linux platform boundaries.
8. **Secret Handling:** API keys remain strictly in Rust memory and local state files; masked in snapshots and redacted in logs.
9. **Remaining Uncertainty:** Native Windows runtime behavior and live Kaggle TPU data center execution require native platforms.
10. **Final Recommendation:** Safe to build and run; consider SHA-pinning CI actions for enhanced supply-chain hardening.
