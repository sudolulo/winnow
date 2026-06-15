# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.19] - 2026-06-15

### Fixed

- **`fetch_face_data` no longer falls back to an arbitrary person's face**: when `person_id` is provided but not found in the Immich `/api/faces` response, the function now returns `None` instead of falling back to `faces[0]`. Previously a Frigate group photo where the target person's face entry was missing would inject a different person's bounding box, causing the wrong face crop to be uploaded as training data.

- **`tracked_assets` PRIMARY KEY now includes `person_name`**: the old `PRIMARY KEY (asset_id, status)` meant that `INSERT OR REPLACE` for person B on an asset already tracked for person A silently overwrote `person_name`, destroying the JOIN between `tracked_assets` and `frigate_files` for person A and breaking quality replacement. The key is now `(asset_id, person_name, status)`, giving each person their own row per asset. Existing databases are migrated automatically on first open.

- **`filter_recent_assets` treats `years=0` as "no age filter"**: previously `years = years or Config.YEARS_FILTER` evaluated `0` as falsy and fell through to the default (10 years), silently discarding all older assets when the user explicitly set `YEARS_FILTER=0`. The check is now `if years is None: years = Config.YEARS_FILTER` followed by an early return for `years=0`.

- **`_is_module_available` now correctly returns False for absent modules**: `importlib.util.find_spec` returns `None` (not raises) for missing top-level modules, so the previous `try: find_spec(); return True` always reported modules as installed. Fixed to `return find_spec(...) is not None`, ensuring `is_embedding_available()` returns False when InsightFace or onnxruntime are not installed.

- **Error handler in `execute_jobs` uses `asset.get("id")` instead of `asset["id"]`**: a malformed asset dict missing the `"id"` key would cause a secondary `KeyError` inside the `except` block, propagating uncaught out of `execute_jobs()` and aborting the run mid-job. Changed to `asset.get("id", "<unknown>")`.

- **HTTP 422 now triggers `mark_rejected`**: only `HTTP 400` with `"face"` in the body triggered permanent rejection; `HTTP 422` (Unprocessable Entity) left the asset untracked and caused it to be re-selected and re-attempted on every future run. Both codes are now treated as permanent rejections.

- **Frigate filename reconciliation sort is now deterministic**: `sorted(new_files, key=_ts)` sorted a `set` — when `_ts()` returns `0.0` for non-matching filenames, Python's stable sort preserves the set's hash-randomised input order, producing non-deterministic `asset_id → frigate_filename` mappings. Changed the key to `lambda f: (_ts(f), f)` so equal-timestamp files sort alphabetically.

### Changed

- **`_resolve_strategy` uses `_getenv_optional_int("LIMIT")`**: the inline `os.environ.get("LIMIT", "").strip()` + `int()` + `logger.warning` block in `jobs.py` re-implemented the logic already in `_getenv_num`. A new `_getenv_optional_int` helper (delegating to `_getenv_num(name, None, int)`) replaces the duplicate, consolidating LIMIT parse warnings with the rest of the env-var helpers.

- **`frigate_api.py` uses a shared `_get_frigate_url()` accessor**: `os.environ.get("FRIGATE_URL", "").rstrip("/")` was copy-pasted into all four public functions. A private helper eliminates the duplication so URL normalization is defined once.

- **`blur_score_from_image()` extracted to `quality.py`**: the time-spread blur-score fallback in `execute_jobs` (resize to 1440px, RGB convert, `assess_quality`) is now a shared `blur_score_from_image(img, max_dim=1440)` helper. Both the executor and any future callers use the same cap and error handling so the score scale can't silently diverge between code paths.

## [0.5.18] - 2026-06-15

### Fixed

- **`scripts/benchmark.py` now uses `_getenv_bool` for `FORCE_CPU`**: the script retained an inline `os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes")` pattern in `_mode_label()` after `_getenv_bool` was introduced. Replaced with a local import of `_getenv_bool` consistent with how all other winnow imports in the script are deferred into function bodies.

## [0.5.17] - 2026-06-15

### Fixed

- **`_getenv_num` and `_getenv_bool` now treat an explicitly-empty env var as unset**: previously, `YEARS_FILTER=` (blank) in a `.env` or Compose file caused `int("")` to raise `ValueError`, logging a spurious "not a valid int" warning and returning the default. Both helpers now strip whitespace and treat an empty string the same as an absent variable, returning the typed default silently. This affects all numeric config vars (`YEARS_FILTER`, `MIN_FACE_WIDTH`, `MIN_FACE_COUNT`, `MAX_AUTO_IMAGES`, `BLUR_THRESHOLD`, `MIN_CONFIDENCE`, `FACE_MARGIN`) and all boolean config vars. `_getenv_optional_float` already handled this correctly.

- **`FORCE_CPU` now uses `_getenv_bool`**: `embeddings.py` retained the old inline `os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes")` pattern after v0.5.16 introduced `_getenv_bool`. The inline copy is now replaced so the canonical truthy-string set is defined in one place.

## [0.5.16] - 2026-06-15

### Changed

- **`_getenv_int` and `_getenv_float` now share a single `_getenv_num` implementation**: the two helpers were structurally identical (read env var, return typed default if absent, try-cast, warn and return default on `ValueError`) with only the cast differing. Both are now thin wrappers around a private `_getenv_num(name, default, cast)`, eliminating the duplicated warning logic.

- **`FRIGATE_SCORE_CEILING` now uses `_getenv_optional_float`**: the previous 9-line inline block (`os.getenv("FRIGATE_SCORE_CEILING", "").strip()` + try/except) has been replaced with a new `_getenv_optional_float(name) -> float | None` helper that encapsulates the "empty-string means None, parse-error means None" semantics, making it consistent with the other numeric env var helpers.

- **Boolean env vars now use `_getenv_bool`**: the `.lower() in ("true", "1", "yes")` pattern was repeated across 11 sites in `config.py`, `jobs.py`, and `cli.py`. A new `_getenv_bool(name, default)` helper centralises the canonical truthy-string set; all sites have been updated to call it.

- **`_resolve_strategy` no-embedding branch uses `_getenv_int`**: the inline `int(custom_limit)` try/except block in `jobs.py` for the time-spread path has been replaced with `_getenv_int("LIMIT", 30)`, matching the pattern used in `config.py`. The smart-mode path retains its own try/except because its fallback is to the strategy map rather than to a numeric default.

- **`cache.py` tmp path uses `str.removesuffix`**: `final[:-4] + ".tmp.npy"` replaced with `final.removesuffix(".npy") + ".tmp.npy"` — the assumption that the cache path ends in `.npy` is now explicit and self-documenting rather than expressed as a magic numeric slice.

## [0.5.15] - 2026-06-15

### Fixed

- **Embedding cache writes were silently no-ops since v0.5.13**: the atomic-write path used `tmp = final + ".tmp"` where `final` ends in `.npy` (e.g. `abc.npy`), producing a tmp path of `abc.npy.tmp`. `np.save` auto-appends `.npy` to paths not already ending in `.npy`, so it wrote to `abc.npy.tmp.npy` instead. The subsequent `os.replace("abc.npy.tmp", "abc.npy")` then raised `FileNotFoundError` (caught silently at DEBUG), meaning no cache entry was ever committed and leaked `*.npy.tmp.npy` files accumulated on disk. The fix inserts `.tmp` before the `.npy` extension: `tmp = final[:-4] + ".tmp.npy"` so `np.save` sees a path already ending in `.npy` and does not re-append.

- **`_getenv_int`/`_getenv_float` no longer route the default through `str()` conversion**: the previous form `os.getenv(name, str(default))` converted the default to a string so it could be fed through `int()`/`float()` — an unnecessary round-trip that would cause `_getenv_int("FOO", 4.0)` to log a spurious "not a valid integer" warning and return the float. The helpers now use `raw = os.getenv(name); return default if raw is None else int(raw)`, passing the typed default through directly.

- **`execute_jobs` progress task now removed via `try/finally`**: `progress.remove_task(job_task)` was duplicated in three early-exit paths (ValueError, symlink TOCTOU, OSError) plus once at normal completion. The entire per-job body is now wrapped in `try/finally: progress.remove_task(job_task)`; the three inner `continue` statements trigger the `finally` automatically before advancing to the next job, making the invariant structurally impossible to violate by a future code path.

## [0.5.14] - 2026-06-15

### Fixed

- **Invalid env var values for numeric config now warn and use defaults**: `YEARS_FILTER`, `MIN_FACE_WIDTH`, `MIN_FACE_COUNT`, `MAX_AUTO_IMAGES`, `BLUR_THRESHOLD`, `MIN_CONFIDENCE`, and `FACE_MARGIN` all used bare `int()`/`float()` with no error handler. A typo such as `YEARS_FILTER=10 ` (trailing space) or `MIN_FACE_WIDTH=auto` raised `ValueError` from inside `__getattr__`, surfacing as a cryptic traceback on the first config access rather than at the config-validation step where a helpful error is expected. The values are now parsed with module-level `_getenv_int` / `_getenv_float` helpers that log a `WARNING` and fall back to the documented default on parse failure, matching the existing pattern already used for `FRIGATE_SCORE_CEILING`.

## [0.5.13] - 2026-06-15

### Fixed

- **`execute_jobs` output-dir OSError now skips the job instead of aborting the run**: `shutil.rmtree` and `os.makedirs` were not wrapped in any error handler — an `OSError` or `PermissionError` (e.g. read-only filesystem, lingering lock) propagated out of the `for job in jobs` loop, abandoning `job_task` in the Rich progress display and silently dropping all remaining jobs. Both calls are now wrapped in `try/except OSError`; on failure the error is logged, the progress task is removed, and the loop continues to the next job.

- **Embedding cache writes are now atomic**: `cache.py` previously called `np.save(path, embedding)` directly to the final `.npy` path. A process kill or container stop mid-write left a truncated file that `np.load` would subsequently raise on. Because `get()` catches the exception and returns `None`, the slot appeared empty on every future run — the corrupted file was never cleaned up and the embedding was silently recomputed forever. The write now goes to a `.tmp` sibling and is renamed into place with `os.replace` (atomic on POSIX); the tmp file is removed on any write failure.

- **`filter_recent_assets` guards against non-string `fileCreatedAt`**: the previous `if not created_at_str` guard passed truthy non-string values (e.g. a Unix-epoch integer returned by some Immich API versions), after which `created_at_str.replace("Z", "+00:00")` raised `AttributeError`. That exception was not caught by the surrounding `except ValueError`, so a single non-string timestamp aborted the entire filtering pass for the person being processed. The guard is now `if not isinstance(created_at_str, str) or not created_at_str`.

- **SQLite connection timeout raised to 30 s**: `sqlite3.connect` defaulted to a 5-second busy timeout. Under concurrent access (scheduled and manual runs overlapping), 5 s was often insufficient, causing `OperationalError: database is locked` that propagated through `upload_to_frigate` and dropped upload-tracking records — assets would then be re-uploaded on the next run. The timeout is now 30 s, matching the typical upload cycle length.

## [0.5.12] - 2026-06-15

### Fixed

- **Progress task leak on skipped jobs**: `progress.add_task()` is called unconditionally at the top of the job loop, but both early-exit `continue` paths — the `ValueError` skip from `_safe_person_dir` and the symlink-TOCTOU skip added in v0.5.11 — bypassed `progress.remove_task()`, leaving orphaned 0% rows in the terminal display for the rest of the run. Both `continue` paths now call `progress.remove_task(job_task)` before continuing.

## [0.5.11] - 2026-06-15

### Fixed

- **`execute_jobs` symlink TOCTOU gap closed**: the v0.5.10 guard `os.path.isdir(person_dir) and not os.path.islink(person_dir)` silently skipped the wipe when `person_dir` was a symlink-to-directory, then called `os.makedirs` which followed the symlink — allowing crop writes to land outside `output_dir` with no log or skip. The guard is replaced by an explicit pre-check: if `os.path.islink(person_dir)` is True, log an error and `continue`, matching the established `ValueError` pattern from `_safe_person_dir`. The `isdir` / `rmtree` block is restored to its original simple form.

## [0.5.10] - 2026-06-15

### Fixed

- **Reconcile `< target` branch re-escalated to WARNING**: when fewer Frigate files appear than expected after the full backoff window, the affected files are permanently unmapped — identical in consequence to the `> target` (external upload race) case fixed in v0.5.9. The v0.5.9 demotion to `INFO` was incorrect; both post-loop branches now log at `WARNING` and include the "permanently unmapped" label.

- **`execute_jobs` symlink guard added before `shutil.rmtree`**: `os.path.isdir` follows symlinks and returns `True` for a symlink pointing at a directory. If a race condition replaces `person_dir` with such a symlink, the old guard would pass and `shutil.rmtree` would raise an unhandled `OSError`, aborting all remaining jobs in the batch. The guard is now `os.path.isdir(person_dir) and not os.path.islink(person_dir)`, so a symlink-to-directory is silently skipped. The comment is also corrected: `shutil.rmtree` raises `OSError` (not `NotADirectoryError`) on a top-level symlink.

## [0.5.9] - 2026-06-15

### Fixed

- **Reconcile log severity corrected**: the external-upload branch (`len(new_files) > target`) was logged at `INFO` while the timeout branch (`< target`) was logged at `WARNING`. The severity is now inverted to match impact: external upload causes permanent mapping loss (those files are never eligible for quality replacement) and is now `WARNING`; timeout is transient and recoverable next cycle and is now `INFO`.

- **`fetch_all_assets` docstring: lower-bound caveat now covers both interruption cases**: previously only noted that a network error makes `total_raw` a lower bound. An all-garbage page (every item non-dict) also terminates pagination early, leaving later pages unfetched — this case is now documented alongside the network error case.

- **`shutil.rmtree` symlink safety documented**: added a comment above the `rmtree` call in `execute_jobs` noting that POSIX `shutil.rmtree` raises `NotADirectoryError` on a top-level symlink, so a race-replaced symlink cannot cause out-of-tree deletion.

- **`_entry()` in `get_person_summary` no longer allocates default dict for present keys**: `setdefault` evaluates its default argument before checking whether the key exists, allocating and immediately discarding a 5-key dict on every call for an already-present person. Replaced with an explicit `if name not in summary` guard.

## [0.5.8] - 2026-06-15

### Fixed

- **`_safe_person_dir` docstring corrected**: the previous comment stated "checking after realpath would be too late because realpath follows the link first," which implied `islink` was the primary security guard. The load-bearing path-traversal check is `realpath + startswith`; it rejects both `../../` traversal and symlinks. The `islink` check is a supplementary early-exit that provides a cleaner error message for the symlink sub-case only.

- **`total_raw` not inflated by all-garbage pages**: `fetch_all_assets` previously added `page_count` to `total_raw` before checking whether any valid dict items existed. A page returning only non-dict items would inflate `total_raw` and produce a misleading "N total, 0 recent" display. `total_raw` now accumulates only after `valid_assets` is confirmed non-empty, so all-garbage pages break without contributing. Mixed pages (some valid, some non-dict) still count `page_count` so transient schema glitches on a partial page don't cause `MIN_FACE_COUNT` to incorrectly skip a real person.

- **Pagination interruption warning**: when a `RequestException` breaks pagination mid-way (page > 1), a `WARNING` is now logged noting that `total_raw` is a lower bound. Previously the exception was logged at `ERROR` with no indication that the `MIN_FACE_COUNT` comparison was using a partial count.

- **Config TOCTOU residue**: `config.py` line 135 re-stat'd `_data_cfg` when it was the selected config file, creating a second TOCTOU window after the fix in v0.5.7. The check is now `if _data_cfg_exists or config_file.exists():` — the primary path is never stat'd again, and the legacy path is stat'd at most once.

## [0.5.7] - 2026-06-15

### Fixed

- **Symlink guard moved before `realpath`**: `_safe_person_dir` now checks whether the raw (unresolved) path is a symlink before calling `os.path.realpath`. The previous check in `execute_jobs` ran after `realpath` had already resolved the link, making it unreachable dead code.

- **`fetch_all_assets` returns raw item count**: the function now returns `(assets, total_raw)` where `total_raw` is the total items seen across all pages before non-dict filtering. `auto_configure` and `_configure_person` use `total_raw` for the `MIN_FACE_COUNT` guard and display, so transient non-dict API items cannot cause a person to be incorrectly skipped.

- **Pagination stop on all-non-dict page now logs a warning**: when `valid_assets` is empty but the page was non-empty (all items were non-dict), a `WARNING` is emitted explaining why pagination stopped, distinguishing it from natural end-of-data.

- **`_data_cfg.exists()` called once**: the result is cached in `_data_cfg_exists` so the dual-config warning check and the `config_file` selection always agree — previously two separate `stat()` calls created a TOCTOU window where the log could claim one file while the code loaded another.

## [0.5.6] - 2026-06-15

### Fixed

- **Pagination runaway on all-non-dict page**: the empty-page break in `fetch_all_assets` now fires after non-dict filtering rather than before, so a page whose items are all non-dict (e.g. all nulls) correctly terminates pagination instead of looping to MAX_PAGES.

- **Non-dict API items upgraded to warning**: items skipped in a paginated response are now logged at `WARNING` (previously `DEBUG`) so silent asset loss is visible at default log levels.

- **Single-pass page filtering**: `fetch_all_assets` now partitions valid and invalid items in one loop instead of iterating `page_assets` twice with inverse predicates.

- **Reconciliation checks Frigate before sleeping**: the poll loop now performs an initial check immediately after upload, then backs off with `_RECONCILE_POLL_DELAYS` only if needed. Previously the loop always slept ≥1 s before any check.

- **Reconciliation set subtraction computed once**: `current_files - known_files_before` was computed twice per poll iteration (once for the count check, once for the final mapping). It is now computed once and reused.

## [0.5.5] - 2026-06-15

### Changed

- **Module-level constants in `diversity.py`**: magic numbers `3000` (pool cap), `20` (pool scale), and `32` (embedding batch size) extracted to named constants `_POOL_CAP`, `_POOL_SCALE`, and `_EMBEDDING_BATCH_SIZE`.

- **Reconciliation poll delays extracted**: `(1, 2, 4, 8)` back-off delays in `reconcile.py` extracted to `_RECONCILE_POLL_DELAYS` with an explanatory comment.

- **`_VALID_SCORE_COLS` comment**: explains that the frozenset is a SQL-injection guard for dynamic column interpolation, not a runtime filter.

- **`record_frigate_files_batch` docstring**: clarifies that all mappings are written atomically — no partial failure is possible.

- **`get_person_summary()` refactored**: eliminated four repeated default-dict blocks using a local `_entry()` helper with `setdefault`.

- **`encoded` → `encoded_name` in `frigate_api.py`**: renamed the URL-encoded person name variable for clarity.

- **Dual response shape comment in `immich_api.py`**: documents that Immich ≥2.x returns `{"assets": {"items": [...]}}` while earlier versions returned `{"assets": [...]}` directly.

- **Non-dict item debug log in `fetch_all_assets`**: skipped non-dict items in a page response now emit a `logger.debug` line with the count and page number.

## [0.5.4] - 2026-06-14

### Fixed

- **Quality replacement slot floor used wrong score**: when a blur-score replacement deleted a low-quality Frigate file but the subsequent upload failed, `min_quality_score_for_slot` was set to the failed candidate's score rather than the deleted file's score. This caused subsequent candidates that were better than the deleted file (but worse than the failed upload) to be skipped, leaving the freed slot unfilled for the rest of that run. Fixed by using `target_score` (deleted file's score) as the floor, matching the documented intent in the surrounding comment.

## [0.5.3] - 2026-06-14

### Fixed

- **`LIMIT` env var crash**: non-integer values (e.g. `"30.5"`, `"all"`) now log a warning and fall back to the default instead of raising `ValueError` at startup.

- **Symlink guard on person output dir**: `shutil.rmtree` is now skipped if `person_dir` resolves to a symlink, preventing traversal out of `OUTPUT_DIR` on a shared volume.

- **`person["id"]` KeyError**: malformed Immich API responses missing the `id` field now log an error and skip that person instead of crashing the job.

- **Face data response type validation**: `fetch_face_data` now validates that the `/api/faces` response is a list before indexing, guarding against null or non-list API responses.

- **Pagination error log includes page number**: the exception log in `fetch_all_assets` now includes the page number that failed.

- **`Image.open()` wrapped for non-image responses**: PIL parse errors on thumbnail fetches (e.g. reverse-proxy HTML error page returning 200) are now caught and logged instead of propagating.

- **Frigate version `v`-prefix handling**: `v0.16.0`-style version strings are now correctly parsed; the leading `v` was previously misread, causing the too-old warning to never fire.

- **`FRIGATE_SCORE_CEILING` parse guard**: a non-float value in `.env` now logs a warning and disables the ceiling instead of crashing at startup.

- **Dual config file warning**: a log warning is emitted when both `DATA_DIR/.immich_config.json` and the legacy CWD config file exist simultaneously.

- **PID file write guard**: `OSError` on `/tmp/winnow.pid` write is now caught and logged instead of crashing the scheduler.

- **Scheduler sleep clamped to 60 s**: bounds recovery time after an NTP clock step.

- **`get_frigate_person_files` non-list debug log**: consistent with `get_all_frigate_person_files`.

## [0.5.2] - 2026-06-14

### Fixed

- **Immich v2.7.5 compatibility**: `auto_configure` no longer pre-filters people by `assetCount` from the `/api/people` response, which Immich v2.7.5 dropped. The `MIN_FACE_COUNT` check now runs after `fetch_all_assets` so the actual asset count is used instead of the missing field.

- **Dockerfile supply-chain**: replaced `curl | sh` uv installer with `COPY --from=ghcr.io/astral-sh/uv:0.11.21` to eliminate the network-executed script.

- **HEALTHCHECK**: replaced the static file-existence check with `kill -0 $(cat /tmp/winnow.pid)` so the container reports unhealthy when the scheduler process actually dies, not just when a script file is missing.

- **`CONFIG_FILE` volume safety**: the config file path now resolves to `DATA_DIR/.immich_config.json` so it persists across container restarts. The legacy CWD location is still read as a fallback for existing setups.

- **EmbeddingCache singleton isolation**: `get_cache()` now tracks the `cache_dir` argument and re-creates the cache when it changes, preventing test runs from sharing state across different `DATA_DIR` values.

- **File descriptor leak in `_suppress_output()`**: `devnull_fd`, `saved_out`, and `saved_err` are now all closed in a nested `finally` chain, preventing fd exhaustion on long runs.

- **Silent exception in `upload_tracker`**: `except Exception: pass` on SQLite connection close is now `except Exception as e: logger.debug(...)` so connection errors are visible in debug logs.

- **Frigate API unknown-key logging**: `get_all_frigate_person_files` now logs unexpected non-list keys at DEBUG level instead of silently skipping them.

- **Reconcile debug log**: added a debug log entry before the FIFO timestamp mapping step in `reconcile_frigate_mappings` to make the mapping assumption visible in logs.

- **CI action SHA pinning**: all five GitHub Actions workflows now pin every third-party action to a full commit SHA. Updated `setup-uv` v7→v8.2.0, `upload-artifact` v4→v7.0.1, `download-artifact` v4→v8.0.1, `ruff-action` v3→v4.0.0.

## [0.5.1] - 2026-06-14

### Changed

- **`CACHE_DIR` renamed to `DATA_DIR`**: the environment variable that sets the path for the embedding cache and SQLite tracker database is now called `DATA_DIR` (default: `data`; Docker default: `/app/data`). The old `CACHE_DIR` still works with a startup deprecation warning — rename it to `DATA_DIR` in your `.env` or `compose.yml` to silence the warning. The container-side default path changes from `/app/.if_cache` to `/app/data`; update your volume mount accordingly.

## [0.5.0] - 2026-06-14

### Changed

- **SQLite upload tracker**: `upload_tracker.py` is fully rewritten on top of SQLite (stdlib `sqlite3`). The JSON pair (`frigate_uploaded_ids.json` / `frigate_rejected_ids.json`) is replaced by a single `winnow_tracker.db` (WAL journal, `check_same_thread=False`). Existing JSON files are migrated atomically on first run and renamed to `.json.bak`. No user action required; the tracker API (`mark_uploaded`, `mark_rejected`, `filter_already_uploaded`, `get_person_summary`, etc.) is unchanged.

- **Config lazy singleton**: `_Config` now uses `__getattr__` to defer all I/O until the first attribute access. `load_dotenv()` no longer runs at module import time — it runs on the first access to any `Config` attribute. Empty-string env vars (`IMMICH_URL=`, `OUTPUT_DIR=`) are now correctly distinguished from unset ones so a `.env` file value never silently overrides an explicit `""` set in the environment. `Config.reset()` clears the loaded state for clean test isolation.

- **Reconcile module extracted**: `reconcile_frigate_mappings` and `enrich_asset_with_face_data` are extracted from `executor.py` into a new `winnow/reconcile.py` module. No behaviour change; reduces `executor.py` length and clarifies responsibility boundaries.

- **Single lockfile**: `pyproject-gpu.toml`, `pyproject-cpu.toml`, `pyproject-rocm.toml`, `pyproject-intel.toml` and their separate lockfiles are removed. GPU/ROCm/Intel/CPU variant deps are now declared as `[project.optional-dependencies]` extras in `pyproject.toml` with `[tool.uv] conflicts` for mutual exclusion. A single `uv.lock` covers all variants. The Dockerfile selects the correct extra via `uv sync --extra $VARIANT`.

- **Ubuntu base bumped**: amd64 GPU base updated from `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` to `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`. amd64 ROCm and CPU bases updated from Ubuntu 22.04 to Ubuntu 26.04. arm64 bases remain Ubuntu 24.04.

### Fixed

- **Frigate API unreachable at upload start no longer crashes reconciliation**: when the Frigate `GET /api/faces` call fails at upload start, reconciliation is now skipped entirely for that batch (`_skip_reconcile = True`). Previously, falling back to the tracker's known filenames as the pre-upload baseline caused the `> target` guard to fire on unmapped manual files, silently dropping all mappings.

- **Polling `== target` guards against wrong-file mapping**: the reconcile poll loop now breaks on `len(new_files) == target` and sets an "external upload detected" flag when `> target`. The old `>= target` break would have proceeded with an incorrect file set when a concurrent external upload was present, causing wrong asset-ID mappings. The poll loop now also exits early on `> target` rather than exhausting all four retry intervals (up to 15 s wasted per person with a concurrent external uploader).

- **`auto_cap` post-selection truncation removed**: the diversity selector now receives the correct upper bound (`capacity` or `min(limit, capacity)`) directly instead of selecting up to `MAX_AUTO_IMAGES` and then silently truncating the result list. The old approach produced a selection biased toward the first `capacity` items in embedding space rather than the globally optimal diverse subset.

- **Dockerfile unknown VARIANT now fails loudly**: added an explicit `elif [ "$VARIANT" = "gpu" ]` branch and an `else … exit 1` for unrecognised values. Previously, any unknown variant silently fell through to the `cpu` branch.

- **JSON migration partial-rename data loss**: if the rename of one of the two JSON files failed (e.g. a `PermissionError`), the other file's data was committed to SQLite but the `COUNT(*) > 0` guard on the next run would skip re-migration of the remaining file, permanently losing its data. The guard is removed (idempotent `INSERT OR IGNORE` makes re-running safe). Each rename is now wrapped in its own `try/except OSError` so a failure on one file is logged and does not prevent the other from completing.

- **SQL column allowlist in `_pick_mapped_file`**: the `score_col` f-string interpolation into SQL is now guarded by a `frozenset` allowlist at the function boundary, raising `ValueError` on any value outside `{"blur_score", "frigate_score"}`.

- **`load_dotenv` no longer runs at import time**: moving `load_dotenv()` to the first line of `_load()` prevents side-effects during module import (which could interfere with test environment setup) and makes the load order deterministic relative to `os.environ` overrides.

- **Empty-string env var priority fix**: `if self.IMMICH_URL or …` treated `IMMICH_URL=""` as falsy and silently fell through to the config file. Changed to `if self.IMMICH_URL is None` so an empty-string explicit env var is respected.

### Added

- **Diversity test suite expanded** (PR #11): 33 new tests covering k-medoids clustering, farthest-point sampling, adaptive threshold computation, near-duplicate deduplication, and time-spread selection. Total: 93 tests (was 60).

- **Known-limitation annotations** (PR #12): `TODO(frigate-api)` comments placed at each FIFO-ordering assumption, manual-file-invisibility note, and async-rebuild limitation in `executor.py` and `reconcile.py`. These mark spots where a richer Frigate API would allow a deeper fix.

## [0.4.11] - 2026-06-14

### Removed

- **Object mode pipeline fully removed**: YOLO object detection, SigLIP image classification, `TRAINING_MODE`, and `OBJECT_CLASS` env vars are gone. Frigate has no training API for objects; the ~2 GB model stack (torch, torchvision, transformers, ultralytics) was dead weight.
- **Dead Immich embedding path removed**: `FaceData.embedding` field and the `immich_embedding` parameter to `get_embedding()` were never consumed by any caller. Both removed along with the NumPy import in `immich_api.py` that existed solely for that path.
- **Dead `mode` config key removed**: `"mode": "face"` was written into job config dicts in `jobs.py` but never read after object mode removal.

### Fixed

- **InsightFace `FutureWarning` suppressed in crop-alignment path**: the `insightface_app.get()` call in `image_processing.py` now wraps the same `warnings.catch_warnings()` suppressor already present in `embeddings.py`, preventing scikit-image deprecation noise in logs.

### Changed

- **Variant pyproject files synced to current state**: `pyproject-rocm.toml`, `pyproject-cpu.toml`, `pyproject-intel.toml` were at v0.2.13 and still listed torch/transformers/ultralytics. Updated to v0.4.11 and cleaned to face-only deps. Note: corresponding lockfiles (uv-rocm.lock, uv-cpu.lock, uv-intel.lock) need regeneration in their respective platform environments.
- **`MERGE_DUPLICATE_PEOPLE` documented**: README and wiki now explain the default warn-and-skip behaviour vs. setting `true` for a permanent Immich merge, with irreversibility callout.
- **Wiki fully updated**: all five wiki pages rewritten to remove object mode references, correct model size (~300 MB InsightFace vs former ~1–2 GB HuggingFace+InsightFace), fix default values (`MAX_AUTO_IMAGES` 80→20, `MIN_FACE_COUNT` 0→3), add `MERGE_DUPLICATE_PEOPLE` coverage, and update GPU verification commands for current ONNX provider API.

## [0.4.10] - 2026-06-14

### Changed

- **`MAX_AUTO_IMAGES` default lowered from `80` to `20`**: winnow is designed to fill the gap where manual Frigate training images don't exist — not to be the primary dataset. A conservative default ensures winnow-imported images remain secondary to hand-picked ones where both exist.

## [0.4.9] - 2026-06-14

### Changed

- **`FRIGATE_SCORE_CEILING` is now dynamic by default**: previously defaulted to `0` (disabled). Now unset (default) enables a self-calibrating novelty gate — below-cap candidates are skipped if their pre-upload Frigate score exceeds the most-redundant tracked file's score. This catches conditions already covered by manually-added Frigate images that winnow cannot track. Set `FRIGATE_SCORE_CEILING=0` to disable entirely; set a positive value (e.g. `0.85`) for a fixed hard ceiling.
- **Quality replacement branches consolidated**: the Frigate-score and blur-score replacement paths in the upload loop shared identical structure. Merged into a single code path parameterised by score source and comparison direction.
- **`MIN_FACE_COUNT` default raised from `0` to `3`**: people with fewer than 3 tagged photos produce degenerate training sets; skipping them by default avoids noisy runs.
- **`STRATEGY=adaptive`** is the new primary name for embedding-based diversity selection; `auto` remains a silent alias for backwards compatibility.
- **`MERGE_DUPLICATE_PEOPLE` and `TRACE_CROP_SIZE`** added to the README env var table (were in the codebase but undocumented).
- **CUDA version corrected** in the image tags table (was 13.3, actual base image is 12.8.1).

## [0.4.8] - 2026-06-14

### Changed

- **Tracker write-through cache**: `upload_tracker` now keeps an in-memory copy of each JSON file keyed by its resolved path. All reads after the first hit the cache instead of disk; writes go to both disk and cache atomically. Cuts per-person disk I/O in the upload loop from ~90 reads to ~1, with no API or behaviour changes.

## [0.4.7] - 2026-06-14

### Changed

- **`_dedup_embeddings` pre-allocated buffer**: replaced the grow-on-keep `np.vstack` pattern with a pre-allocated `(Q, D)` buffer filled row-by-row. Eliminates O(K²) copy work and the GC pressure from K intermediate heap allocations while keeping identical arithmetic for the similarity checks.
- **`_kmedoids` cost computation vectorized**: the Python-level `sum(dist_matrix[i, medoids[labels[i]]] for i in range(n))` generator (called once per swap evaluation) is replaced with `dist_matrix[np.arange(n), np.array(medoids)[labels]].sum()` — a single numpy fancy-index + reduction, ~20–50× faster in the swap loop.
- **`_reconcile_frigate_mappings` single-write batch**: previously called `record_frigate_file` once per uploaded file, each doing a full JSON load + save (O(L) disk round-trips per person). Now builds the full `{frigate_filename: asset_id}` mapping dict and writes it in one `record_frigate_files_batch` call (O(1) disk round-trip).

## [0.4.6] - 2026-06-14

### Fixed

- **OOM when Immich returns many pages per person**: `fetch_all_assets` now stops fetching once 5000 assets have been collected — the diversity selection pool is already capped at 3000 items, so fetching up to 1,000,000 was wasteful and could exhaust memory on large libraries. 5000 provides ample headroom for the pool cap while bounding per-person memory to ~2 MB.
- **Non-dict items in Immich asset pages silently skipped**: a malformed or partially-null Immich response page could include `null` or non-object items in the assets array. These are now filtered at fetch time rather than causing `AttributeError` downstream.

## [0.4.5] - 2026-06-14

### Fixed

- **Near-duplicate dedup O(N²) allocation**: `np.vstack(kept_normed)` was rebuilt on every loop iteration even for candidates that would be dropped; the stack is now rebuilt only when a new item is kept, reducing memory pressure significantly for large pools.
- **`quality_score` falsy-zero in dedup sort**: the sort key used `c.get("quality_score") or 0.0`, which treated a legitimate `quality_score=0.0` identically to a missing key. Changed to an explicit `None` check so zero is preserved as-is, and object-mode candidates (which have no `quality_score`) continue to sort stably to the back.
- **Post-dedup pool not re-checked against limit**: after near-duplicate removal the pool could silently shrink below the requested limit with no warning. A second `len < limit` guard now fires after dedup and emits the same "Only N embeddings" warning that the pre-dedup guard does.
- **`mark_rejected` could miss plain-text 400 bodies longer than 100 bytes**: `error_detail = resp.text[:100]` was being searched for the keyword `"face"` to gate `mark_rejected()`, so a response body with `"face"` after byte 100 would never mark the asset rejected and it would be retried on every future run. The `"face"` check now uses the full response body; truncation is kept only for the displayed snippet.
- **`_safe_person_dir` raised ValueError for all person names when `output_dir` resolved to `/`**: `base + os.sep` produced `"//"` when base was `"/"`, and valid paths like `/alice` don't start with `"//"`. Fixed by using `base` directly as the prefix when `base == os.sep`.

## [0.4.4] - 2026-06-14

### Added

- **`RESET_PERSON=*` bulk reset**: resets every tracked person at once (deletes their Frigate training files and clears tracker data). Any other value still resets that specific person by name. If a person is literally named `*` they are reset as part of the bulk operation, and a warning is printed to clarify this.
- **Near-duplicate removal before diversity selection**: a greedy dedup pass now runs after embedding collection and before clustering. Candidates within 0.20 cosine distance of a higher-quality image are dropped, eliminating burst shots and same-event lookalike photos that produce redundant training images. The best-quality frame from each near-identical group is kept. Dropped count is logged per person.

### Fixed

- **HTTP 500 upload errors no longer show Frigate's misleading "Try restarting Frigate" message**: the response body is now logged at debug level only. HTTP 400 detail (e.g. "No face was detected") is still shown since it is actionable.
- **`RuntimeWarning: Mean of empty slice`** when a person has only one image after quality filtering: `_compute_adaptive_threshold` now returns the floor value immediately when there are no pairwise distances to sample, and the k-medoids cluster count is floored at 1 to prevent `k=0`.
- **Path traversal guard on output directory**: person names with `../` sequences or absolute paths (e.g. `/etc`) are now rejected before any filesystem operation, logging an error and skipping the job rather than writing outside the output tree.

## [0.4.3] - 2026-06-14

### Added

- **InsightFace landmark-based face crop alignment**: face crops for Frigate training are now aligned using InsightFace's `norm_crop` (ArcFace 112×112 alignment with 5-point facial landmarks). Previously, Immich's API returned only bounding boxes with no landmarks, so `align_face()` was dead code and crops were plain bbox slices — resulting in misaligned or partial crops (e.g. foreheads). The fix runs InsightFace detection on an expanded region around the Immich bbox, finds the nearest face, and uses its keypoints for proper alignment. Controlled by `ENABLE_FACE_ALIGNMENT` (default `true`).
- **Duplicate Immich person detection and handling**: when multiple Immich person records share the same name, winnow now detects this at startup and warns with a per-group summary. Without handling, two jobs would run for the same Frigate folder and overwrite each other's output. By default (`MERGE_DUPLICATE_PEOPLE=false`) only the first person per name is processed. Set `MERGE_DUPLICATE_PEOPLE=true` to permanently merge duplicate records inside Immich (keeps the person with the most assets).

## [0.4.2] - 2026-06-13

### Changed

- **GPU image now uses CUDA 12.8.1** (was 13.3): CUDA 13.3 requires driver ≥ 575; driver 570 (the current stable release) was incorrectly rejected with "CUDA driver version is insufficient" at startup. The `:latest` image now works with any NVIDIA driver ≥ 570.

### Added

- **`scripts/benchmark.py`**: measures InsightFace and SigLIP inference latency and throughput across GPU and CPU modes. Run inside the container with `python /app/scripts/benchmark.py`. RTX 2070 SUPER results: InsightFace 12.8 ms / 78 img/s (8× CPU), SigLIP batch 32 at 5.4 ms/img / 187 img/s (33× CPU).

## [0.4.1] - 2026-06-13

### Fixed

- **`RESET_PERSON` no longer creates duplicate Frigate files**: previously, resetting a person only wiped the local tracker — existing Frigate training files were left as unmanaged orphans, causing the next run to upload a full new batch on top of them. `reset_person` now deletes all winnow-managed files for that person from Frigate before clearing the tracker. Manually-added Frigate files are unaffected.
- **No spurious warning when `FRIGATE_URL` is unset and `RESET_PERSON` is used**: the deletion step is now skipped silently at info level rather than logging a misleading "could not delete" warning.

## [0.4.0] - 2026-06-13

### Added

- **Pre-upload Frigate recognition scores**: `recognize_face` is now called before each upload to measure how novel the candidate is relative to the existing training set. The score is stored in the tracker (`frigate_scores` field) and drives quality replacement in subsequent runs. Adds ~200 ms per upload.
- **`ENABLE_FRIGATE_SCORES`** (default `true`): controls all pre-upload Frigate recognize calls. Set `false` to use blur-score replacement only and skip the Frigate round-trip entirely.
- **`FRIGATE_SCORE_CEILING`** (default `0.0`): skip uploads whose pre-upload recognize score already exceeds this value — those face conditions are already well-covered by the training set. `0` disables (no ceiling); requires at least one prior run to have stored scores.
- **`get_most_redundant_mapped_file()`**: new upload-tracker function that returns the mapped file with the highest Frigate pre-upload score. High score = the training set already covers that face condition well = the best deletion target for quality replacement.
- **Cold-start notice**: first run (no existing Frigate model) now logs a clear message explaining why Frigate scores are unavailable and that they will populate on subsequent runs.
- **4 new tests** for `get_most_redundant_mapped_file` covering score ordering, ties, excludes, and no-score cases.

### Changed

- **Quality replacement now uses Frigate scores**: when Frigate scores are available, at-cap replacement targets the _most redundant_ mapped file (highest pre-upload score) and replaces it only when the candidate is _more novel_ (lower score). Falls back to blur-score comparison when no Frigate scores have been stored yet.
- **`recognize_face` returns `(face_name, score) | None`** instead of `float | None`: the caller now validates that the recognized person matches the expected person before using the score. Wrong-person scores no longer drive ceiling skips or replacement decisions.
- **Bootstrap fix**: recognize was previously called below-cap only when `FRIGATE_SCORE_CEILING > 0`, so `frigate_scores` was never populated with default settings and the Frigate replacement path never activated. Recognize is now called for all below-cap uploads when `ENABLE_FRIGATE_SCORES=true`, seeding scores for future at-cap runs regardless of ceiling setting.
- **Batch GET `/api/faces`**: Frigate file-count lookups are now batched to reduce round-trip overhead on runs with many people.
- **Skip candidate download on low Frigate confidence**: candidates where the Immich detection confidence is below threshold are now filtered before the full-resolution download, saving bandwidth.

### Removed

- **Post-upload quality gate (`FRIGATE_SCORE_THRESHOLD`)**: enforcement of a Frigate score threshold after upload has been removed. Post-upload scores are taken after the image is already in the training set, so the model has already retrained on it — deleting it at that point is wasteful and disrupts the model for the next Frigate run. Pre-upload scoring (`FRIGATE_SCORE_CEILING`) provides a cleaner signal at the right moment.

### Fixed

- **Frigate replacement path never activated with default settings**: with `FRIGATE_SCORE_CEILING=0.0` (default), the bootstrap call to `recognize_face` was gated behind `CEILING > 0`, so `frigate_scores` stayed empty, `has_frigate_scores` was always False, and the Frigate replacement branch was permanently unreachable. Removing the ceiling guard from the below-cap recognize call breaks the circular dependency.
- **Schema comment contradiction**: `upload_tracker.py` line-16 comment described `frigate_scores` as "post-upload" while the block comment on lines 22–24 said "pre-upload". Corrected to "pre-upload" throughout.
- **README default values**: `MIN_FACE_WIDTH` was documented as `50` (actual default: `90`); `BLUR_THRESHOLD` was documented as `100.0` (actual default: `120.0`). Both corrected.
- **README missing env vars**: `FRIGATE_SCORE_CEILING` and `ENABLE_FRIGATE_SCORES` were present in `config.py` and `.env.example` but absent from the README env var table. Both added.
- **README quality-replacement description**: Step 8 and the `QUALITY_REPLACEMENT` row now document the dual-mode behaviour (Frigate-score path and blur-score fallback) instead of describing only the original blur-score path.

## [0.3.3] - 2026-06-13

### Fixed

- **`MIN_FACE_WIDTH` default raised from 50 → 90px**: 50px crops produce 2,500–4,225 total pixels, well below Frigate's own camera capture range of 16k–50k px. 90px guarantees ≥8,100 total pixels even when face margins are fully clipped by image edges, keeping winnow training crops above the floor Frigate considers useful.

## [0.3.2] - 2026-06-13

### Added

- **Crop dimension tracing**: winnow now records the pixel dimensions (width × height) of each face crop at upload time in the tracker (`crop_dims` field). Run `TRACE_CROP_SIZE=3848 winnow` to look up which Immich asset produced a crop with that pixel dimension — output includes person name, asset ID, Immich URL, blur score, and the Frigate filename. Useful for tracing low-quality or unexpected images visible in Frigate back to their source.

## [0.3.1] - 2026-06-13

### Fixed

- **Lint**: split overly long line in `quality.py` (`E501`, 146 → ≤120 chars).
- **CI — lockfile update workflow**: added `branches: ['**']` filter to `on.push` so tag pushes no longer trigger the job; tag checkouts land in detached HEAD and the subsequent `git push` had no branch target.
- **CI — release workflow**: split four Docker image builds into parallel jobs (`build-gpu`, `build-cpu`, `build-rocm`, `build-intel`), each with its own runner. Previously all four ran in a single job; building the GPU and CPU multi-platform images exhausted disk, causing ROCm and Intel builds to be cancelled.

## [0.3.0] - 2026-06-13

### Added

- **`get_tracked_frigate_filenames()`** — new upload-tracker function that returns the set of Frigate filenames currently mapped for a person. Used internally as a reconciliation baseline when the Frigate GET endpoint is unreachable; also available to callers that need the mapped filename set without a count.
- **Community scaffolding**: `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue templates (bug report, feature request), and pull request template.
- **OCI image labels**: `org.opencontainers.image.*` labels added to the runtime stage of the Dockerfile so image metadata is surfaced by container registries.
- **Additional tracker tests**: coverage added for `get_tracked_frigate_filenames` and for `get_lowest_quality_mapped_file` with the `exclude` parameter.

### Fixed

- **Quality score scale mismatch (USE_FULL_RESOLUTION=true)**: the time-spread quality-score fallback called `assess_quality` on the full-resolution download, while the embedding path always scores on preview thumbnails. Laplacian variance scales with image resolution, so the two paths produced incomparable scores for people with mixed-mode files. The fallback now caps the image at 1440 px before scoring to match the thumbnail scale.
- **assess_quality failure left file permanently unreplaceable**: if `assess_quality` raised an exception (e.g. an RGBA image with an unsupported channel count), `score_map` kept `None` and `mark_uploaded(score=None)` skipped writing the score. The uploaded file was then permanently invisible to `get_lowest_quality_mapped_file` because it had no entry in `scores{}`. The fallback now converts the image to RGB before scoring and stores `0.0` on any exception, so every uploaded file is eligible for future quality replacement.
- **Uploads during Frigate GET outage never mapped**: when `GET /api/faces` failed at upload start, the reconciliation guard (`_snapshot is not None`) correctly skipped the post-upload diff — but uploads that succeeded during the outage were never recorded in `frigate_files`, causing `get_tracked_frigate_file_count` to permanently under-report and Frigate to eventually exceed `MAX_AUTO_IMAGES`. The code now uses the tracker's mapped filenames as a pre-upload baseline when the live snapshot is unavailable, so reconciliation proceeds normally (the `>target` guard handles concurrent external uploads as before).
- **Freed quality-replacement slot could be filled by a worse image**: when a replacement delete succeeded but the subsequent upload failed all retries, `effective_count` stayed decremented and the next file in the iteration uploaded unconditionally — it could have a lower quality score than the file that was deleted. A `min_quality_score_for_slot` variable now records the deleted file's score on a successful delete; any candidate that doesn't beat that floor is skipped until the slot is filled by a qualifying image or the run ends.
- **CI multi-arch and lockfile bot**: hardened the build workflow — lockfile-update bot no longer races against Docker publish on the same push event; CPU image now builds for both `linux/amd64` and `linux/arm64`; `paths-ignore` prevents documentation-only pushes from triggering image builds.
- **README pipeline diagram updated**: step 8 now explicitly documents the quality-replacement decision tree (`below cap → upload`, `at cap + enabled → swap if better`, `at cap + disabled → skip`).

## [0.2.13] - 2026-06-13

### Added

- **Quality replacement**: when a person is at `MAX_AUTO_IMAGES`, winnow now checks each new candidate against the lowest-quality image already in Frigate and swaps it in if the new image scores higher. Only images winnow uploaded (tracked in `frigate_files`) are ever replaced — files added manually through Frigate's UI are left untouched permanently. Enabled by default; set `QUALITY_REPLACEMENT=false` to revert to the previous behaviour of skipping people at cap.
- **Frigate filename mapping**: each successful upload now records the mapping from Frigate's assigned filename to the originating Immich asset ID and face confidence score in the tracker (`frigate_files` field). This is the foundation for quality replacement and future management of the Frigate training set.
- **`QUALITY_REPLACEMENT` env var** (default `true`): controls whether at-cap people are eligible for quality replacement. When disabled, people at `MAX_AUTO_IMAGES` are skipped as before.
- **NOTICES file**: third-party attribution for if_curator (MIT, Copyright © 2026 Sebastian) added to satisfy upstream license requirements.

## [0.2.12] - 2026-06-13

### Added

- **ROCm (AMD GPU) support**: new `:rocm` image tag. InsightFace runs via `ROCmExecutionProvider`; SigLIP runs via PyTorch ROCm 6.3 (ROCm builds expose `torch.cuda.is_available() == True`, so the existing CUDA path is reused automatically). Requires `/dev/kfd` and `/dev/dri` device passthrough plus `video` and `render` group membership — see `compose.yml` for the snippet.
- **Intel GPU support**: new `:intel` image tag. InsightFace runs via `OpenVINOExecutionProvider` from `onnxruntime-openvino`. By default OpenVINO targets CPU (no device passthrough needed); set `OPENVINO_DEVICE=GPU` to target Intel Arc discrete or integrated graphics. Intel's GPU compute runtime (Level Zero + OpenCL ICD) is installed automatically from Intel's official graphics repo in the image — no manual package installation required. SigLIP uses CPU inference for now (Intel Extension for PyTorch has no Python 3.13 wheels yet; the `torch.xpu` path is wired and will activate automatically when they ship).
- **`OPENVINO_DEVICE` env var**: controls the OpenVINO execution provider device for the `:intel` variant. `CPU` (default) requires no device passthrough. `GPU` targets Intel Arc discrete and integrated graphics via Level Zero.
- **AMD and Intel device passthrough snippets in `compose.yml`**: documented as commented-out alternatives to the NVIDIA `deploy:` block.
- **`:rocm` and `:intel` CI jobs**: `docker-publish.yml` now builds and pushes `:rocm` / `:dev-rocm` and `:intel` / `:dev-intel` alongside `:latest` and `:cpu`. `release.yml` builds all four variants on tag push.

### Fixed

- **Interactive custom-count prompt firing for all choices**: in the no-embedding fallback path of the strategy selector, `IntPrompt.ask` was inside a dict literal and evaluated eagerly — users selecting Standard (30) or Broad (100) were still prompted to enter a custom image count. Each choice is now handled in a dedicated branch.

## [0.2.11] - 2026-06-12

### Fixed

- **GPU broken on x86_64 Linux**: `insightface` 1.0.1 (pulled in by the 0.2.10 lock update) added a hard dependency on the CPU `onnxruntime` package. Combined with an incorrect `override-dependencies` entry introduced in 0.2.10, both `onnxruntime` (CPU) and `onnxruntime-gpu` were being installed into the same venv. The CPU package landed last and overwrote the GPU one, causing `CUDAExecutionProvider` to disappear from the provider list even when a GPU was present. Fixed by declaring `onnxruntime` and `onnxruntime-gpu` as conflicting packages in uv's resolver, ensuring only the correct one is installed per platform.
- **OOM crash on large person libraries (CPU mode)**: All candidate thumbnails were downloaded into a single in-memory dict before any processing began. At ~5 MB per decoded preview image, a person with 472 candidates would accumulate ~2.4 GB of thumbnail data alone, exhausting a 4 GB container memory limit. Thumbnails are now downloaded and processed in batches of 32, with each image released immediately after embedding. Peak in-flight thumbnail memory is now bounded to ~256 MB regardless of candidate pool size. GPU users also benefit from lower host RAM pressure and faster time-to-first-result on large libraries.

## [0.2.10] - 2026-06-12

### Added

- **`VERBOSE` env var**: set `VERBOSE=true` to enable DEBUG-level console output. The log file always captures DEBUG; this flag controls what appears on the terminal. Useful when diagnosing issues without a full shell into the container.
- **`:cpu` Docker image tag**: a separate CPU-only image (`ghcr.io/sudolulo/winnow:cpu`) is now built and pushed alongside `:latest`. Uses `onnxruntime` instead of `onnxruntime-gpu`; ~2 GB smaller. Suitable for systems without an NVIDIA GPU.
- **Empty `CRON_SCHEDULE` keeps container alive**: setting `CRON_SCHEDULE=` (empty string) starts the container without running immediately and without exiting — useful for `docker exec` ad-hoc runs on a long-lived container. Previously, an empty value was treated the same as unset (run once, then exit).

### Changed

- **TTY auto-detection replaces `AUTO_MODE`**: winnow now detects whether a TTY is attached (`sys.stdin.isatty()`) and switches between interactive and auto mode automatically. `AUTO_MODE=true` becomes an explicit override for forcing auto mode in a terminal session. No config change needed for normal Docker deployments.
- **Logging levels audited**: internal algorithmic detail (clustering steps, asset fetch progress, per-page pagination) demoted from INFO to DEBUG. INFO now reflects meaningful pipeline milestones only (model ready, selection complete, quality filtered). Reduces noise in production logs without losing information.
- **Model load logging improved**: SigLIP and InsightFace loading now reports cache hit/miss, download size estimate, device used, and load time.

### Fixed

- **GPU OOM crash loop (production)**: `CUDAExecutionProvider` was silently absent even with a GPU attached, causing InsightFace to run on CPU and exhaust RAM processing large person libraries. Root cause: CUDA/cuDNN libraries in nvidia pip packages were invisible to onnxruntime. Fixed by running `ldconfig` over all `nvidia-*/lib/` directories in the venv at image build time.
- **ldconfig path now Python-version-agnostic**: the `find` command used to register nvidia pip libraries hardcoded `python3.13`; replaced with `python3.*` glob so the path survives a Python upgrade without silently producing an empty ldconfig config.
- **`PYTHONPATH=/app` added to Dockerfile**: the entry point script sets `sys.path[0]` to the script directory, not `/app`. Since `uv sync` runs before `COPY winnow/`, the wheel has only dist-info in site-packages. `PYTHONPATH=/app` makes the `winnow` package importable without reverting to `python -m`.
- **CPU fallback retrying broken GPU provider**: InsightFace CPU fallback omitted `providers=["CPUExecutionProvider"]`, causing onnxruntime to retry `CUDAExecutionProvider` on every inference call. Now explicitly sets the CPU provider and suppresses C-extension noise via fd-level redirect.
- **`_suppress_output` stderr loss on fd exhaustion**: if the first `os.dup2` in the finally block raised `OSError`, the second call was skipped, permanently redirecting stderr to `/dev/null` for the process lifetime. Wrapped in nested `try/finally` so both restores are always attempted.
- **Frigate `/api/faces` response parsing**: the response is `{person_name: [files], "train": [...]}` — `"train"` is a flat pending list, not a person. Previous code called `.items()` on the `"train"` value (a list), crashing with `AttributeError`. Now skips the `"train"` key explicitly.
- **Immich 401 detection**: a stale or invalid API key now logs a clear error message (`Immich API key is invalid or expired (401 Unauthorized)`) instead of raising an unhandled exception.
- **Falsy-zero detection confidence**: `face.get("score") or face.get("confidence")` treated a valid `score=0.0` as falsy, falling through to the `confidence` field (often `None`). Replaced with an explicit `None` check. Affected both quality filtering and hard-example weighting in diversity selection.
- **Face crop using wrong person's image dimensions**: in multi-person assets, `_crop_face_from_thumbnail`'s scale-factor loop matched the first person with any face regardless of `person_id`, producing incorrectly scaled bounding box coordinates for the target person. Loop now applies the same `person_id` filter as `_get_face_bbox`.
- **Adaptive stopping bypassed for partially-trained people**: in auto mode with `already_uploaded > 0`, `limit` was converted from `"auto"` to an integer, disabling the FPS adaptive threshold and early-stop check. Now keeps `limit="auto"` through selection and trims the result to the remaining capacity afterward.
- **Embedding cache key mismatch**: HuggingFace cache path check hardcoded the model slug string; replaced with a derivation from `model_name` using `"models--" + model_name.replace("/", "--")` so the check stays correct if the model name changes.
- **Scheduler: sleep until next run**: the loop slept a fixed 60 seconds regardless of schedule interval, causing runs to fire up to 59 seconds late and waking the process unnecessarily on long schedules (e.g. weekly). Now sleeps exactly until `next_run`.
- **Scheduler swallowing `SystemExit`**: `except BaseException` in the run wrapper was replaced with `except Exception` (with `KeyboardInterrupt` re-raised above), so `sys.exit()` calls propagate correctly.
- **Log handler leak**: `setup_logging` now closes and removes existing handlers before adding new ones, preventing file handle accumulation across repeated calls.
- **`RETRY_REJECTED` silently applied in interactive mode**: the env var was applied unconditionally even in interactive sessions. Now used only as the default for the interactive prompt so users can override it per-run.
- **`compose.yml` comment inverted**: a comment stated `-it` forces non-interactive mode; corrected to reflect that `-it` allocates a TTY (interactive mode).

### Security

- API key is no longer stored in any config file. All authentication uses environment variables or `.env` only.

## [0.2.9] - 2026-06-12

### Fixed

- **arm64: `onnxruntime-gpu` has no arm64 wheels**: `onnxruntime-gpu` only publishes `manylinux_2_27_x86_64` and `manylinux_2_28_x86_64` wheels — `uv sync` on arm64 failed with exit code 2. Gated `onnxruntime-gpu` behind `sys_platform == 'linux' and platform_machine == 'x86_64'`; arm64 and non-Linux installs now get the CPU `onnxruntime` package instead.
- **uv lockfile now covers arm64**: Added `required-environments` to `[tool.uv]` so the lockfile is solved for both `linux/x86_64` and `linux/aarch64`, preventing silent resolution gaps for the non-build platform.

## [0.2.8] - 2026-06-12

### Fixed

- **Dockerfile: use `add-apt-repository` with isolated GNUPGHOME**: the `curl | gpg --dearmor` approach was silently failing — gpg exits 0 on bad/empty input, leaving an invalid keyring and causing apt to skip the deadsnakes PPA entirely. Reverted to `add-apt-repository ppa:deadsnakes/ppa` with `GNUPGHOME=$(mktemp -d)` to prevent gpg from touching any pre-existing agent socket (the original QEMU crash cause).
- **arm64 base: removed `--platform=$BUILDPLATFORM`**: the Ubuntu 24.04 base for arm64 is now pulled for the target platform, so the arm64 image contains real arm64 binaries rather than amd64 binaries in an arm64 manifest.

### Changed

- **License changed from MIT to AGPLv3+**: source and network use of winnow now require derivative works to be open-sourced under the same terms.

## [0.2.7] - 2026-06-12

### Fixed

- **arm64 base reverted to Ubuntu 24.04**: Ubuntu 26.04 ships Python 3.14, not 3.13 — `python3.13` was not locatable in its repos. Reverted arm64 base to `ubuntu:24.04`.
- **Deadsnakes PPA now uses curl+gpg instead of `add-apt-repository`**: `add-apt-repository` spawns a gpg-agent which crashes under QEMU (arm64 CI). The PPA is now added by fetching the key via `curl` and piping through `gpg --dearmor` — no agent, works on both architectures.
- **PPA conditional removed**: both amd64 (Ubuntu 22.04 CUDA) and arm64 (Ubuntu 24.04) now go through the same deadsnakes install path, eliminating the per-arch branching and the `ARG TARGETARCH` dependency in RUN commands.

## [0.2.6] - 2026-06-12

### Fixed

- **Dockerfile: `ARG TARGETARCH` re-declared in each stage**: Docker's automatic platform ARGs are only in scope for `FROM` instructions, not `RUN` commands. The `$TARGETARCH` variable in the deadsnakes PPA conditional was silently empty, so the PPA was never added and `python3.13` could not be located on the Ubuntu 22.04 CUDA base. Adding `ARG TARGETARCH` at the top of both the `build` and `runtime` stage bodies fixes the amd64 build.

## [0.2.5] - 2026-06-12

### Changed

- **CUDA base upgraded to 13.3.0**: amd64 base image bumped from `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04` to `nvidia/cuda:13.3.0-cudnn-runtime-ubuntu22.04`. Python packages (torch, onnxruntime-gpu) still target CUDA 12.6 via pip-installed libraries; the base image upgrade requires a host GPU driver that supports CUDA 13+.
- **`torchvision>=0.27.0` added as explicit dependency**: routed through the `pytorch-cu126` index for linux/x86_64, ensuring it resolves to `0.27.0+cu126` (paired with `torch 2.12.0+cu126`) rather than being pulled from PyPI where the older `0.21.0` wheel would downgrade torch to 2.6.0.
- **`torch>=2.12.0`** floor raised from 2.6.0 to prevent silent downgrade.
- **`nvidia-cudnn-cu12>=9.0.0`** floor kept intentionally loose — torch pins an exact cuDNN ABI version (`9.10.2.21`) and must own that constraint.

## [0.2.4] - 2026-06-12

### Changed

- **Python 3.13 across all platforms**: bumped from 3.12 to 3.13. amd64 installs Python 3.13 from the deadsnakes PPA on the Ubuntu 22.04 CUDA base; arm64 uses the Python 3.13 package available natively in Ubuntu 26.04. All confirmed dependencies (`insightface 1.0.1`, `onnxruntime-gpu 1.26.0`, `torch 2.12.0+cu126`) have Python 3.13 wheels.
- **arm64 base: Ubuntu 26.04**: Python 3.12 was removed from Ubuntu 26.04's default repos; upgrading the base pulls Python 3.13 without a PPA.
- **`requires-python = ">=3.13"`** and `target-version = "py313"` updated in `pyproject.toml`.
- **CI updated to Python 3.13**: `test.yml` and `update-lockfile.yml` now install Python 3.13.
- **`uv.lock` regenerated** for Python 3.13.5.

## [0.2.3] - 2026-06-12

### Fixed

- **Clear-text API key storage**: `config.py` no longer writes `API_KEY` to `.immich_config.json`. The key must come from an environment variable or `.env` file. Interactive mode now prints a tip directing users to `.env`. Resolves CodeQL `py/clear-text-storage-sensitive-data`.

### Changed

- **CI workflow permissions**: `test.yml` and `lint.yml` now declare `permissions: contents: read`, following least-privilege principle and resolving `actions/missing-workflow-permissions` scanner alerts.
- **CI lockfile race condition**: removed the `verify-lockfile` pre-job from `docker-publish.yml` and `release.yml`. The `update-lockfile.yml` bot maintains the lockfile; the verify step raced against it on the same push event and caused false failures. `release.yml` now runs `uv lock` inline so tag-triggered builds are always self-consistent.
- **Docs moved to wiki**: `docs/` folder removed from the repository. Setup, Troubleshooting, and FAQ pages are now at the [GitHub wiki](https://github.com/sudolulo/winnow/wiki).

## [0.2.2] - 2026-06-12

### Added

- **Confidence scores in upload tracker**: Immich face confidence scores are now stored per asset in `frigate_uploaded_ids.json` under `by_person[name].scores`. Lays the groundwork for future replacement logic (remove low-confidence uploads when better images are found).
- **Frigate-authoritative capacity tracking**: At startup, `GET /api/faces` is queried on the Frigate host to retrieve the actual number of trained images per person from the `train` directory (pending/unclassified queue is excluded). This count is stored as `frigate_count` in the tracker JSON so it survives Frigate downtime.
- **Lifetime cap uses Frigate count**: `MAX_AUTO_IMAGES` is now enforced against Frigate's live training image count rather than the local uploaded-asset tally. Fallback priority: live Frigate API → last cached `frigate_count` in JSON → local uploaded count.
- **Startup summary shows Frigate count**: Tracker summary at startup now includes the last known Frigate training count per person (e.g. `78 uploaded, 2 rejected, 42 in Frigate`).
- **`winnow/frigate_api.py`**: new module encapsulating Frigate API helpers; currently exposes `get_frigate_face_counts()`.

### Changed

- `upload_tracker.py`: `by_person` entries migrated from flat list to `{asset_ids, scores, frigate_count}` dict. Old list format is read and migrated transparently on first write.
- `mark_uploaded()` now accepts an optional `score` keyword argument.
- `get_person_summary()` now returns `frigate_count` and `scores` fields alongside `uploaded` and `rejected`.

## [0.2.1] - 2026-06-12

### Fixed

- **Container startup reinstalling packages**: `entrypoint.sh` used `uv run`, which performs a sync check on every startup and re-downloaded `ruff` and rebuilt the package each time. Replaced with direct `.venv/bin/python` calls to skip the sync entirely.
- **InsightFace double `models/` path**: `INSIGHTFACE_HOME=/models` caused InsightFace to download Buffalo_L to `/models/models/buffalo_l` (InsightFace always appends `models/` to the root). Updated default in `compose.yml` and `.env.example` to `/models/.insightface`.
- **Lint errors in CI**: unused imports in `tests/test_config.py` and `tests/test_upload_tracker.py`, unsorted imports in `scheduler.py` — all would have failed the ruff CI check.

## [0.2.0] - 2026-06-12

First release of winnow. Forked from [if-curator](https://github.com/ds-sebastian/if_curator) by Sebastian and rewritten for headless Docker deployment.

### Added

**Headless operation**
- `AUTO_MODE` env var — runs without any interactive prompts; required for Docker/cron use
- `DRY_RUN` env var — previews selection without downloading, cropping, or uploading anything
- `RETRY_REJECTED` env var — re-attempts assets previously rejected by Frigate's face API
- `RESET_PERSON` env var — clears upload and rejection history for one named person

**Docker and scheduling**
- `Dockerfile` — multi-stage build (CUDA 12.9 on amd64, plain Ubuntu on arm64); runtime stage excludes build tools (g++, python3.12-dev, curl, gnupg)
- `compose.yml` — fully annotated with inline comments grouped by concern
- `entrypoint.sh` — runs the tool once on startup, then hands off to the scheduler if `CRON_SCHEDULE` is set
- `scheduler.py` — in-process cron scheduler that keeps the container (and loaded models) alive between runs
- `CRON_SCHEDULE` env var — standard cron expression for recurring runs; unset exits after first run
- `.dockerignore` — keeps `.venv`, `__pycache__`, test files, and logs out of the image context
- Multi-arch image: `linux/amd64` and `linux/arm64` built and merged into a single manifest on GHCR
- `tini` as PID 1 init process for correct signal handling
- Non-root container user (`appuser`, uid 568)
- `HEALTHCHECK` in Dockerfile

**Object mode**
- `TRAINING_MODE=object` — runs YOLOv9c detection on full images, crops each detected instance of a target class, and saves crops to the output volume (Frigate has no training API for objects — crops are placed manually)
- `OBJECT_CLASS` env var — target YOLO class label (e.g. `dog`, `cat`, `car`); defaults to `dog`

**People filtering**
- `ONLY_PEOPLE` env var — comma-separated whitelist; only these people are processed
- `SKIP_PEOPLE` env var — comma-separated list; these people are skipped
- `MIN_FACE_COUNT` env var — skip people with fewer than N tagged assets in Immich
- `YEARS_FILTER` env var — ignore assets older than N years (default: 10)

**Image quality controls** (previously hardcoded)
- `BLUR_THRESHOLD` env var — Laplacian variance threshold for blur rejection
- `MIN_CONFIDENCE` env var — minimum Immich face detection confidence
- `MAX_AUTO_IMAGES` env var — hard cap on auto-diversity selection
- `FACE_MARGIN` env var — padding around bounding box crops as a fraction of face size
- `USE_FULL_RESOLUTION` env var — download full-res originals vs preview thumbnails
- `ENABLE_FACE_ALIGNMENT` env var — align to ArcFace 112×112 format via InsightFace landmarks

**GPU and model configuration**
- `FORCE_CPU` env var — disable GPU; fall back to CPU for embedding computation
- `INSIGHTFACE_HOME` env var — controls model persistence for Buffalo_L
- `HF_HOME` env var — HuggingFace model cache path for SigLIP
- `LD_LIBRARY_PATH` set in the image to expose CUDA and cuDNN pip libraries so `onnxruntime-gpu` can find them at runtime

**Caching and upload tracking**
- `ENABLE_CACHE` / `CACHE_DIR` env vars — opt-in embedding cache to skip recomputation on reruns
- Per-person upload tracker persisted as JSON; prevents the same asset from being uploaded twice across runs weeks apart, even if the container is recreated
- Startup summary showing uploaded and rejected counts per person
- `LIMIT` env var — exact image count overriding `STRATEGY` preset

**CI/CD**
- `docker-publish.yml` — builds multi-arch image and pushes to GHCR on push to `main` (`:latest`) or `dev` (`:dev`)
- `release.yml` — triggered by `v*` tags or `workflow_dispatch`; creates a GitHub Release, extracts changelog notes, builds and pushes versioned image to GHCR
- `lint.yml` — runs Ruff on push/PR to `main` and `dev`
- `test.yml` — runs pytest on push/PR to `main` and `dev`
- `update-lockfile.yml` — regenerates `uv.lock` and commits it when `pyproject.toml` changes
- Dependabot: weekly grouped PRs for Python dependencies (uv ecosystem) and GitHub Actions versions

**Testing**
- 24 unit tests across four modules: `test_config`, `test_immich_api`, `test_jobs`, `test_upload_tracker`

**Documentation**
- `docs/setup.md` — step-by-step install and GPU passthrough guide
- `docs/troubleshooting.md` — common failure modes with fixes
- `docs/faq.md` — answers to questions new users will ask
- `.env.example` — copy-paste starting point with every env var and inline comments
- README rewritten: pipeline diagram, env var reference tables, scheduling behaviour, requirements

### Changed

- `cli.py` split into three focused modules — `cli.py` (entry point), `jobs.py` (configuration and strategy resolution), `executor.py` (download, crop, upload)
- Dependency management replaced with [`uv`](https://astral.sh/uv); `uv.lock` pins the full transitive graph for reproducible builds
- `compose.yml` fully annotated; all env vars documented with inline comments
- `LD_LIBRARY_PATH` extended to include both cuDNN and CUDA runtime libraries

### Fixed

- **EXIF orientation**: PIL opens JPEGs without applying rotation metadata; Immich computes face bounding boxes on orientation-corrected images, so portrait photos produced misaligned crops. `ImageOps.exif_transpose()` now normalizes orientation before any coordinate math.
- **Model persistence**: `FaceAnalysis` was initialized with `root="~/.insightface"` (hardcoded), ignoring `INSIGHTFACE_HOME`. Buffalo_L was re-downloaded into the container on every run instead of persisting to the mounted volume.
- **Upload deduplication**: `upload_to_frigate()` scanned the output directory with `os.listdir()`, picking up leftover files from previous runs and re-uploading them. Now only files created in the current run are uploaded.
- **RGBA images**: Images in RGBA mode raised an error when encoding to JPEG. All images are now converted to RGB before saving.
- **Object mode uploads**: Object mode incorrectly called the Frigate face registration API. Frigate has no API for object training data — object mode now only saves crops to disk.
- **Stale output files**: The output directory was not cleaned between runs, causing crops to accumulate. Now wiped at the start of each face-mode run.
- **JSON decode errors**: `get_people()` and `fetch_all_assets()` only caught `RequestException`, leaving `JSONDecodeError` unhandled on non-JSON 200 responses from Immich.
- **Spaces in names**: People names with spaces caused downstream errors.
- **Inconsistent headers**: Some API calls used a raw header dict instead of `get_headers()`.
- **Docker layer caching**: `uv sync` was placed after `COPY if_curator/`, so any source change invalidated the 800 MB dependency cache. Dependencies are now installed before source is copied.

### Security

- CUDA base image bumped from `nvidia/cuda:12.6.3` to `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04`, picking up Ubuntu security patches flagged by Dependabot.
