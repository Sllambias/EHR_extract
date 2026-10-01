# Known limitations

Bugs found while writing these docs, kept here so they are discoverable from the scripts they
affect. Each is reproducible on the current `main`.

## Open

1. **`summary.py`'s CLI is unusable as shipped.** `summary_from_cfg` requires a `base_population`
   block ([summary.py:36](../EHR_extract/summary.py#L36)) that no config in the repo defines. Its
   working role is supplying `get_summary()` to [`table.py`](table.md).
2. **`filter_tables_on_hashes.py` ignores `time_col` and `columns`** and writes every column of each
   table ([filter_tables_on_hashes.py:24-29](../EHR_extract/filter_tables_on_hashes.py#L24-L29)).
   See [docs/filter_tables.md](filter_tables.md#which-fields-each-script-honours).
3. **`custom_split_fn: kfold` writes its fold CSVs, then crashes.** `kfold` has no `return`, so
   unpacking its result at [split.py:100](../EHR_extract/split.py#L100) raises on `None`. The files
   are already on disk by then. See [docs/split.md](split.md).
4. **`configs/testing/test_SL_ehr_table.yaml` inherits `SL_EHR_noMP_img_V5@`, which does not exist.**
5. **`configs/tables_merged/EHR_SP.yaml` is not valid YAML.** The `table:` block under the second
   `conditional_bool_criteria` condition is indented one level too deep, so the file fails to parse
   (`line 1233, column 16`). The other eight `configs/tables_merged/*` configs compose cleanly.
6. Many configs hardcode machine-specific absolute paths (`/Users/zcr545/...`,
   `/projects/users/data/UCPH/...`, `/storage/archive/...`).

## Fixed

Listed because two of them change output, so tables regenerated now will not match ones built
before the fix.

- **`deduplicate_on_key` kept the least complete row.** It counts nulls per row as a completeness
  tiebreak but sorted them descending, so the row with the *most* nulls won
  ([utils.py:157-162](../EHR_extract/utils/utils.py#L157-L162)). It now keeps the row with the
  fewest. **Changes output** for every config that sets `population.deduplication_key`.
- **`split_tables_on_hashes.py`'s `max_ids` had no effect** — the sample was drawn after the patient
  list was built. It now caps the export at that many distinct IDs, sampled with seed 4215.
  **Changes output** for any config that sets `max_ids`: the script previously exported every
  patient.
- **`table.py` could not run any `configs/tables_merged/*` config.** It read `cfg.allow_duplicates`
  by attribute, and none of those configs define it; with `allow_duplicates: False` it also called
  `check_duplicates()` with a kwarg that function does not accept. `allow_duplicates` is now
  optional and defaults to `False` — see [docs/table.md](table.md).
- **`utils/custom_split_functions.py` imported `from extract import ...`** rather than
  `from EHR_extract.extract import ...`, so `EHR_extract` was not importable as a package; it only
  resolved because running a script by path puts `EHR_extract/` on `sys.path`.
