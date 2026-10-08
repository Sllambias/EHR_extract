# filter_tables_on_hashes.py / split_tables_on_hashes.py

Two ways to take a cohort and pull the raw source tables down to it. Both read the same keys; they
differ in output shape and in which per-table fields they honour.

```bash
# one CSV per source table, filtered to the cohort
python EHR_extract/filter_tables_on_hashes.py --config-name template_filter_tables

# one .xlsx per patient, one worksheet per source table
python EHR_extract/split_tables_on_hashes.py --config-name template_split_tables
```

Templates: [`template_filter_tables.yaml`](../configs/templates/template_filter_tables.yaml),
[`template_split_tables.yaml`](../configs/templates/template_split_tables.yaml).

The population is any CSV with an ID column — usually the `_population_train.csv` from
[extract.py](extract.md).

## Shared keys

```yaml
paths:
  output_dir: ${oc.env:EHR_EXTRACT_OUTPUTS}/tables_filtered/${hydra:job.config_name}
  population_table: /path/to/population.csv

population_id_column: CPR_MOR   # the ID column in population_table
max_ids: 30                     # sample this many IDs (seed 4215); null = all

tables:
  - table: ${paths.input_dir_SP}/Mor - CPMI - Diagnoseliste.csv
    id_col: MOR_CPR             # the matching ID column in this table
    time_col: Noteret_dato      # split_tables only
    columns:                    # split_tables only; null = all columns
      - Diagnosekode
```

`paths.output_dir` must be set: neither script creates its own directory, so both rely on Hydra
making the run directory that `configs/default.yaml` points there.

## Which fields each script honours

| Field | `filter_tables_on_hashes.py` | `split_tables_on_hashes.py` |
|---|---|---|
| `table`, `id_col` | yes | yes |
| `time_col` | **ignored** | sorts rows, newest first |
| `columns` | **ignored** | selects columns (plus `id_col` and `time_col`) |
| `filters`, `time_window` | **ignored** | keep only matching rows — see [below](#row-filters-and-time-windows) |
| `max_ids` | yes (samples rows) | yes (samples distinct IDs) |

`filter_tables_on_hashes.py` writes every column of each table regardless of what `columns` says.
The template omits both fields to avoid implying otherwise.

## filter_tables_on_hashes.py

Inner-joins each table to the population on `id_col` = `population_id_column` and writes
`<output_dir>/<original filename>.csv`. Rows for IDs outside the cohort are dropped; every column
survives.

Use it to hand a collaborator a self-contained slice of the source tables.

## split_tables_on_hashes.py

Writes `<output_dir>/<patient_id>.xlsx`, one file per patient in the population, with one worksheet
per entry in `tables`. Rows are filtered to that patient and, when `time_col` is set, sorted newest
first.

Worksheet names are the source filename with `" - CPMI"` removed and truncated to Excel's 31
characters ([split_tables_on_hashes.py:22-23](../EHR_extract/split_tables_on_hashes.py#L22-L23)), so
sources whose names collide after truncation will collide as sheet names.

Use it for manual chart review — one file per patient, their whole record in tabs.

Note this writes one file per distinct patient in the population. Set `max_ids` to cap that — it
samples that many IDs with seed 4215, and is capped at the number available.

### Row filters and time windows

Each table may also take `filters` and `time_window`, to show only the rows relevant to the review.
Both are applied before `columns`, so they may use columns that are not exported.

- `filters` — a list of `{column, operator, value}`, all of which a row must pass. Operators as in
  [reference.md](reference.md#operators).
- `time_window` — the row's `time_col` must fall inside the named window, inclusive at both ends.

Windows are defined once at the top level in `time_conditionals` and referenced by name, in the same
format `table.py` uses. Each bound is a column of `population_table` plus `offset_days`, so every
patient gets their own window; `date_col: null` leaves that side open.

```yaml
time_conditionals:
  pregnancy:
    min_date: {date_col: BIRTHDAY, offset_days: -300}
    max_date: {date_col: BIRTHDAY, offset_days: 0}
  up_to_birth:
    min_date: {date_col: null, offset_days: 0}
    max_date: {date_col: BIRTHDAY, offset_days: 0}

tables:
  - table: ${paths.input_dir_SP}/Mor - CPMI - Diagnoseliste.csv
    id_col: MOR_CPR
    time_col: Noteret_dato
    columns:
    filters:
      - {column: Diagnosekode, operator: startswith_any, value: ["DO"]}
    time_window: pregnancy
```

The population usually has one row per child, so a mother with several births has several windows.
A row is kept if it falls inside **any** of them, and is written once even when windows overlap, as
they do for twins.

Dates are parsed as in `table.py` (`YYYY-MM-DD`, optionally with a time). A row whose `time_col`, or
whose patient's bound column, is missing or unparseable is dropped. The bound columns are always read
from `population_table`, never from the source table, even if it has a column of the same name. A
table with `time_window` must set `time_col`.

A runnable example on the local fixtures is
[`configs/testing/test_split_table_time_window.yaml`](../configs/testing/test_split_table_time_window.yaml).
