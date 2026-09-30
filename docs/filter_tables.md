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
| `max_ids` | yes | **no effect** (see README Known limitations) |

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
characters ([split_tables_on_hashes.py:16-17](../EHR_extract/split_tables_on_hashes.py#L16-L17)), so
sources whose names collide after truncation will collide as sheet names.

Use it for manual chart review — one file per patient, their whole record in tabs.

Note this writes one file per patient in the population, and `max_ids` does not currently limit that.
Point `population_table` at a small CSV when testing.
