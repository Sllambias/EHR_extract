# table.py — building a feature table

```bash
python EHR_extract/table.py --config-name template_table
```

Turns a population into one wide row per subject: a base table of key columns, then one column per
criterion, each drawn from source tables within a named time window.

Template: [`configs/templates/template_table.yaml`](../configs/templates/template_table.yaml).
Production configs: [`configs/tables_merged/`](../configs/tables_merged/).

> `allow_duplicates: True` is required at the top level. See the README's Known limitations — no
> `configs/tables_merged/*` config sets it, and `False` currently raises.

## Order of operations

[table.py:219-227](../EHR_extract/table.py#L219-L227): `base_table` → `extract_criteria` →
`conditional_bool_criteria` → `custom_extract_criteria` → optional summary.

## Nested joins

Anywhere a `table:` is expected, you may give either a path or a join spec. `table1` is left-joined
to `table2`, and specs nest recursively
([utils.py:51-94](../EHR_extract/utils/utils.py#L51-L94)). This is how maternal tables get a child ID
attached before matching.

```yaml
table:
  table1: ${paths.input_dir_SP}/Mor - CPMI - Diagnoseliste.csv
  table2: ${paths.input_dir_SP}/Population.csv
  left_on: [MOR_CPR]
  right_on: [MOR_CPR]
```

## base_table

The skeleton: which subjects, and the key columns every criterion joins against.

```yaml
base_table:
  population: ${paths.population}     # a CSV, e.g. extract.py's output
  population_column: b_cpr            # the ID column, after renaming
  tables:
    - table: {table1: ..., table2: ..., left_on: [...], right_on: [...]}
      columns: {MOR_CPR: m_cpr, BABY_CPR: b_cpr, Fødselstidspunkt: pregnancy_end, Gestationsalder: GA_weeks}
  key_columns: [b_cpr, m_cpr, GA_weeks, pregnancy_end]
  dtypes: {b_cpr: string, m_cpr: string, GA_weeks: string, pregnancy_end: date}
  add_columns: [...]
```

Each source is reduced to `key_columns`, filtered to the population, and stacked. Every key column is
then cast to its `dtypes` entry and **rows that fail to cast are dropped** and recorded in the
discards JSON. Valid dtypes: `string`, `integer`, `float`, `boolean`, `date`, `datetime`
([utils.py:177-191](../EHR_extract/utils/utils.py#L177-L191)).

`add_columns` derives columns in order, each also cast and null-dropped. Typically GA is parsed to
days and the pregnancy start is back-calculated, because the time windows below are expressed
relative to it:

```yaml
add_columns:
  - column: GA_days
    function: find_GA_days
    args: {GA_weeks_col: GA_weeks, GA_days_col: GA_days}
    dtype: integer
  - column: pregnancy_start
    function: find_pregnancy_start
    args: {birth_date_col: pregnancy_end, GA_days_col: GA_days, pregnancy_start_col: pregnancy_start}
    dtype: date
```

Functions: [reference.md](reference.md#custom-functions-for-tablepy).

## time_conditionals

Named date windows, referenced by name from the criteria blocks. Bounds are a column plus an offset
in days; `date_col: null` means unbounded on that side.

```yaml
time_conditionals:
  before_pregnancy:
    min_date: {date_col: null, offset_days: 0}
    max_date: {date_col: pregnancy_start, offset_days: -1}
  during_pregnancy:
    min_date: {date_col: pregnancy_start, offset_days: 0}
    max_date: {date_col: pregnancy_end, offset_days: 0}
  year_before_pregnancy_and_during_pregnancy:
    min_date: {date_col: pregnancy_start, offset_days: -365}
    max_date: {date_col: pregnancy_end, offset_days: 0}
```

The bound columns must exist on the base table — hence deriving `pregnancy_start` in `add_columns`.

## extract_criteria — latest value as a column

Joins a column in and keeps the most recent value per key, by `date_col`. Multiple `sources` are
stacked, so one column can be filled from several tables.

```yaml
extract_criteria:
  - name: ethnicity
    key_column: m_cpr
    dtype: string
    sources:
      - table: ${paths.input_dir_SP}/Mor - Patientinfo tillæg til CPMI.csv
        match_on: MOR_CPR
        column: Etnicitet
        date_col: "Dato for seneste opdatering"
```

No time window: this block always takes the latest value.

## conditional_bool_criteria — a boolean column

True when a matching row exists inside `time_window`. This is how diagnoses and procedures become
flags.

```yaml
conditional_bool_criteria:
  - name: smoking_status
    key_column: b_cpr           # the column the flag is written against
    match_on: m_cpr             # the base-table column joined to the source
    time_window: before_and_during_pregnancy
    conditions:
      - condition:              # null / and / or
        table: {table1: ..., table2: ..., left_on: [MOR_CPR], right_on: [MOR_CPR]}
        match_on: BABY_CPR      # the source column joined against
        column: Diagnosekode
        operator: startswith
        value: ["BKUA32"]
        date_col: Noteret_dato
```

`condition:` groups exactly like `standard:` in `extract.py` — `null` starts a group, `and`
intersects, `or` closes it. See [extract.md](extract.md#the-indentation-is-deliberate).

## custom_extract_criteria — arbitrary values

For anything the two blocks above cannot express: filtered values, conditional lookups, derived
quantities. `main_table`, `min_date`, `max_date` and `allow_duplicates` are injected from
`time_window` — do not put them in `args`.

```yaml
custom_extract_criteria:
  - name: alcohol_consumption
    time_window: before_and_during_pregnancy
    function: extract_latest_value
    args:
      table: {table1: ..., table2: ..., left_on: [MOR_CPR], right_on: [MOR_CPR]}
      left_on: b_cpr
      right_on: BABY_CPR
      target_col: "Alkohol - Antal ugentlige genstande"
      new_col_name: alcohol_consumption
      date_col: "Dato for seneste opdatering"
      dtype: string
```

`[]` is a valid value. Functions: [reference.md](reference.md#custom-functions-for-tablepy).

## summary_table

```yaml
summary_table:
  make_table: True
  ignore_columns: [b_cpr, m_cpr, pregnancy_end, pregnancy_start]
  n_samples: 100000
```

Per column: null count, null percentage, and a summary — the mean for numerics, the true-percentage
for booleans, a value-count percentage map otherwise. `n_samples` larger than the table is clamped,
with a warning that percentages are out of the actual row count.

`make_table: True` requires a column named exactly `GA`: the summary is also written split at
`GA < 259` days ([table.py:238](../EHR_extract/table.py#L238)). Without that column the run fails, so
the template leaves it `False`.

## Outputs

| File | When |
|---|---|
| `table_<name>.csv` | always |
| `discards_<name>.json` | always — per key column, how many subjects it dropped and which |
| `summary_<name>.csv` | `make_table: True` |
| `ptb_summary_<name>.csv`, `non_ptb_summary_<name>.csv` | `make_table: True`, split at `GA < 259` |
