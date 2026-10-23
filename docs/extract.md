# extract.py — building a cohort

```bash
python EHR_extract/extract.py --config-name template_population
```

Builds a population table from one or more sources, applies inclusion/exclusion criteria in order,
and writes the survivors plus a record of everything discarded. A criterion can also add a column
instead of filtering.

Template: [`configs/templates/template_population.yaml`](../configs/templates/template_population.yaml),
[`template_population_images.yaml`](../configs/templates/template_population_images.yaml).

## Order of operations

[extract.py:235-282](../EHR_extract/extract.py#L235-L282)

1. `population.tables` — stack sources
2. `population.composed_tables` — join-then-stack sources
3. `population.deduplication_key` — one row per key (but see the caveat below)
4. `conditional_criteria` — include/exclude, or add a column
5. `imaging_table` — attach images
6. `imaging_matching_criteria` — filter images
7. write

## Required keys

These are read by attribute, so they must be present. They may be `null` where noted.

| Key | Note |
|---|---|
| `paths.output_dir` | also Hydra's run dir |
| `paths.population_save_path` | suffixes are appended to it |
| `paths.discards_save_path` | |
| `paths.holdout_csv` | may be `null` |
| `strict` | `False` sets `ignore_errors=True` on the CSV reader |
| `population.population_key` | the ID the cohort is defined on |
| `population.deduplication_key` | may be `null` |
| `population.split_key` | only when `holdout_csv` is set |

`population.file_path_key` is **not** required and is read by no Python code. It exists so criteria
can interpolate `${population.file_path_key}`; the code hardcodes `FILE_PATH`
([extract.py:193](../EHR_extract/extract.py#L193)).

## Sources

`columns` is always `NEW_NAME: OLD_NAME`. Renaming to shared names is what lets sources with
different column names stack.

### `population.tables` — stack

Each table is reduced to `columns`, renamed, and appended. The result has as many rows as the sum of
its sources. Rows whose `GA` is not numeric are dropped
([utils.py:171-172](../EHR_extract/utils/utils.py#L171-L172)).

```yaml
population:
  population_key: CPR_BARN
  tables:
    - table: ${paths.input_dir_SDS}/mfr.csv
      columns: {CPR_BARN: CPR_BARN, CPR_MOR: CPR_MODER, GA: GESTATIONSALDER_DAGE, BIRTHDAY: FOEDSELSDATO}
    - table: ${paths.input_dir_SDS}/nyfoedte.csv
      columns: {CPR_BARN: CPRnummer_Barn, CPR_MOR: CPRnummer_Mor, GA: Gestationsalder, BIRTHDAY: FoedselsDato_Barn}
```

### `population.composed_tables` — join, then stack

Use when one record needs columns from several tables. Tables in an entry are joined on `merge_on`
before being appended to the population. Every entry must set `format_SP_GA`.

`format_SP_GA: True` converts Sundhedsplatformen's GA strings to total days: `"38 2"` → `268`, a
bare `"38"` → `266`. It expects **weeks and days separated by a space** and reads only the first
character of the second token; anything else yields an empty string, and a token like
`"38 weeks 2 days"` raises ([utils.py:270-282](../EHR_extract/utils/utils.py#L270-L282)). Check what
your source column actually contains before enabling it.

```yaml
population:
  composed_tables:
    - merge_on: CPR_BARN
      format_SP_GA: True
      tables:
        - table: ${paths.input_dir_SP}/Population.csv
          columns: {CPR_BARN: BABY_CPR, CPR_MOR: MOR_CPR, BIRTHDAY: Fødselstidspunkt}
        - table: ${paths.input_dir_SP}/Barn - Fødselsinfo.csv
          columns: {CPR_BARN: BABY_CPR, GA: Gestationsalder}
```

Both blocks may be used together; `tables` is stacked first.

### `deduplication_key`

Collapses stacked sources to one row per key, keeping the row with the **fewest** nulls
([utils.py:157-162](../EHR_extract/utils/utils.py#L157-L162)).


## conditional_criteria

A list applied in order. `action: include` keeps only the matched IDs; `action: exclude` drops them;
`action: add_column` keeps everyone and records the match in a new column
([below](#adding-a-column-instead-of-filtering)).

```yaml
conditional_criteria:
  - action: include
    conditions:
      - standard:
        table: population      # "population" = the table built above, or a path to a CSV
        match_on: CPR_BARN     # the ID column in that table
        column: GA
        operator: between
        value: [154, 258]
```

### The indentation is deliberate

`standard:` carries the boolean connective **as its own value**, and the other keys are its siblings
at the same indent. `standard:` with nothing after it is `standard: null`, meaning "first condition
of a group". It looks like a YAML mistake; it is not.

```yaml
- standard:            # null  -> starts a group
  table: population
  ...
- standard: and        # intersect with the group so far
  ...
- standard: or         # close the group, start a new one
  ...
```

`and` intersects, `or` unions the accumulated group into the result. The net effect is an **OR of
AND-groups — `and` binds tighter than `or`**
([extract.py:146-162](../EHR_extract/extract.py#L146-L162)).

Omitting `operator` matches every row of the table whose `match_on` is in the population — useful for
"appears in this table at all".

Available operators and their casting rules: [reference.md](reference.md#operators).

#### Example: exclude subjects with no GA recorded anywhere

Three sources, OR'd, so a subject is dropped only if GA is missing in all of them.

```yaml
- action: exclude
  conditions:
    - standard:
      table: ${paths.input_dir_SDS}/mfr.csv
      match_on: CPR_BARN
      column: GESTATIONSALDER_DAGE
      operator: "missing"
      value:
    - standard: and
      table: ${paths.input_dir_SDS}/nyfoedte.csv
      match_on: CPRnummer_Barn
      column: Gestationsalder
      operator: "missing"
      value:
    - standard: and
      table: ${paths.input_dir_SP}/Barn - Fødselsinfo.csv
      match_on: BABY_CPR
      column: Gestationsalder
      operator: "missing"
      value:
```

### Custom conditions

`custom:` carries the connective the same way. `population` and `population_key_column` are injected
by the script — do not put them in `args`.

```yaml
- action: include
  conditions:
    - custom:
      function: match_years_with_child_cpr_on_birthdate
      args:
        date_start: "01012024"
        date_end: "31122024"
        value_table_path: ${paths.input_dir_SP}/Population.csv
        value_time_column: Fødselstidspunkt
        value_child_cpr_column: BABY_CPR
```

Functions and their arguments: [reference.md](reference.md#custom-functions-for-extractpy).

### Adding a column instead of filtering

`action: add_column` keeps every row and adds a column called `name`: `True` where the row's
`population_key` matched the conditions, `False` elsewhere.

```yaml
- action: add_column
  name: preterm
  conditions:
    - standard:
      table: population
      match_on: CPR_BARN
      column: GA
      operator: "<"
      value: 259
```

Give a standard condition `retrieve: <column>` to get that column's value from the matching row
instead of `True`. Rows that did not match are left empty.

```yaml
- action: add_column
  name: caesarean_code
  conditions:
    - standard:
      table: ${paths.input_dir_SDS}/nyfoedte.csv
      match_on: CPRnummer_Barn
      column: Kejsersnit
      operator: "!="
      value: "-1"
      retrieve: Kejsersnit
    - standard: or             # fallback for children not matched above
      table: ${paths.input_dir_SDS}/mfr.csv
      match_on: CPR_BARN
      column: MARKOER_KEJSERSNIT
      operator: "!="
      value: "-1"
      retrieve: MARKOER_KEJSERSNIT
```

- **Several matching rows for one ID**: set `date_col` on the condition to keep the value with the
  latest date. Without it, different values for one ID raise an error. Empty values are skipped.
- **Several conditions with `retrieve`**: the first one, in config order, that matched the ID and has
  a value wins, so `or` conditions work as fallbacks. A value only counts if its whole `and` group
  matched the ID. Sources with different types are cast to a common one (numbers and text give text).
- **Conditions without `retrieve`** still decide which IDs match, so a custom condition can narrow a
  lookup. `retrieve` on a custom condition raises: those functions only return IDs.
- **`name`** must not already be a population column. Later criteria can use the new column with
  `table: population`, e.g. `operator: is_true` on a flag.
- The column is carried through image matching into every output CSV. The criterion is listed in the
  discards JSON with `n_discards: 0`.

`include` and `exclude` ignore `retrieve` and `date_col`.

## Matching images

Images are recorded against the **mother**, so matching them to a child is a two-step process.

`imaging_table` joins the image table on the maternal ID, then keeps only images whose `STUDY_DATE`
falls in `[BIRTHDAY - GA days, BIRTHDAY]` — the pregnancy window. The output gains one row per
(child, image): a twin pregnancy's images appear once per twin, and a child appears as many times as
the mother had scans.

```yaml
imaging_table:
  table: ${paths.input_dir_img_db}/all_images_2026-05-29.csv
  columns:
    CPR_MOR: phair_hash    # must match the population's maternal ID column
    FILE_PATH: file_path
    STUDY_DATE: study_date
```

`imaging_matching_criteria` then filters those rows in order. Unlike `conditional_criteria` these
functions filter images rather than IDs, and each records its own discard statistics.

```yaml
imaging_matching_criteria:
  - function: find_images_within_time_windows
    args:
      scan_date_column: STUDY_DATE
      image_path_column: ${population.file_path_key}
      population_delivery_date_column: BIRTHDAY
      population_ga_in_days_at_delivery_column: GA
      min_diff_days_scan_to_delivery: 0
      max_diff_days_scan_to_delivery: 210
      min_ga_in_days_at_scan: 112
      max_ga_in_days_at_scan: 168
```

## Outputs

| File | Contents |
|---|---|
| `<population_save_path>_train_and_test.csv` | the full surviving population |
| `<population_save_path>_train.csv` / `_test.csv` | only when `paths.holdout_csv` is set |
| `<discards_save_path>` | JSON, one entry per criterion |

The train/test split assigns rows by `population.split_key` against the IDs in `holdout_csv`. If a
subject ends up in both, it is removed from **test** and a warning is logged
([extract.py:275-280](../EHR_extract/extract.py#L275-L280)).
`paths.exclude_train_subjects_not_in_csv`, if set, further restricts train to IDs in that CSV.

The discards JSON records, per criterion, the criterion as written, `n_discards`, the population
before and after, and the discarded IDs — so a cohort's funnel is auditable:

```json
{
  "0": {
    "criteria": {"action": "include", "conditions": [...]},
    "n_discards": 49,
    "n_population_pre_discard": 1401,
    "n_population_post_discard": 1352,
    "discards": ["..."]
  }
}
```
