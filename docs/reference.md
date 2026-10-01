# Reference — operators and custom functions

## Operators

Used by `conditional_criteria` ([extract.py](extract.md)) and `conditional_bool_criteria`
([table.md](table.md)). Defined at
[utils.py:97-132](../EHR_extract/utils/utils.py#L97-L132).

| Operator | `value` | Casts the column to |
|---|---|---|
| `==`, `!=` | a string | **String** |
| `>`, `<`, `>=`, `<=` | a number | **Float64** |
| `between` | `[low, high]`, inclusive | **Float64** |
| `in`, `not_in` | a list | — |
| `startswith` | a string | — |
| `startswith_any` | a list of prefixes | String |
| `missing`, `not missing`, `not_null` | unused, leave blank | — |
| `is_true`, `is_false` | unused, leave blank | Boolean |

The casts are the two easy mistakes:

- `==` and `!=` cast the column to String, so a numeric `value` **raises**
  (`cannot compare string with numeric type`). Quote it: `value: "1"`, not `value: 1`.
- `>`, `<`, `>=`, `<=` and `between` cast the column to Float64, turning non-numeric cells into
  nulls, which then silently fail the comparison instead of erroring.

Omitting `operator` entirely in a `standard` condition matches every row whose `match_on` is in the
population — i.e. "present in this table at all".

## Custom functions for extract.py

Registered at [extract.py:38-49](../EHR_extract/extract.py#L38-L49). `population` and
`population_key_column` are injected by the script — never put them in `args`.

Where each is legal depends on what it returns.

### In `conditional_criteria` — return a set of IDs

| Function | `args` |
|---|---|
| `find_close_births` | `value`, `operator`, `table`, `match_on`, `mom_column`, `birth_id_column`, `delivery_date_column` |
| `find_duplicated_ids` | `table`, `match_on`, `id_columns` |
| `match_years_with_child_cpr_on_birthdate` | `date_start`, `date_end` (both `"DDMMYYYY"`), `value_table_path`, `value_time_column`, `value_child_cpr_column` |
| `match_value_with_child_cpr_on_birthdate` | `operator`, `value`, `value_table_path`, `value_column`, `value_time_column`, `value_mother_cpr_column`, `population_mother_cpr_column`, `population_child_cpr_column`, `population_birth_column`, `population_gestational_age_column`, optional `include_days_after_birth` (default 0) |
| `match_value_with_child_cpr_on_birth_id` | `operator`, `value`, `value_table_path`, `value_column`, `value_table_birth_id_column`, `mapping_table_path`, `mapping_table_birth_id_column`, `mapping_table_child_cpr_column` |
| `match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate` | as `..._on_birthdate` plus `value_id_column`, `mapping_table_path`, `mapping_table_id_column`, `mapping_table_mom_cpr_column`, `population_mom_cpr_column` |

The three `match_value_*` functions exist because maternal records carry no child ID. Each is a
different route from a maternal row back to a child: via the mother's CPR and the birth date, via a
birth ID, or via an LPR contact ID mapped to the mother's CPR and then the birth date. Pick the one
whose join keys your source table actually has.

### In `imaging_matching_criteria` — return `(population, discard_stats)`

| Function | `args` |
|---|---|
| `find_images_within_time_windows` | `scan_date_column`, `image_path_column`, `min_diff_days_scan_to_delivery`, `max_diff_days_scan_to_delivery`, `min_ga_in_days_at_scan`, `max_ga_in_days_at_scan`, optional `population_delivery_date_column` (default `BIRTHDAY`), `population_ga_in_days_at_delivery_column` (default `GA`) |
| `find_images_with_predicted_classes` | `table`, `classes` (list), `class_column`, `image_path_column`, `population_image_path_column` |

Using one of these in `conditional_criteria`, or one of the set-returning functions in
`imaging_matching_criteria`, will fail — the return types are not interchangeable.

### Not called directly

- `match_images_with_child` — invoked automatically by the `imaging_table` block.
- `merge_population_on` — registered, but returns a DataFrame rather than a set of IDs and is used by
  no config in the repo.

## Custom functions for table.py

Registered at [table.py:37-47](../EHR_extract/table.py#L37-L47), split across two blocks.

### In `base_table.add_columns`

Row-wise derivations on the table being built. `table` is injected.

| Function | `args` |
|---|---|
| `find_GA_days` | `GA_weeks_col`, `GA_days_col` — parses `"38w 2d"`-style strings to total days |
| `find_GA_weeks` | `GA_days_col`, `GA_weeks_col` — days ÷ 7 |
| `find_pregnancy_start` | `birth_date_col`, `GA_days_col`, `pregnancy_start_col` |
| `find_date_at_GA` | `birth_date_col`, `GA_days_col`, `GA_number`, `date_col` — the calendar date at a given GA |
| `find_GA_at_date` | `birth_date_col`, `GA_days_col`, `study_date_col`, `GA_at_date_col` — the GA on a given date |
| `find_maternal_age` | `m_table_path`, `maternal_birth_date_col`, `maternal_id_col`, `baby_birth_date_col`, `key_column`, `maternal_age_col`, optional `population_maternal_id_col` (default `m_cpr`) |

### In `custom_extract_criteria`

`main_table`, `min_date`, `max_date` and `allow_duplicates` are injected from `time_window`.

| Function | `args` |
|---|---|
| `extract_latest_value` | `left_on`, and either `table` or `sources`; `right_on`, `target_col`, `new_col_name`, `date_col`, `dtype` |
| `extract_filtered_values` | as above plus `filters` — a list of `{column, operator, value}` applied to the source before the latest value is taken |
| `extract_filtered_conditional_values` | as above plus `key_column` and `conditions` |

`sources` is a list of per-source overrides sharing the top-level fields, so one output column can be
filled from several tables:

```yaml
args:
  left_on: b_cpr
  right_on: BABY_CPR
  target_col: Para
  new_col_name: current_parity
  date_col: "Kontakt dato"
  sources:
    - table: ${paths.input_dir_SP}/Mor - Obstetrisk Historik DEL 1.csv
    - table: ${paths.input_dir_SP}/Mor - Obstetrisk Historik DEL 2.csv
      date_col: Dato        # overrides the shared value for this source only
```

Any `table` above may be a nested join spec instead of a path — see
[table.md](table.md#nested-joins).
