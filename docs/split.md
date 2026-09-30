# split.py — train/test splits

```bash
python EHR_extract/split.py --config-name template_split
```

Splits subject IDs into train and test. Splits are made once and reused: `extract.py` takes the
resulting test CSV as its `paths.holdout_csv`, so every cohort derived later shares one holdout.

Template: [`configs/templates/template_split.yaml`](../configs/templates/template_split.yaml).

Note this script defaults to `--config-name split_V3`, unlike the others which default to `default`.

Split on the ID that must not leak — `CPR_MOR` in every shipped config, so that siblings and a
mother's repeat pregnancies never straddle the boundary.

## Outputs

`<output_dir>/train_split_<holdout_frac>_<YYYY-MM-DD>.csv` and `test_split_...`, each a single
column named after `population.population_key`.

Existing files are **never overwritten**: the script logs a warning and writes nothing
([split.py:115-122](../EHR_extract/split.py#L115-L122)). Since the name contains today's date, a
second run on the same day is a no-op — rename or delete the old files, or bump the config name.

## Mode 1 — fresh split

The population is the union of `population.tables` after each is reduced to `population_key`.
`holdout_frac` of the IDs go to test, chosen with `seed`.

```yaml
population:
  population_key: CPR_MOR
  tables:
    - table: ${paths.input_dir_SDS}/mfr.csv
      columns: {CPR_MOR: CPR_MODER}
    - table: ${paths.input_dir_SP}/Population.csv
      columns: {CPR_MOR: MOR_CPR}

holdout_frac: 0.05
seed: 42
```

## Mode 2 — update an existing split

For when subjects have been added since the split was made. IDs already in either old split keep
their side; only previously unseen IDs are split, then appended. Results go to **new files**, so the
originals are untouched.

Add to a mode 1 config:

```yaml
update_train_split: ${paths.split_dir}/split_V3/train_split_0.05_2026-04-27.csv
update_test_split: ${paths.split_dir}/split_V3/test_split_0.05_2026-04-27.csv
```

Only correct when the split parameters are otherwise identical. It logs `NO NEW SAMPLES IN UPDATE`
if nothing changed. See [`configs/splits/split_V3-1.yaml`](../configs/splits/split_V3-1.yaml).

## Mode 3 — subtract IDs from an existing split

Removes a list of IDs from both sides, e.g. to drop a legacy cohort. `population.tables` is not read
in this mode — only `population.population_key`. Either negative list may be `null`.

```yaml
update_splits_negative:
  negative_population_key: cpr_hashed        # the ID column in the negative CSVs
  train_split: ${paths.split_dir}/split_V4/train_split_0.15_2026-06-08.csv
  test_split: ${paths.split_dir}/split_V4/test_split_0.15_2026-06-08.csv
  update_train_split_negative: /path/to/ids_to_remove.csv
  update_test_split_negative: null

population:
  population_key: CPR_MOR
```

See [`configs/splits/split_V4_with_legacy_splits.yaml`](../configs/splits/split_V4_with_legacy_splits.yaml).

## Mode 4 — custom split functions

```yaml
custom_split_fn: kfold
population:
  folds: 6
  output_dir: ${paths.output_dir}
  population_key: CPR_MOR
  tables: [...]
```

`kfold` writes `train_split_fold<i>_<date>.csv` / `test_split_fold<i>_<date>.csv` itself, then the
run crashes — see the README's Known limitations. The fold files are complete before it does.

`preterm_custom1` builds a criteria-driven split where test is the intersection of an imaging and an
EHR criterion, rather than a random draw. Its config nests `conditional_criteria`, `imaging_table`
and `imaging_matching_criteria` **under `population:`** — see
[`configs/splits/split_preterm_custom1.yaml`](../configs/splits/split_preterm_custom1.yaml).

Both are registered at [split.py:18](../EHR_extract/split.py#L18).
