# EHR_extract

Builds cohorts and tables from SDS registry and Sundhedsplatformen (SP) exports, driven by Hydra configs.
How the scripts and configs work: [docs/](docs/).

## Testing

Always test on the synthetic data from the ehr2meds repo (`../ehr2meds`), generated with its
`configs/synthetic_generation/fetal_SDS_SP_from_pop_part{1,2,3}.yaml` configs. Do not edit the generated CSVs
by hand or make ad-hoc fixtures.

The data is in `../ehr2meds/data/raw/fetal_data/SDS_AND_SP_from_population`. The configs in `configs/testing/`
read it through `paths.input_dir_SDS` and `paths.input_dir_SP`. Run in the `EHR_extract` conda env from this
repo's root:

```bash
python EHR_extract/extract.py --config-name test_add_column paths.output_dir=<output dir>
```

Point any config at another data directory with `paths.input_dir_SDS=<dir> paths.input_dir_SP=<dir>`.

### Generating the data

The generator is installed into the `EHR_extract` env by the `test` dependency group: `pip install --group test`
(pip >= 25.1). Run from this repo's root. Run the parts in order, because parts 2 and 3 link to tables written
by part 1:

```bash
export EHR2MEDS_CONFIGS=$(cd ../ehr2meds && pwd)/configs EHR2MEDS_DATA=<dir>
for part in 1 2 3; do
  python -m ehr2meds.generate_synthetic_raw_data --config-name synthetic_generation/fetal_SDS_SP_from_pop_part$part
done
```

The tables are written to `<dir>/raw/fetal_data/SDS_AND_SP_from_population`, which takes about 5 minutes.
`EHR2MEDS_CONFIGS` must be an absolute path. The generator reads its configs from that path, so config changes
on an ehr2meds branch apply without reinstalling. Only changes to the generator's code need
`pip install -e ../ehr2meds`.

The generator seeds with 0, so ehr2meds main reproduces the data above byte for byte.

### When a needed column or table is missing

1. In ehr2meds, create a new branch from main, e.g. `synth/<what is being tested>`.
2. Add the column to the table's entry in the part config that defines it
   (`grep -n "<table name>:" configs/synthetic_generation/fetal_SDS_SP_from_pop_part*.yaml`), or add a new table.
   Infer reasonable values from what the column means:
   - use the column name exactly as the EHR_extract configs spell it;
   - pick a realistic type and range, e.g. `rand_int` 0-10 for an Apgar score;
   - add an `insert_missing` corruption at a rate like the table's other columns.

   A new table gets its IDs through `linked_columns` (e.g. `BABY_CPR` from `Population`), like the tables around
   it. `type: choice` draws IDs with replacement, so one ID can get several rows with different values.
   `choice_unique` gives at most one row per ID. The generator types are in
   `ehr2meds/synthetic_data_generation/generate.py`, and the corruptions are in `corrupt.py`.
3. Generate all three parts into a separate directory, not the default output. A new column shifts the seeded
   random draws, so values in other tables change too, including tables in later parts that link to the
   changed one. Run EHR_extract with `paths.input_dir_SDS` and `paths.input_dir_SP` set to that directory. When
   comparing two EHR_extract versions, run both on the same generated data.
4. Do not commit the ehr2meds changes. Leave them uncommitted on the branch and tell the user the branch name
   and what was added.
