# EHR_extract

A Hydra-configured pipeline for extracting cohorts from Danish perinatal electronic health records
(SDS registers, Sundhedsplatformen tables, and a fetal ultrasound image database). Everything is
driven by YAML: you write a config, run a script, and get CSVs plus a JSON record of what each
criterion discarded.

## Install

```bash
pip install -e .
pip install scikit-learn   # split.py only — not declared in pyproject.toml
pip install pandas         # get_imgs_from_db.py only — not declared in pyproject.toml
```

`split.py` imports sklearn unconditionally (via `utils/custom_split_functions.py`), so it will not
start without it.

Then create a `.env` in the repo root — it is gitignored, so a fresh clone has none and every script
fails until you do:

```bash
cp .env.example .env   # then edit the paths
```

| Variable | Used for |
|---|---|
| `EHR_EXTRACT_CONFIGS` | Hydra's config root. Required by every script. |
| `EHR_EXTRACT_OUTPUTS` | Output root, as `${oc.env:EHR_EXTRACT_OUTPUTS}` in most configs. |
| `EHR_EXTRACT_DATA` | Data root, as `${oc.env:EHR_EXTRACT_DATA}` in `configs/tables_merged/*` and `configs/templates/*`. |

## Running

```bash
python EHR_extract/<script>.py --config-name <config>
```

`--config-name` is the **bare filename** — no directory, no `.yaml`. A search-path plugin
([utils.py:298](EHR_extract/utils/utils.py#L298)) walks every subdirectory of `$EHR_EXTRACT_CONFIGS`,
so `--config-name test_preterm` finds `configs/testing/test_preterm.yaml`. Config names must
therefore be unique across the whole tree.

Any field can be overridden on the command line: `--config-name X paths.output_dir=/tmp/y strict=False`.
`--cfg job --resolve` prints the fully merged config and exits without reading any data — the fastest
way to check a config.

Every config must define `paths.output_dir`: `configs/default.yaml` points Hydra's run directory at
it, and each run writes `<output_dir>/log/{config,hydra,overrides}.yaml` alongside its results.

## Scripts

| Script | Does | Docs |
|---|---|---|
| [`split.py`](EHR_extract/split.py) | Train/test or k-fold splits over subject IDs | [docs/split.md](docs/split.md) |
| [`extract.py`](EHR_extract/extract.py) | Build a population, apply inclusion/exclusion criteria, optionally match images | [docs/extract.md](docs/extract.md) |
| [`table.py`](EHR_extract/table.py) | Build a wide feature table from a population | [docs/table.md](docs/table.md) |
| [`filter_tables_on_hashes.py`](EHR_extract/filter_tables_on_hashes.py) | Subset raw tables to a population's IDs | [docs/filter_tables.md](docs/filter_tables.md) |
| [`split_tables_on_hashes.py`](EHR_extract/split_tables_on_hashes.py) | One `.xlsx` per patient for chart review | [docs/filter_tables.md](docs/filter_tables.md) |
| [`get_imgs_from_db.py`](EHR_extract/get_imgs_from_db.py) | Snapshot the ultrasound SQLite DB to `all_images_<date>.csv` | below |
| [`summary.py`](EHR_extract/summary.py) | Per-column distributions | see Known limitations |

The usual order:

```
split.py ──> test_split_*.csv ──┐
                                 ├──> extract.py ──> *_population_{train,test}.csv ──┬──> table.py
get_imgs_from_db.py ──> images ──┘                                                    ├──> filter_tables_on_hashes.py
                                                                                      └──> split_tables_on_hashes.py
```

Operators and the custom-function registry are listed in [docs/reference.md](docs/reference.md).

## Templates

[`configs/templates/`](configs/templates/) holds a minimal, runnable-shaped config per entry point.
Copy one and edit it rather than starting from the 1000+ line production configs.

| Template | For | Shows |
|---|---|---|
| [`template_population.yaml`](configs/templates/template_population.yaml) | `extract.py` | stacking source tables, one criterion |
| [`template_population_images.yaml`](configs/templates/template_population_images.yaml) | `extract.py` | inheriting a config, matching ultrasound images |
| [`template_split.yaml`](configs/templates/template_split.yaml) | `split.py` | a fresh random holdout |
| [`template_table.yaml`](configs/templates/template_table.yaml) | `table.py` | nested joins, time windows, one of each criteria block |
| [`template_filter_tables.yaml`](configs/templates/template_filter_tables.yaml) | `filter_tables_on_hashes.py` | filtering tables to a cohort |
| [`template_split_tables.yaml`](configs/templates/template_split_tables.yaml) | `split_tables_on_hashes.py` | per-patient Excel export |

Production configs live in `configs/populations/` (cohorts), `configs/splits/`,
`configs/tables_merged/` (feature tables), and `configs/testing/` (thin overlays that repoint the
input paths at local fixtures).

## Generating the imaging database

`python EHR_extract/get_imgs_from_db.py` queries the ultrasound SQLite database and writes
`all_images_<YYYY-MM-DD>.csv` into the current directory. The database path is hardcoded at
[get_imgs_from_db.py:5](EHR_extract/get_imgs_from_db.py#L5). Columns:

`file_path`, `no_ocr_preprocessed_file_path`, `phair_hash` (hashed maternal CPR), `study_date`,
`physical_delta_x/y`, `region_location_{min_x0,min_y0,max_x1,max_y1}`.

`phair_hash` is the join key to the mother — see [docs/extract.md](docs/extract.md#matching-images).

## Known limitations

Found while writing these docs. Four more have since been fixed; these remain.

1. **`summary.py`'s CLI is unusable as shipped.** `summary_from_cfg` requires a `base_population`
   block ([summary.py:36](EHR_extract/summary.py#L36)) that no config in the repo defines. Its
   working role is supplying `get_summary()` to `table.py`.
2. **`filter_tables_on_hashes.py` ignores `time_col` and `columns`** and writes every column of each
   table ([filter_tables_on_hashes.py:24-29](EHR_extract/filter_tables_on_hashes.py#L24-L29)).
3. **`custom_split_fn: kfold` writes its fold CSVs, then crashes.** `kfold` has no `return`, so
   unpacking its result at [split.py:100](EHR_extract/split.py#L100) raises on `None`. The files are
   already on disk by then.
4. **`configs/testing/test_SL_ehr_table.yaml` inherits `SL_EHR_noMP_img_V5@`, which does not exist.**
5. **`configs/tables_merged/EHR_SP.yaml` is not valid YAML.** The `table:` block under the second
   `conditional_bool_criteria` condition is indented one level too deep, so the file fails to parse
   (`line 1233, column 16`). The other eight `configs/tables_merged/*` configs compose cleanly.
6. Many configs hardcode machine-specific absolute paths (`/Users/zcr545/...`,
   `/projects/users/data/UCPH/...`, `/storage/archive/...`).

## License

MIT — see [LICENSE](LICENSE).
