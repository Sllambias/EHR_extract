## (optional) Generate list of hahes using EHR_extract
```bash
python EHR_extract/extract.py --config-name test_example_pop         
```

## Quick start

```bash
python EHR_extract/filter_tables_on_hashes.py --config-name test_filter_table
```

## Config format

```yaml
output_folder: /path/to/output (defaults to ./output)

tables:
  - table: Mor - CPMI - Notater
    id_col: MOR_CPR
    time_col: dato
    columns:
      - notater
      - dato
```

