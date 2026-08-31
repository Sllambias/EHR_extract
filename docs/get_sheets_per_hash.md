## Using the SP-Query-Tool https://lab.compute.dtu.dk/sonai/sp-query-tool/


## (optional) Generate list of hahes using EHR_extract
```bash
python EHR_extract/extract.py --config-name astridv1
```

## Quick start

```bash
python src/sp-query-tool \ 
  --hashes /path/to/hashlist.csv \
  --config /path/to/config.yaml
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

