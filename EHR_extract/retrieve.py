import hydra
import json
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from EHR_extract.paths import get_config_path
from EHR_extract.utils import (
    get_python_operator,
    load_table,
    dtype_from_cfg,
    convert_to_date,
    convert_to_datetime,
    check_duplicates,
)
from omegaconf import DictConfig

load_dotenv()

def cast_types(table, dtype, column):
    if dtype == pl.Date:
        return table.with_columns(convert_to_date(column).alias(column))
    if dtype == pl.Datetime:
        return table.with_columns(convert_to_datetime(column).alias(column))
    return table.with_columns(pl.col(column).cast(dtype, strict=False))

def make_main_table(cfg, strict):
    all_discards = []
    with open(cfg.population, "r") as fp:
        population = json.load(fp)
    print("Population size:", len(population))

    # Get the barebones main table
    main_table = pl.DataFrame()
    for table in cfg.tables:
        table_df = load_table(table.table, strict=strict)
        table_df = table_df.rename(table.columns)[cfg.key_columns]
        table_df = table_df.filter(pl.col(cfg.population_column).is_in(population))
        main_table = main_table.vstack(table_df)

    # Check for duplicates    
    main_table = main_table.filter(pl.col(cfg.population_column) != "INVALID")
    main_table = check_duplicates(main_table, cfg.population_column, allow_duplicates=False)
    
    # Check population size
    if len(population) != len(main_table[cfg.population_column]):
        population_set = set(population)
        print(f"Population size mismatch. Population size: {len(population)}, Main table size: {len(main_table[cfg.population_column].unique())}")
        all_discards.append([
            cfg.population_column,
            list(population_set.difference(set(main_table[cfg.population_column].unique()))),
            len(population),
            len(main_table[cfg.population_column]),
        ])

    # Dropping nulls
    for key in cfg.key_columns:
        if key == cfg.population_column:
            continue
        dtype = dtype_from_cfg(cfg.dtypes[key])
        subset_table = cast_types(main_table, dtype, key)
        subset_table = subset_table.drop_nulls(key)
        population = set(main_table[cfg.population_column])
        subset_population = set(subset_table[cfg.population_column])
        all_discards.append([
            key, 
            list(population.difference(subset_population)),
            len(population),
            len(subset_population),
        ])
        main_table = subset_table
    
    return main_table, all_discards

def get_extract_criteria(cfg, main_table):
    for extract_criterion in cfg.extract_info:
        left_on = extract_criterion.key_column
        for source in extract_criterion.sources:
            print("Extract criterion:", extract_criterion.name)
            print("\tTable:", source.table)
            right_on = source.match_on
            table = load_table(source.table, strict=cfg.strict, null_values=["."])

            # Filter values
            py_operator = get_python_operator(source.operator)
            table = table.filter(py_operator(pl.col(source.column), source.value))

            # Merge
            tmp_table = main_table.join(
                table.select([right_on, source.column, source.date_col]),
                left_on=left_on,
                right_on=right_on,
                how="left",
            )

            # Add provenance: which source table contributed the value (only when it matches).
            source_name_col = "source_name"
            contributed = (
                pl.when(pl.col(source.column).is_not_null())
                .then(pl.lit(source.table))
                .otherwise(None)
            )
            if source_name_col in tmp_table.columns:
                tmp_table = tmp_table.with_columns(
                    pl.coalesce([pl.col(source_name_col), contributed]).alias(source_name_col)
                )
            else:
                tmp_table = tmp_table.with_columns(contributed.alias(source_name_col))

            if right_on != left_on and right_on in tmp_table.columns:
                tmp_table = tmp_table.drop(right_on)

            main_table = tmp_table
    return main_table


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    main_table, discards = make_main_table(
        cfg.base_table,
        strict=cfg.strict,
    )
    main_table = get_extract_criteria(cfg, main_table)


    d = {}
    for i in range(len(discards)):
        d[i] = {
            "key_column": discards[i][0],
            "n_discards": discards[i][2] - discards[i][3],
            "n_population_pre_discard": discards[i][2],
            "n_population_post_discard": discards[i][3],
            "discards": discards[i][1],
        }

    with open(cfg.paths.extract_save_path, "w") as fp:
        Path(cfg.paths.extract_save_path).parent.mkdir(parents=True, exist_ok=True)
        main_table.write_csv(fp)

if __name__ == "__main__":
    main()
