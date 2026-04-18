import hydra
import json
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import (
    find_GA_weeks,
    find_pregnancy_start,
)
from EHR_extract.paths import get_config_path
from EHR_extract.utils import (
    filter_numeric_rows,
    get_python_operator,
    load_table,
    update_population,
    dtype_from_cfg,
    convert_to_date,
    date_bound_expr,
)
from omegaconf import DictConfig, OmegaConf

load_dotenv()

custom_functions = {
    "find_GA_weeks": find_GA_weeks,
    "find_pregnancy_start": find_pregnancy_start,
}

BOOL_ALLOW_MANY_TO_ONE_BABY_ID = True


def check_duplicates(table, key_column, allow_duplicates=False):
    duplicates = table[key_column].value_counts().filter(pl.col("count") > 1)
    if duplicates.height > 0:
        if not allow_duplicates:
            raise ValueError(f"Duplicate entries for key column {key_column}. Examples: {duplicates.head(5)}")
        else:
            table = table.group_by(key_column).agg(pl.col("*").first())
            assert(len(table[key_column].unique()) == len(table[key_column]))
    return table

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
    main_table = check_duplicates(main_table, cfg.population_column, allow_duplicates=BOOL_ALLOW_MANY_TO_ONE_BABY_ID)
    
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
        subset_table = main_table.with_columns(
            pl.col(key).cast(dtype, strict=False)
        )
        subset_table = subset_table.drop_nulls(key)
        population = set(main_table[cfg.population_column])
        subset_population = set(subset_table[cfg.population_column])
        all_discards.append([
            key, 
            list(population.difference(subset_population)),
            len(population),
            len(subset_population),
        ])
    
    # Add the customs columns
    for column in cfg.add_columns:
        fn = custom_functions[column.function]
        args = column.args
        main_table = fn(**args, table=main_table)

    return main_table, all_discards

def get_extract_criteria(cfg, main_table):
    for extract_criterion in cfg.extract_criteria:
        extract_table = pl.DataFrame()
        left_on = extract_criterion.key_column
        dtype = dtype_from_cfg(extract_criterion.dtype)
        for source in extract_criterion.sources:
            print("Extract criterion:", extract_criterion.name)
            table = load_table(source.table, strict=cfg.strict)
            right_on = source.match_on

            tmp_table = main_table.join(
                table.select([right_on, source.column]),
                left_on=left_on,
                right_on=right_on,
                how="left",
            ).select([left_on, source.column]).rename({source.column: extract_criterion.name})
            tmp_table = tmp_table.with_columns(
                pl.col(extract_criterion.name).cast(dtype, strict=False)
            ).drop_nulls(extract_criterion.name)
            extract_table = extract_table.vstack(tmp_table)

        extract_table = check_duplicates(extract_table, extract_criterion.key_column, allow_duplicates=BOOL_ALLOW_MANY_TO_ONE_BABY_ID)
        main_table = main_table.join(extract_table, on=left_on, how="left")
    return main_table

def get_conditional_criteria(cfg, main_table):
    for conditional_criterion in cfg.conditional_criteria:
        left_on = conditional_criterion.match_on
        key_col = conditional_criterion.key_column
        condition_name = conditional_criterion.name
        time_window = conditional_criterion.time_window
        min_date = cfg.time_conditionals[time_window].min_date
        max_date = cfg.time_conditionals[time_window].max_date
        condition_matches = set()
        for condition in conditional_criterion.conditions:
            print("Extracting:", conditional_criterion.name)
            table = load_table(condition.table, strict=cfg.strict)
            right_on = condition.match_on

            tmp_table = main_table.join(
                table.select([right_on, condition.column, condition.date_col]),
                left_on=left_on,
                right_on=right_on,
                how="left",
            )

            # Filter on time 
            event_d = convert_to_date(condition.date_col)
            lo = date_bound_expr(**min_date)
            if lo is not None:
                tmp_table = tmp_table.filter(event_d >= lo)
            hi = date_bound_expr(**max_date)
            if hi is not None:
                tmp_table = tmp_table.filter(event_d <= hi)

            # Filter on operator
            py_operator = get_python_operator(condition.operator)
            tmp_table = tmp_table.filter(
                py_operator(pl.col(condition.column), condition.value)
            )

            if condition.condition is None:
                last_condition = set(tmp_table[key_col])
            elif condition.condition == "and":
                last_condition = last_condition.intersection(set(tmp_table[key_col]))
            elif condition.condition == "or":
                condition_matches = condition_matches.union(last_condition)
                last_condition = set(tmp_table[key_col])
            else:
                print("wow, weird condition")

        condition_matches = condition_matches.union(last_condition)
        main_table = main_table.with_columns(
            pl.col(key_col).is_in(list(condition_matches)).alias(condition_name)
        )
    return main_table

def table_from_cfg(cfg):
    main_table, discards = make_main_table(
        cfg.base_table,
        strict=cfg.strict,
    )
    main_table = get_extract_criteria(cfg, main_table)
    main_table = get_conditional_criteria(cfg, main_table)
    
    m_cpr_w_ptb = set(main_table.filter(pl.col("current_fibroids") == True)["m_cpr"])
    # m_cpr_multi_entries = set(
    #     main_table.group_by("m_cpr").len().filter(pl.col("len") > 2)["m_cpr"]
    # )
    m_cpr_subset = m_cpr_w_ptb #| m_cpr_multi_entries
    table_w_ptb = main_table.filter(pl.col("m_cpr").is_in(list(m_cpr_subset))).drop(
    )
    print(
        table_w_ptb.sort(
            by=[
                "m_cpr",
                pl.col("pregnancy_start").cast(pl.Date, strict=False),
            ],
            nulls_last=True,
        ).head(20)
    )
    # print(main_table.filter(pl.col("m_cpr").is_in(m_cpr_w_ptb)).sort(["m_cpr", "pregnancy_start"]))
    return main_table, discards


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    table, discards = table_from_cfg(cfg)

    d = {}
    for i in range(len(discards)):
        d[i] = {
            "key_column": discards[i][0],
            "n_discards": discards[i][2] - discards[i][3],
            "n_population_pre_discard": discards[i][2],
            "n_population_post_discard": discards[i][3],
            "discards": discards[i][1],
        }

    with open(cfg.paths.table_save_path, "w") as fp:
        table.write_csv(fp)
    with open(cfg.paths.discards_save_path, "w") as fp:
        json.dump(d, fp, indent=4)

if __name__ == "__main__":
    main()
