import hydra
import json
import polars as pl
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import (
    extract_filtered_conditional_values,
    extract_filtered_values,
    extract_latest_value,
    find_date_at_GA,
    find_GA_at_date,
    find_GA_days,
    find_GA_weeks,
    find_maternal_age,
    find_pregnancy_start,
)
from EHR_extract.paths import get_config_path
from EHR_extract.summary import get_summary
from EHR_extract.utils import (
    RecursiveSearchpathPlugin,
    check_duplicates,
    convert_to_date,
    convert_to_datetime,
    date_bound_expr,
    dtype_from_cfg,
    get_python_operator,
    load_table,
    safe_save_df,
    take_latest_row,
)
from hydra.core.plugins import Plugins
from omegaconf import DictConfig
from pathlib import Path

load_dotenv()
Plugins.instance().register(RecursiveSearchpathPlugin)

custom_functions = {
    "find_pregnancy_start": find_pregnancy_start,
    "find_GA_days": find_GA_days,
    "find_GA_weeks": find_GA_weeks,
    "find_date_at_GA": find_date_at_GA,
    "find_GA_at_date": find_GA_at_date,
    "find_maternal_age": find_maternal_age,
    "extract_filtered_values": extract_filtered_values,
    "extract_latest_value": extract_latest_value,
    "extract_filtered_conditional_values": extract_filtered_conditional_values,
}


def cast_types(table, dtype, column):
    if dtype == pl.Date:
        return table.with_columns(convert_to_date(column))
    if dtype == pl.Datetime:
        return table.with_columns(convert_to_datetime(column))
    return table.with_columns(pl.col(column).cast(dtype, strict=False))


def make_main_table(cfg, strict, allow_duplicates=False):
    all_discards = []
    population = pl.read_csv(cfg.population)[cfg.population_column].unique().to_list()
    print("Population size:", len(population))

    # Get the barebones main table
    main_table = pl.DataFrame()
    for table in cfg.get("tables", []):
        table_df = load_table(table.table, strict=strict)
        table_df = table_df.rename(table.columns)[cfg.key_columns]
        table_df = table_df.filter(pl.col(cfg.population_column).is_in(population))
        main_table = main_table.vstack(table_df)
    print("Main table size:", len(main_table))

    if not allow_duplicates:
        main_table = check_duplicates(main_table, cfg.population_column, allow_duplicates=allow_duplicates)

    # Dropping nulls
    for key in cfg.key_columns:
        if key == cfg.population_column:
            continue
        population_before = set(main_table[cfg.population_column])
        dtype = dtype_from_cfg(cfg.dtypes[key])
        main_table = cast_types(main_table, dtype, key)
        main_table = main_table.drop_nulls(key)
        population_after = set(main_table[cfg.population_column])
        all_discards.append(
            [
                key,
                list(population_before.difference(population_after)),
                len(population_before),
                len(population_after),
            ]
        )

    # Add the customs columns
    for column in cfg.add_columns:
        population_before = set(main_table[cfg.population_column])
        fn = custom_functions[column.function]
        args = column.args
        dtype = dtype_from_cfg(column.dtype)
        main_table = fn(**args, table=main_table)
        main_table = cast_types(main_table, dtype, column.column)
        main_table = main_table.drop_nulls(column.column)
        population_after = set(main_table[cfg.population_column])
        all_discards.append(
            [
                column.column,
                list(population_before.difference(population_after)),
                len(population_before),
                len(population_after),
            ]
        )
        if not allow_duplicates:
            main_table = check_duplicates(main_table, cfg.population_column)
    return main_table, all_discards


def get_extract_criteria(cfg, main_table):
    for extract_criterion in cfg.extract_criteria:
        extract_table = pl.DataFrame()
        left_on = extract_criterion.key_column
        dtype = dtype_from_cfg(extract_criterion.dtype)
        for source in extract_criterion.sources:
            print("Extract criterion:", extract_criterion.name)
            print("\tTable:", source.table)
            table = load_table(source.table, strict=cfg.strict)
            right_on = source.match_on

            tmp_table = (
                main_table.join(
                    table.select([right_on, source.column, source.date_col]),
                    left_on=left_on,
                    right_on=right_on,
                    how="left",
                )
                .select([left_on, source.column, source.date_col])
                .rename({source.column: extract_criterion.name})
            )
            tmp_table = tmp_table.with_columns(pl.col(extract_criterion.name).cast(dtype, strict=False)).drop_nulls(
                extract_criterion.name
            )
            tmp_table = take_latest_row(tmp_table, left_on, source.date_col)
            tmp_table = tmp_table.select([left_on, extract_criterion.name])
            extract_table = extract_table.vstack(tmp_table)

        main_table = main_table.join(extract_table, on=left_on, how="left")
    return main_table


def get_custom_extract_criteria(cfg, main_table):
    for custom_extract_criterion in cfg.custom_extract_criteria:
        print("Custom extract criterion:", custom_extract_criterion.name)
        fn = custom_functions[custom_extract_criterion.function]
        time_window = custom_extract_criterion.time_window
        min_date = cfg.time_conditionals[time_window].min_date
        max_date = cfg.time_conditionals[time_window].max_date
        main_table = fn(
            **custom_extract_criterion.args,
            main_table=main_table,
            min_date=min_date,
            max_date=max_date,
            allow_duplicates=cfg.allow_duplicates,
        )
        if not cfg.allow_duplicates:
            main_table = check_duplicates(main_table, custom_extract_criterion.args.left_on)

    return main_table


def get_conditional_bool_criteria(cfg, main_table):
    for conditional_criterion in cfg.conditional_bool_criteria:
        left_on = conditional_criterion.match_on
        key_col = conditional_criterion.key_column
        condition_name = conditional_criterion.name
        time_window = conditional_criterion.time_window
        min_date = cfg.time_conditionals[time_window].min_date
        max_date = cfg.time_conditionals[time_window].max_date
        condition_matches = set()
        for condition in conditional_criterion.conditions:
            print("Extracting:", conditional_criterion.name)
            print("\tTable:", condition.table)
            table = load_table(condition.table, strict=cfg.strict)
            right_on = condition.match_on

            # Filter on operator
            py_operator = get_python_operator(condition.operator)
            table = table.filter(py_operator(pl.col(condition.column), condition.value))
            # Merge
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
            print("\tNumber of matches", len(tmp_table))

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
        main_table = main_table.with_columns(pl.col(key_col).is_in(list(condition_matches)).alias(condition_name))
    return main_table


def table_from_cfg(cfg):
    main_table, discards = make_main_table(
        cfg.base_table,
        strict=cfg.strict,
        allow_duplicates=cfg.allow_duplicates,
    )
    main_table = get_extract_criteria(cfg, main_table)
    main_table = get_conditional_bool_criteria(cfg, main_table)
    main_table = get_custom_extract_criteria(cfg, main_table)

    summary_cfg = cfg.get("summary_table")
    if summary_cfg is not None and summary_cfg.get("make_table", False):
        summary_table = get_summary(
            main_table,
            list(summary_cfg.ignore_columns),
            (summary_cfg.get("n_samples", None)),
        )
        print(summary_table)

        ptb_table = main_table.filter(pl.col("GA").cast(pl.Int64, strict=False) < 259)
        sum_ptb = get_summary(ptb_table, list(summary_cfg.ignore_columns), (summary_cfg.get("n_samples", None)))
        # print("GA < 259 ")
        # print(sum_ptb)

        non_ptb_table = main_table.filter(pl.col("GA").cast(pl.Int64, strict=False) >= 259)
        sum_non_ptb = get_summary(non_ptb_table, list(summary_cfg.ignore_columns), (summary_cfg.get("n_samples", None)))
        # print("GA > 259 ")
        # print(sum_non_ptb)

        extra_tables = {
            "ptb": sum_ptb,
            "non_ptb": sum_non_ptb,
        }

    else:
        summary_table = None
        extra_tables = None
    return main_table, summary_table, discards, extra_tables


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    table, sum_table, discards, extra_tables = table_from_cfg(cfg)

    d = {}
    for i in range(len(discards)):
        d[i] = {
            "key_column": discards[i][0],
            "n_discards": discards[i][2] - discards[i][3],
            "n_population_pre_discard": discards[i][2],
            "n_population_post_discard": discards[i][3],
            "discards": discards[i][1],
        }

    if extra_tables is not None:
        for key, value in extra_tables.items():
            Path(cfg.paths.summary_save_path.replace("summary", f"{key}_summary")).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.paths.summary_save_path.replace("summary", f"{key}_summary"), "w") as fp:
                safe_save_df(value).write_csv(fp)

    Path(cfg.paths.table_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.table_save_path, "w") as fp:
        table.write_csv(fp)
    Path(cfg.paths.discards_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.discards_save_path, "w") as fp:
        json.dump(d, fp, indent=4)
    if sum_table is not None:
        Path(cfg.paths.summary_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.paths.summary_save_path, "w") as fp:
            safe_save_df(sum_table).write_csv(fp)


if __name__ == "__main__":
    main()
