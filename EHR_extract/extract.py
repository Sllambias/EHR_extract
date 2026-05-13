import hydra
import json
import logging
import os
import polars as pl
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import (
    find_close_births,
    find_images_with_predicted_classes,
    find_images_within_time_windows,
    find_multiple_births,
    match_images_with_child,
    merge_population_on,
)
from EHR_extract.paths import get_config_path
from EHR_extract.utils import (
    filter_numeric_rows,
    format_criterion,
    get_python_operator,
    load_table,
    merge_composed_population_tables,
    merge_population_tables,
    update_population,
    RecursiveSearchpathPlugin,
    deduplicate_barn_cpr,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from hydra.core.plugins import Plugins

load_dotenv()
Plugins.instance().register(RecursiveSearchpathPlugin)


def match_value_with_child_cpr_on_birth_id_if_ga_below_threshold(
    operator,
    value,
    value_table_path,
    value_column,
    value_table_birth_id_column,
    mapping_table_path,
    mapping_table_birth_id_column,
    mapping_table_child_cpr_column,
    population,
    population_key_column,
    population_gestational_age_column,
    ga_threshold,
):
    value_table = load_table(value_table_path)

    py_operator = get_python_operator(operator)
    if isinstance(value, ListConfig):
        value_table = value_table.filter(pl.any_horizontal([py_operator(pl.col(value_column), val) for val in value]))
    else:
        value_table = value_table.filter(py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path)
    mapping_table = mapping_table.filter(pl.col(mapping_table_child_cpr_column).is_in(set(population[population_key_column])))

    joined = value_table.join(
        mapping_table,
        left_on=value_table_birth_id_column,
        right_on=mapping_table_birth_id_column,
        how="inner",
    )
    joined = joined.join(population, left_on=mapping_table_child_cpr_column, right_on=population_key_column, how="inner")
    joined = joined.filter(pl.col(population_gestational_age_column).cast(pl.Int64) < ga_threshold)

    matches = set(joined[mapping_table_child_cpr_column].unique())
    return matches


def match_value_with_child_cpr_on_birthdate_if_ga_below_threshold(
    operator,
    value,
    value_table_path,
    value_column,
    value_time_column,
    value_mother_cpr_column,
    population,
    population_mother_cpr_column,
    population_child_cpr_column,
    population_birth_column,
    population_gestational_age_column,
    population_key_column,
    ga_threshold,
):
    value_table = load_table(value_table_path)

    # Filter based on operator and value
    py_operator = get_python_operator(operator)
    if isinstance(value, ListConfig):
        value_table = value_table.filter(pl.any_horizontal([py_operator(pl.col(value_column), val) for val in value]))
    else:
        value_table = value_table.filter(py_operator(pl.col(value_column), value))

    # Join on mother's CPR
    joined = value_table.join(population, left_on=value_mother_cpr_column, right_on=population_mother_cpr_column, how="inner")

    # Calculate conception date: birthdate - gestational_age weeks
    joined = filter_numeric_rows(joined, population_gestational_age_column)
    joined = joined.with_columns(
        conception_date=pl.col(population_birth_column).str.to_datetime()
        - pl.duration(days=pl.col(population_gestational_age_column).cast(pl.Int64))
    )

    # Check if procedure_time is within conception_date +/- time_window_days
    joined = joined.filter(
        (pl.col(value_time_column).str.to_datetime() >= pl.col("conception_date"))
        & (pl.col(value_time_column).str.to_datetime() <= pl.col(population_birth_column).str.to_datetime())
    )
    joined = joined.filter(pl.col(population_gestational_age_column).cast(pl.Int64) < ga_threshold)

    # Get the unique child CPRs
    matches = set(joined[population_child_cpr_column].unique())
    return matches


custom_functions = {
    "find_close_births": find_close_births,
    "find_images_within_time_windows": find_images_within_time_windows,
    "find_images_with_predicted_classes": find_images_with_predicted_classes,
    "find_multiple_births": find_multiple_births,
    "match_images_with_child": match_images_with_child,
    "match_value_with_child_cpr_on_birthdate_if_ga_below_threshold": match_value_with_child_cpr_on_birthdate_if_ga_below_threshold,
    "match_value_with_child_cpr_on_birth_id_if_ga_below_threshold": match_value_with_child_cpr_on_birth_id_if_ga_below_threshold,
    "merge_population_on": merge_population_on,
}


def extract_from_cfg(cfg, population):
    all_discards = []

    logging.info(
        f"Population size: {len(population)} with unique IDs: {population[cfg.population.population_key].n_unique()}",
    )
    for criterion in cfg.get("conditional_criteria", {}):
        criterion_population = set()
        for condition in criterion.conditions:
            if condition.table == "population":
                table = population.clone()
            else:
                table = load_table(condition.table, strict=cfg.strict)
            logging.debug(
                f"Table rows / unique IDs total: {len(table)} / {table[condition.match_on].n_unique()} \
                    for table: {condition.table}"
            )

            table = table.filter(pl.col(condition.match_on).is_in(population[cfg.population.population_key]))
            logging.debug(
                f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
                after filtering on {condition.match_on}"
            )

            if condition.get("operator", None) is None:
                last_condition_population = set(table[condition.match_on])
                continue

            py_operator = get_python_operator(condition.operator)
            if condition.operator in [">", "<", ">=", "<="]:
                table = filter_numeric_rows(table, condition.column)
            table = table.filter(py_operator(pl.col(condition.column), condition.value))
            logging.debug(
                f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
                    after filtering on {condition.column} {condition.operator} {condition.value}"
            )

            if condition.condition is None:
                last_condition_population = set(table[condition.match_on])
            elif condition.condition == "and":
                last_condition_population = last_condition_population.intersection(set(table[condition.match_on]))
            elif condition.condition == "or":
                criterion_population = last_condition_population
                last_condition_population = set(table[condition.match_on])
            else:
                logging.warn("wow, weird condition")

        criterion_population = criterion_population.union(last_condition_population)
        population, discards, n_discards, n_population_before_discard = update_population(
            population=population,
            key=cfg.population.population_key,
            subset=set(criterion_population),
            action=criterion.action,
        )
        logging.info(f"Population size: {len(population)} after filtering on criteria {format_criterion(criterion)} \n")
        all_discards.append([OmegaConf.to_container(criterion), list(discards), n_discards, n_population_before_discard])

    logging.info("\n ### Applying custom criteria ### \n")
    for custom_cfg in cfg.get("custom_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        set_of_matches = fn(**args, population=population, population_key_column=cfg.population.population_key)
        population, discards, n_discards, n_population_before_discard = update_population(
            population=population,
            key=cfg.population.population_key,
            subset=set_of_matches,
            action=custom_cfg.action,
        )
        all_discards.append([OmegaConf.to_container(custom_cfg), list(discards), n_discards, n_population_before_discard])

        logging.info(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function} \n")

    logging.info("\n ### Applying imaging matching criteria ### \n")
    if "imaging_table" in cfg.keys():
        population = match_images_with_child(
            table_cfg=cfg.imaging_table,
            population=population,
        )
        logging.info(
            f"After matching images with patients: \n"
            f"Valid image+patient matches: {len(population)} with "
            f"unique {cfg.population.population_key}: {population[cfg.population.population_key].n_unique()} "
            f"and unique FILE_PATH: {population['FILE_PATH'].n_unique()}"
        )

    for custom_cfg in cfg.get("imaging_matching_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        population, discard_stats = fn(
            **args,
            population=population,
            population_key_column=cfg.population.population_key,
        )

        all_discards.append(
            [
                discard_stats["criteria"],
                discard_stats["discards"],
                discard_stats["n_discards"],
                discard_stats["n_population_before_discard"],
            ]
        )
        logging.info(
            f"After filtering on custom criteria: {custom_cfg.function} \n"
            f"Valid image+patient matches: {len(population)} with "
            f"unique {cfg.population.population_key}: {population[cfg.population.population_key].n_unique()} "
            f"and unique FILE_PATH: {population['FILE_PATH'].n_unique()}"
        )
    return population, all_discards


def make_train_test_split(holdout_csv_path, population, split_key):
    holdout = load_table(holdout_csv_path)
    holdout = holdout.get_column(split_key).to_list()
    train = population.filter(~pl.col(split_key).is_in(holdout))
    test = population.filter(pl.col(split_key).is_in(holdout))
    return train, test


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population = pl.DataFrame()

    if cfg.population.get("tables", None) is not None:
        population = merge_population_tables(cfg.population.tables, population)
    if cfg.population.get("composed_tables", None) is not None:
        population = merge_composed_population_tables(
            population, cfg.population.population_key, cfg.population.composed_tables
        )
    population = deduplicate_barn_cpr(population, population_key=cfg.population.population_key)
    population, discards = extract_from_cfg(cfg, population=population)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    d = {}
    for i in range(len(discards)):
        d[i] = {
            "criteria": discards[i][0],
            "n_discards": discards[i][2],
            "n_population_pre_discard": discards[i][3],
            "n_population_post_discard": discards[i][3] - discards[i][2],
            "discards": discards[i][1],
        }

    with open(cfg.paths.discards_save_path + ".json", "w") as fp:
        json.dump(d, fp, indent=4)

    population.write_csv(cfg.paths.population_save_path + "_train_and_test.csv")

    if cfg.paths.holdout_csv is not None:
        train_pop, test_pop = make_train_test_split(cfg.paths.holdout_csv, population, cfg.population.split_key)

        intersection = set(train_pop[cfg.population.population_key]).intersection(set(test_pop[cfg.population.population_key]))
        if len(intersection) > 0:
            logging.warning(
                f"leak detected in train and test splits. Removing leaked samples from TEST but this should be investigated. Leaked IDs: {intersection}"
            )
            test_pop = test_pop.filter(~pl.col(cfg.population.population_key).is_in(intersection))
        train_pop.write_csv(cfg.paths.population_save_path + "_train.csv")
        test_pop.write_csv(cfg.paths.population_save_path + "_test.csv")


if __name__ == "__main__":
    main()
