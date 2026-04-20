import hydra
import json
import logging
import polars as pl
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import (
    find_close_births,
    find_images_and_timedeltas,
    find_images_with_predicted_classes,
    find_multiple_pregnancies,
    find_scantime_ga,
    merge_population_on,
)
from EHR_extract.paths import get_config_path
from EHR_extract.utils import (
    filter_numeric_rows,
    get_python_operator,
    load_table,
    merge_population_tables,
    update_population,
    write_imaging_metadata_to_formats,
)
from omegaconf import DictConfig, OmegaConf

load_dotenv()

custom_functions = {
    "find_close_births": find_close_births,
    "find_images_and_timedeltas": find_images_and_timedeltas,
    "find_scantime_ga": find_scantime_ga,
    "find_multiple_pregnancies": find_multiple_pregnancies,
    "find_images_with_predicted_classes": find_images_with_predicted_classes,
    "merge_population_on": merge_population_on,
}


def extract_from_cfg(cfg, population):
    all_discards = []

    logging.info(
        f"Population size: {len(population)} with unique IDs: {population[cfg.population.population_key].n_unique()}",
    )
    for criterion in cfg.conditional_criteria:
        criterion_population = set()
        for condition in criterion.conditions:
            table = load_table(condition.table, strict=cfg.strict)
            logging.debug(
                f"Table rows / unique IDs total: {len(table)} / {table[condition.match_on].n_unique()} for table: {condition.table}"
            )

            table = table.filter(pl.col(condition.match_on).is_in(population[cfg.population.population_key]))
            logging.debug(
                f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} after filtering on {condition.match_on}"
            )

            if condition.get("operator", None) is None:
                last_condition_population = set(table[condition.match_on])
                continue

            py_operator = get_python_operator(condition.operator)
            if condition.operator in [">", "<", ">=", "<="]:
                table = filter_numeric_rows(table, condition.column)
            table = table.filter(py_operator(pl.col(condition.column), condition.value))
            logging.debug(
                f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} after filtering on {condition.column} {condition.operator} {condition.value}"
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
        logging.info(f"Population size: {len(population)} after filtering on criteria {criterion} \n")
        all_discards.append([OmegaConf.to_container(criterion), list(discards), n_discards, n_population_before_discard])

    logging.info("\n ### Applying custom criteria ### \n")
    for custom_cfg in cfg.get("custom_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        set_of_matches = fn(**args, population=set(population.get_column(cfg.population.population_key)))
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
        population = merge_population_on(
            table=cfg.imaging_table.table,
            merge_key=cfg.imaging_table.population_key,
            population=population,
            population_key_column=cfg.population.population_key,
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

        logging.info(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function} \n")
    return population, all_discards


def make_train_test_split(holdout_csv_path, population, file_path_key, prefix):
    holdout = load_table(holdout_csv_path, has_header=False)
    holdout = holdout.with_columns(pl.col("column_1").str.replace_all(prefix, ""))
    train = population.filter(pl.col(file_path_key).is_in(holdout["column_1"]).not_())
    test = population.filter(pl.col(file_path_key).is_in(holdout["column_1"]))
    return train, test


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population = merge_population_tables(cfg.population.tables)
    population, discards = extract_from_cfg(cfg, population=population)

    d = {}
    for i in range(len(discards)):
        d[i] = {
            "criteria": discards[i][0],
            "n_discards": discards[i][2],
            "n_population_pre_discard": discards[i][3],
            "n_population_post_discard": discards[i][3] - discards[i][2],
            "discards": discards[i][1],
        }

    with open(cfg.paths.discards_save_path, "w") as fp:
        json.dump(d, fp, indent=4)

    population.write_csv(cfg.paths.population_save_path + ".csv")

    train_pop, test_pop = make_train_test_split(cfg.paths.holdout_csv, population, cfg.population.population_key, cfg.prefix)

    train_pop.write_csv(cfg.paths.population_save_path + "train.csv")
    test_pop.write_csv(cfg.paths.population_save_path + "test.csv")


if __name__ == "__main__":
    main()
