import hydra
import json
import logging
import polars as pl
from dotenv import load_dotenv
from EHR_extract.utils.custom_find_functions import (
    find_close_births,
    find_duplicated_ids,
    find_images_with_predicted_classes,
    find_images_within_time_windows,
    match_images_with_child,
    match_value_with_child_cpr_on_birth_id,
    match_value_with_child_cpr_on_birthdate,
    match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate,
    match_years_with_child_cpr_on_birthdate,
    merge_population_on,
)
from EHR_extract.utils.paths import get_config_path
from EHR_extract.utils.utils import (
    RecursiveSearchpathPlugin,
    deduplicate_on_key,
    filter_numeric_rows,
    format_criterion,
    get_python_operator,
    load_table,
    merge_composed_population_tables,
    merge_population_tables,
    update_population,
)
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

load_dotenv()
Plugins.instance().register(RecursiveSearchpathPlugin)


custom_functions = {
    "find_close_births": find_close_births,
    "find_images_within_time_windows": find_images_within_time_windows,
    "find_images_with_predicted_classes": find_images_with_predicted_classes,
    "find_duplicated_ids": find_duplicated_ids,
    "match_images_with_child": match_images_with_child,
    "match_value_with_child_cpr_on_birth_id": match_value_with_child_cpr_on_birth_id,
    "match_value_with_child_cpr_on_birthdate": match_value_with_child_cpr_on_birthdate,
    "match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate": match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate,
    "match_years_with_child_cpr_on_birthdate": match_years_with_child_cpr_on_birthdate,
    "merge_population_on": merge_population_on,
}


def handle_standard_condition(condition, population, population_key_column, population_key, strict):
    if condition.table == "population":
        table = population.clone()
    else:
        table = load_table(condition.table, strict=strict)
    logging.debug(
        f"Table rows / unique IDs total: {len(table)} / {table[condition.match_on].n_unique()} \
            for table: {condition.table}"
    )

    table = table.filter(pl.col(condition.match_on).is_in(population[population_key]))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
        after filtering on {condition.match_on}"
    )

    if condition.get("operator", None) is None:
        return set(table[condition.match_on])

    py_operator = get_python_operator(condition.operator)
    if condition.operator in [">", "<", ">=", "<="]:
        table = filter_numeric_rows(table, condition.column)
    table = table.filter(py_operator(pl.col(condition.column), condition.value))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
            after filtering on {condition.column} {condition.operator} {condition.value}"
    )
    return set(table[condition.match_on])


def extract_from_cfg(cfg, population):
    all_discards = []

    logging.info(
        f"Population size: {len(population)} with unique IDs: {population[cfg.population.population_key].n_unique()}",
    )
    for criterion in cfg.get("conditional_criteria", {}):
        criterion_population = set()
        for condition in criterion.conditions:
            if "standard" in condition.keys():
                matched_ids = handle_standard_condition(
                    condition, population, cfg.population.population_key, cfg.population.population_key, cfg.strict
                )
                conditional = condition.standard
            elif "custom" in condition.keys():
                fn = custom_functions[condition.function]
                args = condition.args
                matched_ids = fn(**args, population=population, population_key_column=cfg.population.population_key)
                conditional = condition.custom
            if conditional is None:
                current_condition_population = matched_ids
            elif conditional == "and":
                current_condition_population = current_condition_population.intersection(matched_ids)
            elif conditional == "or":
                criterion_population = criterion_population.union(current_condition_population)
                current_condition_population = matched_ids
            else:
                logging.warn("wow, weird condition")

        criterion_population = criterion_population.union(current_condition_population)
        population, discards, n_discards, n_population_before_discard = update_population(
            population=population,
            key=cfg.population.population_key,
            subset=set(criterion_population),
            action=criterion.action,
        )
        logging.info(f"Population size: {len(population)} after filtering on criteria {format_criterion(criterion)} \n")
        all_discards.append([OmegaConf.to_container(criterion), list(discards), n_discards, n_population_before_discard])

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
    output_dir_path = Path(cfg.paths.output_dir)
    output_dir_path.mkdir(exist_ok=True)
    discards_file_path = Path(cfg.paths.discards_save_path)
    discards_file_path.parent.mkdir(exist_ok=True)

    population = pl.DataFrame()

    if cfg.population.get("tables", None) is not None:
        population = merge_population_tables(cfg.population.tables, population)
    if cfg.population.get("composed_tables", None) is not None:
        population = merge_composed_population_tables(
            population, cfg.population.population_key, cfg.population.composed_tables
        )
    if cfg.population.deduplication_key is not None:
        population = deduplicate_on_key(population, population_key=cfg.population.deduplication_key)
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

    discards_file_path.write_text(json.dumps(d, indent=4))

    population.write_csv(cfg.paths.population_save_path + "_train_and_test.csv")

    if cfg.paths.holdout_csv is not None:
        train_pop, test_pop = make_train_test_split(cfg.paths.holdout_csv, population, cfg.population.split_key)

        if cfg.paths.get("exclude_train_subjects_not_in_csv", None) is not None:
            _, train_pop = make_train_test_split(
                cfg.paths.exclude_train_subjects_not_in_csv, population, cfg.population.split_key
            )

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
