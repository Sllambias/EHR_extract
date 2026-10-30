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
    check_duplicates,
    deduplicate_on_key,
    filter_numeric_rows,
    format_criterion,
    get_python_operator,
    load_table,
    merge_composed_population_tables,
    merge_population_tables,
    take_latest_row,
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
    """Rows of the condition's table whose ID is in the population and that pass its operator."""
    if condition.table == "population":
        table = population.clone()
    else:
        columns = [condition.match_on, condition.get("column"), condition.get("retrieve"), condition.get("date_col")]
        table = load_table(condition.table, strict=strict, columns=columns)
    logging.debug(
        f"Table rows / unique IDs total: {len(table)} / {table[condition.match_on].n_unique()} \
            for table: {condition.table}"
    )

    table = table.filter(pl.col(condition.match_on).rechunk().is_in(population[population_key].implode()))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
        after filtering on {condition.match_on}"
    )

    if condition.get("operator", None) is None:
        return table

    py_operator = get_python_operator(condition.operator)
    if condition.operator in [">", "<", ">=", "<="]:
        table = filter_numeric_rows(table, condition.column)
    table = table.filter(py_operator(pl.col(condition.column), condition.value))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[condition.match_on].n_unique()} \
            after filtering on {condition.column} {condition.operator} {condition.value}"
    )
    return table


def get_retrieved_values(table, condition, population_key, name):
    """The condition's `retrieve` value for each matched ID, as columns [population_key, name]."""
    values = table.drop_nulls(condition.retrieve)
    if condition.get("date_col") is not None:
        values = take_latest_row(values, condition.match_on, condition.date_col)
    values = check_duplicates(values.select([condition.match_on, condition.retrieve]).unique(), condition.match_on)
    return values.rename({condition.match_on: population_key, condition.retrieve: name})


def keep_matched_values(value_tables, matched_ids, population_key):
    """Keep only the values of IDs matched by every condition in the group."""
    return [values.filter(pl.col(population_key).is_in(matched_ids)) for values in value_tables if values is not None]


def add_criterion_column(population, population_key, name, matched_ids, value_tables):
    """Add column `name`: the retrieved values if any condition has `retrieve`, else whether the ID matched."""
    if name in population.columns:
        raise ValueError(f"Cannot add column {name}: the population already has a column with that name")
    if len(value_tables) == 0:
        matched = population.get_column(population_key).rechunk().is_in(matched_ids)
        return population.with_columns(matched.alias(name))
    values = pl.concat(value_tables, how="vertical_relaxed")
    values = values.unique(subset=population_key, keep="first", maintain_order=True)
    return population.join(values, on=population_key, how="left", maintain_order="left")


def extract_from_cfg(cfg, population):
    all_discards = []

    logging.info(
        f"Population size: {len(population)} with unique IDs: {population[cfg.population.population_key].n_unique()}",
    )
    for criterion in cfg.get("conditional_criteria", {}):
        criterion_population = set()
        criterion_values = []
        for condition in criterion.conditions:
            condition_values = None
            if "standard" in condition.keys():
                table = handle_standard_condition(
                    condition, population, cfg.population.population_key, cfg.population.population_key, cfg.strict
                )
                matched_ids = set(table[condition.match_on])
                if criterion.action == "add_column" and condition.get("retrieve") is not None:
                    condition_values = get_retrieved_values(table, condition, cfg.population.population_key, criterion.name)
                del table  # free the rows before the next condition loads its table
                conditional = condition.standard
            elif "custom" in condition.keys():
                if criterion.action == "add_column" and condition.get("retrieve") is not None:
                    raise NotImplementedError("retrieve is only supported on standard conditions")
                fn = custom_functions[condition.function]
                args = condition.args
                matched_ids = fn(**args, population=population, population_key_column=cfg.population.population_key)
                conditional = condition.custom
            if conditional is None:
                current_condition_population = matched_ids
                current_values = [condition_values]
            elif conditional == "and":
                current_condition_population = current_condition_population.intersection(matched_ids)
                current_values.append(condition_values)
            elif conditional == "or":
                criterion_population = criterion_population.union(current_condition_population)
                criterion_values += keep_matched_values(
                    current_values, current_condition_population, cfg.population.population_key
                )
                current_condition_population = matched_ids
                current_values = [condition_values]
            else:
                logging.warn("wow, weird condition")
        criterion_population = criterion_population.union(current_condition_population)
        criterion_values += keep_matched_values(current_values, current_condition_population, cfg.population.population_key)
        if criterion.action == "add_column":
            population = add_criterion_column(
                population, cfg.population.population_key, criterion.name, criterion_population, criterion_values
            )
            logging.info(
                f"Added column {criterion.name}: {len(criterion_population)} IDs matched criteria "
                f"{format_criterion(criterion)} \n"
            )
            all_discards.append([OmegaConf.to_container(criterion), [], 0, len(population)])
            continue
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
    train = population.filter(~pl.col(split_key).rechunk().is_in(holdout))
    test = population.filter(pl.col(split_key).rechunk().is_in(holdout))
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
