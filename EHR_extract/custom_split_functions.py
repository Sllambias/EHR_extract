import logging
from EHR_extract.utils import merge_population_tables
from extract import handle_standard_condition
from EHR_extract.custom_find_functions import (
    find_images_with_predicted_classes,
    find_images_within_time_windows,
    match_images_with_child,
    match_value_with_child_cpr_on_birthdate,
)

custom_functions = {
    "find_images_within_time_windows": find_images_within_time_windows,
    "find_images_with_predicted_classes": find_images_with_predicted_classes,
    "match_images_with_child": match_images_with_child,
    "match_value_with_child_cpr_on_birthdate": match_value_with_child_cpr_on_birthdate,
}


def preterm_custom1(cfg, population):
    population = merge_population_tables(cfg.tables, population=population, strict=False)
    population = match_images_with_child(
        table_cfg=cfg.imaging_table,
        population=population,
    )
    # First get all hashed CPRs with cervix scan within week range
    for custom_cfg in cfg.get("imaging_matching_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        cervix_images_within_timerange, _ = fn(
            **args,
            population=population,
            population_key_column=cfg.population_key,
        )
    print(cervix_images_within_timerange[cfg.population_key])
    print(len(cervix_images_within_timerange), len(population))

    # First get all parity 1 IDs
    for criterion in cfg.get("conditional_criteria", {}):
        criterion_population = set()
        for condition in criterion.conditions:
            print(condition)
            if "standard" in condition.keys():
                matched_ids = handle_standard_condition(condition, population, cfg.population_key, cfg.population_key, False)
                conditional = condition.standard
            elif "custom" in condition.keys():
                fn = custom_functions[condition.function]
                args = condition.args
                matched_ids = fn(**args, population=population, population_key_column=cfg.population_key)
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

    # Then get all cervix scans within 16-24
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
        logging.info(
            f"After filtering on custom criteria: {custom_cfg.function} \n"
            f"Valid image+patient matches: {len(population)} with "
            f"unique {cfg.population.population_key}: {population[cfg.population.population_key].n_unique()} "
            f"and unique FILE_PATH: {population['FILE_PATH'].n_unique()}"
        )
