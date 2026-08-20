import logging
import polars as pl
from EHR_extract.utils import merge_population_tables, merge_composed_population_tables
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
    population = merge_population_tables(cfg.tables, population=population, strict=True)
    population = merge_composed_population_tables(population, cfg.population_key, cfg.composed_tables)
    print("before", len(population))
    img_population = match_images_with_child(
        table_cfg=cfg.imaging_table,
        population=population,
    )

    # population = population.with_columns(pl.col("BIRTHDAY").dt.to_string("%Y-%m-%d %H:%M:%S"))
    # population = population.with_columns(pl.col("STUDY_DATE").dt.to_string("%Y-%m-%d"))
    # population = population.with_columns(pl.col("GA").cast(pl.String))

    # First get all hashed CPRs with cervix scan within week range
    for custom_cfg in cfg.get("imaging_matching_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        cervix_images_within_timerange, _ = fn(
            **args,
            population=img_population,
            population_key_column=cfg.population_key,
        )

    # Then get all parity 1 IDs
    for criterion in cfg.get("conditional_criteria", {}):
        criterion_population = set()
        for condition in criterion.conditions:
            if "standard" in condition.keys():
                matched_ids = handle_standard_condition(condition, population, "CPR_BARN", "CPR_BARN", False)
                conditional = condition.standard
            elif "custom" in condition.keys():
                fn = custom_functions[condition.function]
                args = condition.args
                matched_ids = fn(**args, population=population, population_key_column="CPR_BARN")
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

        parity1_child_ids = criterion_population.union(current_condition_population)
        parity1_population = population.filter(pl.col("CPR_BARN").is_in(parity1_child_ids))

    positives_img_criteria = set(cervix_images_within_timerange[cfg.population_key])
    positives_ehr_criteria = set(parity1_population[cfg.population_key])
    test_population = positives_img_criteria.intersection(positives_ehr_criteria)

    train_population = set(population[cfg.population_key])
    train_population.difference_update(test_population)

    print(f"Made train split of len: {len(train_population)} and test split of len {len(test_population)}")
    assert len(test_population & train_population) == 0, "polution between train & test"

    train_df = pl.DataFrame({cfg.population_key: list(train_population)})
    test_df = pl.DataFrame({cfg.population_key: list(test_population)})

    return train_df, test_df
