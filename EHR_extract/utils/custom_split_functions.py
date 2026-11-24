import logging
import numpy as np
import os
import polars as pl
from datetime import datetime
from EHR_extract.extract import handle_standard_condition
from EHR_extract.utils.custom_find_functions import (
    find_images_with_predicted_classes,
    find_images_within_time_windows,
    match_images_with_child,
    match_value_with_child_cpr_on_birthdate,
)
from EHR_extract.utils.utils import merge_composed_population_tables, merge_population_tables
from sklearn.model_selection import KFold

custom_functions = {
    "find_images_within_time_windows": find_images_within_time_windows,
    "find_images_with_predicted_classes": find_images_with_predicted_classes,
    "match_images_with_child": match_images_with_child,
    "match_value_with_child_cpr_on_birthdate": match_value_with_child_cpr_on_birthdate,
}


def preterm_custom1(cfg, population):
    population = merge_population_tables(cfg.tables, population=population, strict=True)
    population = merge_composed_population_tables(population, cfg.population_key, cfg.composed_tables)
    img_population = match_images_with_child(
        table_cfg=cfg.imaging_table,
        population=population,
    )

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

    positives_intersection = parity1_population.join(cervix_images_within_timerange, on=["CPR_BARN", "CPR_MOR"])
    test_population = set(positives_intersection[cfg.population_key])

    print(
        f"Positives from img criteria: {len(set(cervix_images_within_timerange[cfg.population_key]))}. Positives from EHR criteria: {len(set(parity1_population[cfg.population_key]))}. Intersection: {len(test_population)}"
    )
    train_population = set(population[cfg.population_key])
    train_population.difference_update(test_population)

    print(f"Made train split of len: {len(train_population)} and test split of len {len(test_population)}")
    assert len(test_population & train_population) == 0, "polution between train & test"

    train_df = pl.DataFrame({cfg.population_key: list(train_population)})
    test_df = pl.DataFrame({cfg.population_key: list(test_population)})

    return train_df, test_df


def kfold(cfg, population):
    timestamp = datetime.today().strftime("%Y-%m-%d")

    population = merge_population_tables(cfg.tables, population=population, strict=True)

    kf = KFold(n_splits=cfg.folds, shuffle=True)
    set_of_hashes = np.array(list(set(population[cfg.population_key])))
    print("Population size: ", len(set_of_hashes))
    test_ids = []

    for i, (train_index, test_index) in enumerate(kf.split(set_of_hashes)):
        print(f"Fold {i}:")
        print(f"  Train: index={len(train_index)}")
        print(f"  Test:  index={len(test_index)}")
        train_df = pl.DataFrame({cfg.population_key: set_of_hashes[train_index]})
        test_df = pl.DataFrame({cfg.population_key: set_of_hashes[test_index]})

        train_output_path = os.path.join(cfg.output_dir, f"train_split_fold{i}_{timestamp}.csv")
        test_output_path = os.path.join(cfg.output_dir, f"test_split_fold{i}_{timestamp}.csv")

        if os.path.exists(train_output_path) or os.path.exists(test_output_path):
            logging.warning(
                "SPLITS WITH IDENTICAL NAMES ALREADY EXIST. WILL NOT SAVE THE CURRENTLY GENERATED SPLITS."
                "If this is intended manually delete the old splits or change the name/version of the current."
            )
        else:
            train_df.write_csv(train_output_path, separator=",")
            test_df.write_csv(test_output_path, separator=",")

        test_ids.append(test_index.tolist())

    test_ids = [val for sublist in test_ids for val in sublist]
    assert len(set(test_ids)) == len(set_of_hashes), "something funky happened here"


def test_hospitals(cfg, population):
    """Train / test split of MOTHERS where every mother with at least one image at one of the chosen
    hospitals goes to the TEST set, with all her data (all her images, at every hospital and in every
    pregnancy, because the cohorts are split on the mother). Optionally united with an existing test split.

    cfg (the split config's population block):
      population_key:      CPR_MOR
      tables:              tables listing the mothers (mfr, nyfoedte); the mothers of imaging_table are added
      imaging_table:       table + columns {CPR_MOR: <mother column>, INSTITUTION_NAME: <institution column>}
      test_hospitals:
        mapping_file:      institution_mapping.yaml (needed for `sites`)
        sites:             site names from the mapping (all raw spellings of that hospital)
        institutions:      raw institution names
      previous_test_split: (optional) an existing test split CSV whose mothers stay in the test set
    Returns two tables with one column, population_key: the train mothers and the test mothers.
    """
    key = cfg.population_key
    population = merge_population_tables(cfg.tables, population=population, strict=False)
    all_mothers = set(population.get_column(key).drop_nulls().to_list())

    # mother x institution pairs of the image database (2 columns, read lazily)
    img = cfg.imaging_table
    mother_col, inst_col = img.columns[key], img.columns["INSTITUTION_NAME"]
    pairs = (pl.scan_csv(img.table, infer_schema=False)
               .select(pl.col(mother_col).alias(key), pl.col(inst_col).alias("institution"))
               .unique().collect(engine="streaming"))
    all_mothers |= set(pairs.get_column(key).drop_nulls().to_list())     # mothers with images belong to the population too

    # institution names that belong to the test hospitals
    th = cfg.test_hospitals
    sites = set(th.get("sites") or [])
    names = set(th.get("institutions") or [])
    if sites:
        if not th.get("mapping_file"):
            raise ValueError("test_hospitals: 'sites' needs 'mapping_file' (institution_mapping.yaml)")
        with open(th.mapping_file, encoding="utf-8") as f:
            mapping = yaml.safe_load(f) or {}
        site_of = {}
        for name, entry in mapping.items():
            name = "" if name is None else str(name)
            entry = entry if isinstance(entry, dict) else {}
            site_of[name] = entry["site"] if "site" in entry else name
            site_of.setdefault(name.strip(), site_of[name])
        unknown = sites - set(site_of.values())
        if unknown:
            raise ValueError(f"test_hospitals: sites {sorted(unknown)} are not in {th.mapping_file}")
        for n in pairs.get_column("institution").unique().to_list():
            key_n = "" if n is None else n
            if site_of.get(key_n, site_of.get(key_n.strip())) in sites:
                names.add(n)
    if not names:
        raise ValueError("test_hospitals: no institution name in the image database matches the chosen hospitals")

    at_hospitals = pairs.filter(pl.col("institution").is_in(list(names)))
    hospital_mothers = set(at_hospitals.get_column(key).drop_nulls().to_list())
    also_elsewhere = pairs.filter(pl.col(key).is_in(list(hospital_mothers)) & ~pl.col("institution").is_in(list(names)))
    test_population = set(hospital_mothers)
    logging.info(f"test_hospitals: institution names {sorted(str(n) for n in names)}")
    print(f"Mothers with >= 1 image at the test hospitals: {len(hospital_mothers):,} "
          f"(of whom {also_elsewhere.get_column(key).n_unique():,} also have images at other hospitals: "
          f"all their data goes to TEST)")

    previous = cfg.get("previous_test_split")
    if previous:
        previous_mothers = set(pl.read_csv(previous, infer_schema_length=0).get_column(key).drop_nulls().to_list())
        print(f"Mothers from the previous test split {os.path.basename(previous)}: {len(previous_mothers):,} "
              f"({len(previous_mothers - hospital_mothers):,} not already in the test set)")
        test_population |= previous_mothers

    train_population = all_mothers - test_population
    print(f"Made train split of len: {len(train_population)} and test split of len {len(test_population)}")
    assert len(test_population & train_population) == 0, "polution between train & test"  # check for set polution

    train_df = pl.DataFrame({key: sorted(train_population)}, schema={key: pl.String})
    test_df = pl.DataFrame({key: sorted(test_population)}, schema={key: pl.String})
    return train_df, test_df
