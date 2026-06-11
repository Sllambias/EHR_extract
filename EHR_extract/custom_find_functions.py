import logging
import polars as pl
from EHR_extract.utils import filter_numeric_rows, get_python_operator, load_table


def match_value_on_birthdate(population, value_time_column, population_birthdate_column, population_gestational_age_column):
    population = filter_numeric_rows(population, population_gestational_age_column)
    population = population.with_columns(
        conception_date=pl.col(population_birthdate_column).str.to_datetime()
        - pl.duration(days=pl.col(population_gestational_age_column).cast(pl.Int64))
    )
    population = population.filter(
        (pl.col(value_time_column).str.to_datetime() >= pl.col("conception_date"))
        & (pl.col(value_time_column).str.to_datetime() <= pl.col(population_birthdate_column).str.to_datetime())
    )
    return population


def match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate(
    operator,
    value,
    value_table_path,
    value_column,
    value_time_column,
    value_id_column,
    mapping_table_path,
    mapping_table_id_column,
    mapping_table_mom_cpr_column,
    population,
    population_mom_cpr_column,
    population_child_cpr_column,
    population_birth_column,
    population_gestational_age_column,
    population_key_column,
):
    """
    This function takes tables A, B and C and matches a Value in Table A with a child CPR in Table C by:
    Finding the values, LPR_ID and value_timestamps in Table A
    Then matching the LPR_ID to the mom_CPR in Table B
    Then matching the mom_CPR to child_CPR in Table C
    and finally filtering the child_CPR if the value_timestamps fall within their pregnancy
    """
    value_table = load_table(value_table_path)
    py_operator = get_python_operator(operator)
    value_table = value_table.filter(py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path)
    joined = value_table.join(
        mapping_table,
        left_on=value_id_column,
        right_on=mapping_table_id_column,
        how="inner",
    )

    joined = joined.join(
        population,
        left_on=mapping_table_mom_cpr_column,
        right_on=population_mom_cpr_column,
        how="inner",
    )
    joined = match_value_on_birthdate(
        population=joined,
        value_time_column=value_time_column,
        population_birthdate_column=population_birth_column,
        population_gestational_age_column=population_gestational_age_column,
    )
    matches = set(joined[population_child_cpr_column].unique())
    return matches


def match_value_with_child_cpr_on_birth_id(
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
):
    value_table = load_table(value_table_path)

    py_operator = get_python_operator(operator)
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

    matches = set(joined[mapping_table_child_cpr_column].unique())
    return matches


def match_value_with_child_cpr_on_birthdate(
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
):
    value_table = load_table(value_table_path)

    # Filter based on operator and value
    py_operator = get_python_operator(operator)
    value_table = value_table.filter(py_operator(pl.col(value_column), value))
    # Join on mother's CPR
    joined = value_table.join(population, left_on=value_mother_cpr_column, right_on=population_mother_cpr_column, how="inner")

    joined = match_value_on_birthdate(
        population=joined,
        value_time_column=value_time_column,
        population_birthdate_column=population_birth_column,
        population_gestational_age_column=population_gestational_age_column,
    )

    # Get the unique child CPRs
    matches = set(joined[population_child_cpr_column].unique())
    return matches


def match_years_with_child_cpr_on_birthdate(
    date_start,
    date_end,
    value_table_path,
    value_time_column,
    value_child_cpr_column,
    population,
    population_key_column,
):
    value_table = load_table(value_table_path)

    start = pl.lit(date_start).str.to_date("%d%m%Y")
    end = pl.lit(date_end).str.to_date("%d%m%Y")
    value_table = value_table.filter(pl.col(value_time_column).str.to_datetime(strict=False).dt.date().is_between(start, end))

    joined = population.join(value_table, left_on=population_key_column, right_on=value_child_cpr_column, how="inner")
    matches = set(joined[population_key_column].unique())
    return matches


def match_images_with_child(
    population, table_cfg, study_date_key="STUDY_DATE", mom_key="CPR_MOR", birthday_key="BIRTHDAY", ga_key="GA"
):
    """
    Barn CPR periode fra  fødselsdato - GA i dage til fødselsdato og så er alle billeder fra mor i den periode tilskrevet
    barns CPR. Så kan tvillinger også få tildelt samme billeder.
    """
    table = load_table(table_cfg.table, strict=False)
    table = table.select(list(table_cfg.columns.values()))
    table = table.rename({v: k for k, v in table_cfg.columns.items()})
    table = table.join(population, left_on=mom_key, right_on=mom_key)
    table = table.with_columns(pl.col(birthday_key).str.to_datetime())
    table = table.with_columns(pl.col(study_date_key).cast(pl.String).str.to_date("%Y%m%d"))
    table = table.with_columns(pl.col(ga_key).str.to_integer(strict=False))
    table = table.unique()
    table = table.with_columns(
        image_during_pregnancy=pl.col(study_date_key).is_between(
            pl.col(birthday_key) - pl.duration(days=pl.col(ga_key)), pl.col(birthday_key)
        )
    )

    table = table.filter(pl.col("image_during_pregnancy"))
    logging.info(f"Valid images: {len(table)} after matching image + EHR matching.  \n")
    return table


def merge_population_on(population, table, merge_key, population_key_column):
    logging.info(f"Merging population of size {len(population)} with {table}")
    table = load_table(table)
    logging.info(logging.info(f"Merging population with table of size {len(table)}"))
    population = population.join(table, left_on=population_key_column, right_on=merge_key)
    logging.info(f"Population size after merge: {len(population)}")
    return population


def find_images_within_time_windows(
    scan_date_column,
    image_path_column,
    min_diff_days_scan_to_delivery,
    max_diff_days_scan_to_delivery,
    min_ga_in_days_at_scan,
    max_ga_in_days_at_scan,
    population,
    population_key_column,
    population_delivery_date_column="BIRTHDAY",
    population_ga_in_days_at_delivery_column="GA",
):
    discard_stats = {"n_population_before_discard": len(population)}
    # Calculate absolute difference in days
    population = population.with_columns(
        diff_in_days_scan_to_delivery=((pl.col(population_delivery_date_column) - pl.col(scan_date_column)).dt.total_days())
    )

    population = population.filter(
        (min_diff_days_scan_to_delivery < pl.col("diff_in_days_scan_to_delivery"))
        & (pl.col("diff_in_days_scan_to_delivery") < max_diff_days_scan_to_delivery)
    )

    population = filter_numeric_rows(population, population_ga_in_days_at_delivery_column)
    population = population.with_columns(
        GA_in_days_at_scantime=(pl.col(population_ga_in_days_at_delivery_column)).cast(pl.Float64)
        - pl.col("diff_in_days_scan_to_delivery")
    )
    population = population.filter(
        (min_ga_in_days_at_scan < pl.col("GA_in_days_at_scantime"))
        & (pl.col("GA_in_days_at_scantime") < max_ga_in_days_at_scan)
    )
    population = population.drop(["GA_in_days_at_scantime", "diff_in_days_scan_to_delivery"])
    discard_stats.update(
        {
            "criteria": "find_images_and_timedeltas",
            "discards": "N/A",
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def find_images_with_predicted_classes(
    table,
    classes,
    class_column,
    image_path_column,
    population,
    population_image_path_column,
    population_key_column,
):
    discard_stats = {"n_population_before_discard": len(population)}

    table_path = table
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    matched_paths = table.filter(pl.col(class_column).is_in(classes))[image_path_column]
    logging.debug(f"Table rows matching predicted classes: {len(matched_paths)}")

    population = population.filter(pl.col(population_image_path_column).is_in(matched_paths))
    logging.debug(f"Table rows matching population: {len(population)}")

    discard_stats.update(
        {
            "criteria": "find_images_with_predicted_classes",
            "discards": "N/A",
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def find_close_births(
    value,
    operator,
    table,
    match_on,
    mom_column,
    birth_id_column,
    delivery_date_column,
    population,
    population_key_column,
):
    # Sort by mother and birth date
    population = set(population.get_column(population_key_column))
    py_operator = get_python_operator(operator)
    table_path = table
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    table = table.with_columns(pl.col(delivery_date_column).str.to_date())
    table = table.sort([mom_column, delivery_date_column])

    # Calculate difference between consecutive births for each mother
    table = table.with_columns(
        diff=pl.col(delivery_date_column).diff().over(mom_column),
        prev_child_ID=pl.col(match_on).shift(1).over(mom_column),
        prev_child_birth_ID=pl.col(birth_id_column).shift(1).over(mom_column),
    )
    # Filter to find children with siblings born less than 40 weeks apart
    # 40 weeks = 280 days
    close_siblings = table.filter(
        (py_operator(pl.col("diff").dt.total_days(), value)) & (pl.col(birth_id_column) != pl.col("prev_child_birth_ID"))
    )
    close_siblings = close_siblings.filter(pl.col(match_on).is_in(population))

    # Get the CPR_BARN values to exclude
    siblings_to_exclude = set(close_siblings[match_on]) | set(close_siblings["prev_child_ID"])
    return siblings_to_exclude


def find_duplicated_ids(table, match_on, id_column, population, population_key_column):
    population = set(population.get_column(population_key_column))
    table_path = table
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    duplicated_ids = table.filter(table[id_column].is_duplicated())
    duplicated_ids = duplicated_ids.filter(pl.col(match_on).is_in(population))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[match_on].n_unique()} \
            after filtering on {match_on}"
    )
    duplicated_ids = set(duplicated_ids[match_on])
    return duplicated_ids
