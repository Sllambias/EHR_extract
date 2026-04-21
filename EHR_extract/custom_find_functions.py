import polars as pl
from EHR_extract.utils import load_table, filter_numeric_rows
import logging


def find_scantime_ga(
    table_birth,
    table_scan,
    id_col_birth,
    id_col_scan,
    birth_date_col,
    scan_date_col,
    birth_ga,
    min_diff,
    max_diff,
    population,
):
    """
    WIP

    table_birth_path = table_birth
    table_birth = load_table(table_birth)
    logging.info(f"Table rows total: {len(table_birth)} for table: {table_birth_path}")
    table_birth = table_birth.filter(pl.col(id_col_birth).is_in(population))
    logging.info(f"Table rows matching population IDs: {len(table_birth)} after filtering on {id_col_birth}")
    table_scan_path = table_scan
    table_scan = load_table(table_scan)
    logging.info(f"Table rows total: {len(table_scan)} for table: {table_scan_path}")
    table_scan = table_scan.filter(pl.col(id_col_scan).is_in(population))
    logging.info(f"Table rows matching population IDs: {len(table_scan)} after filtering on {id_col_scan}")

    # Join tables on the ID columns
    joined = table_birth.join(table_scan, left_on=id_col_birth, right_on=id_col_scan, how="inner")

    # Calculate absolute difference in days
    joined = joined.with_columns(
        diff_days=((pl.col(birth_date_col).str.to_date() - pl.col(scan_date_col).str.to_date()).dt.total_days())
    )
    joined = joined.with_columns(GA_at_scantime=(pl.col(birth_ga)).cast(pl.Float64) - pl.col("diff_days"))
    # Filter to find records where difference equals target days
    matching = joined.filter((min_diff < pl.col("GA_at_scantime")) & (pl.col("GA_at_scantime") < max_diff))
    # Get IDs that have the target difference
    matching_ids = set(matching[id_col_birth])
    return matching_ids
    """
    raise NotImplementedError


def match_images_with_child(
    population, table_cfg, study_date_key="STUDY_DATE", mom_key="CPR_MOR", birthday_key="BIRTHDAY", ga_key="GA"
):
    """
    Barn CPR periode fra  fødselsdato - GA i dage til fødselsdato og så er alle billeder fra mor i den periode tilskrevet
    barns CPR. Så kan tvillinger også få tildelt samme billeder.
    """
    table = load_table(table_cfg.table)
    table = table.select(list(table_cfg.columns.values()))
    table = table.rename({v: k for k, v in table_cfg.columns.items()})
    table = table.join(population, left_on=mom_key, right_on=mom_key)
    table = table.with_columns(pl.col(birthday_key).str.to_date())
    table = table.with_columns(pl.col(study_date_key).cast(pl.String).str.to_date("%Y%m%d"))
    table = table.with_columns(pl.col(ga_key).str.to_integer(strict=False))

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
    population = population.join(table, left_on=population_key_column, right_on=merge_key)
    logging.info(f"Population size after merge: {len(population)}")
    return population


def find_images_and_timedeltas(
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


def find_close_births(table, match_on, mom_column, birth_id_column, delivery_date_column, threshold_days, population):
    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[match_on].n_unique()} after filtering on {match_on}"
    )

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
        (pl.col("diff").dt.total_days() < 280) & (pl.col(birth_id_column) != pl.col("prev_child_birth_ID"))
    )
    # Get the CPR_BARN values to exclude
    siblings_to_exclude = set(close_siblings[match_on]) | set(close_siblings["prev_child_ID"])
    return siblings_to_exclude


def find_multiple_pregnancies(table, match_on, birth_id_column, population):
    """
    WIP

    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    logging.info(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    logging.info(f"Table rows matching population IDs: {len(table)} after filtering on {match_on}")
    logging.info(table[birth_id_column].value_counts())
    """
    raise NotImplementedError
