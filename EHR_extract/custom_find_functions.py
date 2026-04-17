import polars as pl
from EHR_extract.utils import load_table, filter_numeric_rows


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
    print(f"Table rows total: {len(table_birth)} for table: {table_birth_path}")
    table_birth = table_birth.filter(pl.col(id_col_birth).is_in(population))
    print(f"Table rows matching population IDs: {len(table_birth)} after filtering on {id_col_birth}")
    table_scan_path = table_scan
    table_scan = load_table(table_scan)
    print(f"Table rows total: {len(table_scan)} for table: {table_scan_path}")
    table_scan = table_scan.filter(pl.col(id_col_scan).is_in(population))
    print(f"Table rows matching population IDs: {len(table_scan)} after filtering on {id_col_scan}")

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


def find_images_and_timedeltas(
    table,
    child_id_column,
    scan_date_column,
    image_path_column,
    min_diff_days_scan_to_delivery,
    max_diff_days_scan_to_delivery,
    min_ga_in_days_at_scan,
    max_ga_in_days_at_scan,
    imaging_metadata,
    population,
    population_key_column,
    delivery_date_column="Birthday",
    ga_in_days_at_delivery_column="GA",
):
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(child_id_column).is_in(population.get_column(population_key_column)))
    print(
        f"Table rows / unique IDs matching population IDs: {len(table)} /  {table[child_id_column].n_unique()} after filtering on {child_id_column}"
    )

    table = table.join(population, left_on=child_id_column, right_on=population_key_column)
    print(table.head())
    # Calculate absolute difference in days
    table = table.with_columns(
        diff_in_days_scan_to_delivery=(
            (
                pl.col(delivery_date_column).str.to_date()
                - pl.col(scan_date_column).cast(pl.String).str.to_date(format="%Y%m%d")
            ).dt.total_days()
        )
    )

    matching = table.filter(
        (min_diff_days_scan_to_delivery < pl.col("diff_in_days_scan_to_delivery"))
        & (pl.col("diff_in_days_scan_to_delivery") < max_diff_days_scan_to_delivery)
    )

    matching = filter_numeric_rows(matching, ga_in_days_at_delivery_column)
    matching = matching.with_columns(
        GA_in_days_at_scantime=(pl.col(ga_in_days_at_delivery_column)).cast(pl.Float64)
        - pl.col("diff_in_days_scan_to_delivery")
    )
    matching = matching.filter(
        (min_ga_in_days_at_scan < pl.col("GA_in_days_at_scantime"))
        & (pl.col("GA_in_days_at_scantime") < max_ga_in_days_at_scan)
    )
    matching_ids = set(matching[child_id_column])
    return matching_ids, matching


def find_images_with_predicted_classes(
    table,
    classes,
    child_id_column,
    class_column,
    image_path_column,
    imaging_metadata,
    imaging_metadata_image_path_column,
    population,
    population_key_column,
):
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    matching = table.filter(pl.col(image_path_column).is_in(imaging_metadata[imaging_metadata_image_path_column]))
    print(f"Table rows matching imaging metadata: {len(matching)}")

    # Filter to find images with predicted class in the provided classes
    matching = matching.filter(pl.col(class_column).is_in(classes))
    print(f"Table rows matching predicted classes: {len(matching)}")

    imaging_metadata = imaging_metadata.filter(pl.col(imaging_metadata_image_path_column).is_in(matching[image_path_column]))
    matching_ids = set(imaging_metadata[child_id_column])
    return matching_ids, imaging_metadata


def find_close_births(table, match_on, mom_column, birth_id_column, delivery_date_column, threshold_days, population):
    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    print(
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
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    print(f"Table rows matching population IDs: {len(table)} after filtering on {match_on}")
    print(table[birth_id_column].value_counts())
    """
    raise NotImplementedError
