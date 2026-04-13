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


def find_images_and_timedeltas(
    table1,
    table2,
    child_id_column1,
    mom_id_column1,
    mom_id_column2,
    date_column1,
    date_column2,
    ga_column1,
    min_diff_days_scan_to_delivery,
    max_diff_days_scan_to_delivery,
    min_diff_ga_scan_to_delivery,
    max_diff_ga_scan_to_delivery,
    population,
):
    table_path1 = table1
    table1 = load_table(table1)
    print(f"Table rows total: {len(table1)} for table: {table_path1}")
    table1 = table1.filter(pl.col(child_id_column1).is_in(population))
    print(f"Table rows matching population IDs: {len(table1)} after filtering on {child_id_column1}")
    table_path2 = table2
    table2 = load_table(table2)
    print(f"Table rows total: {len(table2)} for table: {table_path2}")
    joined = table1.join(table2, left_on=mom_id_column1, right_on=mom_id_column2, how="inner")

    # Calculate absolute difference in days
    joined = joined.with_columns(
        diff_in_days_scan_to_delivery=(
            (pl.col(date_column1).str.to_date() - pl.col(date_column2).str.to_date()).dt.total_days()
        )
    )

    matching = joined.filter(
        (min_diff_days_scan_to_delivery < pl.col("diff_in_days_scan_to_delivery"))
        & (pl.col("diff_in_days_scan_to_delivery") < max_diff_days_scan_to_delivery)
    )

    matching = filter_numeric_rows(matching, ga_column1)
    matching = matching.with_columns(
        GA_in_days_at_scantime=(pl.col(ga_column1)).cast(pl.Float64) - pl.col("diff_in_days_scan_to_delivery")
    )
    matching = matching.filter(
        (min_diff_ga_scan_to_delivery < pl.col("GA_in_days_at_scantime"))
        & (pl.col("GA_in_days_at_scantime") < max_diff_ga_scan_to_delivery)
    )
    matching_ids = set(matching[child_id_column1])

    return matching_ids, matching[
        "CPR_BARN", "CPR_MODER", "image_path", "GA_in_days_at_scantime", "diff_in_days_scan_to_delivery"
    ]


def find_close_siblings(table, match_on, mom_column, delivery_date_column, threshold_days, population):
    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    print(f"Table rows matching population IDs: {len(table)} after filtering on {match_on}")

    table = table.with_columns(pl.col(delivery_date_column).str.to_date())
    table = table.sort([mom_column, delivery_date_column])

    # Calculate difference between consecutive births for each mother
    table = table.with_columns(
        diff=pl.col(delivery_date_column).diff().over(mom_column),
        prev_child_ID=pl.col(match_on).shift(1).over(mom_column),
    )
    # Filter to find children with siblings born less than 40 weeks apart
    # 40 weeks = 280 days
    close_siblings = table.filter(pl.col("diff").dt.total_days() < 280)
    # Get the CPR_BARN values to exclude
    siblings_to_exclude = set(close_siblings[match_on]) | set(close_siblings["prev_child_ID"])
    return siblings_to_exclude
