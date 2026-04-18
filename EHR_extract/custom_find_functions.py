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
    table,
    child_id_column,
    delivery_date_column,
    scan_date_column,
    ga_in_days_column,
    min_diff_days_scan_to_delivery,
    max_diff_days_scan_to_delivery,
    min_diff_ga_in_days_scan_to_delivery,
    max_diff_ga_in_days_scan_to_delivery,
    population,
):
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(child_id_column).is_in(population))
    print(f"Table rows matching population IDs: {len(table)} after filtering on {child_id_column}")
    print(f"Table unique population IDs: {table[child_id_column].n_unique()} after filtering on {child_id_column}")
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

    matching = filter_numeric_rows(matching, ga_in_days_column)
    matching = matching.with_columns(
        GA_in_days_at_scantime=(pl.col(ga_in_days_column)).cast(pl.Float64) - pl.col("diff_in_days_scan_to_delivery")
    )
    matching = matching.filter(
        (min_diff_ga_in_days_scan_to_delivery < pl.col("GA_in_days_at_scantime"))
        & (pl.col("GA_in_days_at_scantime") < max_diff_ga_in_days_scan_to_delivery)
    )
    matching_ids = set(matching[child_id_column])

    return matching_ids


def find_close_births(table, match_on, mom_column, birth_id_column, delivery_date_column, threshold_days, population):
    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    print(f"Table rows matching population IDs: {len(table)} after filtering on {match_on}")
    print(f"Table unique population IDs: {table[match_on].n_unique()} after filtering on {match_on}")

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
    # Sort by mother and birth date
    table_path = table
    table = load_table(table)
    print(f"Table rows total: {len(table)} for table: {table_path}")
    table = table.filter(pl.col(match_on).is_in(population))
    print(f"Table rows matching population IDs: {len(table)} after filtering on {match_on}")

    print(table[birth_id_column].value_counts())

    raise NotImplementedError

def find_pregnancy_start(table, birth_date_col, GA_days_col, pregnancy_start_col):
    table = table.with_columns(
        (
            pl.col(birth_date_col).str.to_date()
            - pl.duration(days=pl.col(GA_days_col).cast(pl.Int64, strict=False))
        ).alias(pregnancy_start_col)
    )
    return table

def find_GA_weeks(table, GA_days_col, GA_weeks_col):
    table = table.with_columns(
        (pl.col(GA_days_col).cast(pl.Int64, strict=False) / 7).alias(GA_weeks_col)
    )
    return table