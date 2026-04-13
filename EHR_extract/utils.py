import polars as pl


def load_table(path, n_rows=None):
    if path.endswith(".csv"):
        return pl.read_csv(path, n_rows=n_rows)
    else:
        raise NotImplementedError(f"Unknown file type for path: {path}. Did you remember to add the file extension?")


def get_python_operator(operator_str):
    if operator_str == "in":
        return lambda col, val: col.is_in(val)
    elif operator_str == "not_in":
        raise NotImplementedError("NOT IN is not implemented as it should not be used. Be precise and use the IN operator.")
    elif operator_str == "==":
        return lambda col, val: col.cast(pl.String) == val
    elif operator_str == "!=":
        return lambda col, val: col.cast(pl.String) != val
    elif operator_str == ">":
        return lambda col, val: col.cast(pl.Float64) > val
    elif operator_str == "<":
        return lambda col, val: col.cast(pl.Float64) < val
    elif operator_str == ">=":
        return lambda col, val: col.cast(pl.Float64) >= val
    elif operator_str == "<=":
        return lambda col, val: col.cast(pl.Float64) <= val
    else:
        raise NotImplementedError(f"Unknown operator: {operator_str}")


def filter_numeric_rows(table, column):
    table = table.with_columns(parsed=pl.col(column).cast(pl.Float64, strict=False))
    table = table.filter(pl.col("parsed").is_not_null())
    return table


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


def find_timedelta(
    table1,
    table2,
    id_column1,
    id_column2,
    date_column1,
    date_column2,
    min_diff,
    max_diff,
    population,
):
    table_path1 = table1
    table1 = load_table(table1)
    print(f"Table rows total: {len(table1)} for table: {table_path1}")
    table1 = table1.filter(pl.col(id_column1).is_in(population))
    print(f"Table rows matching population IDs: {len(table1)} after filtering on {id_column1}")
    table_path2 = table2
    table2 = load_table(table2)
    print(f"Table rows total: {len(table2)} for table: {table_path2}")
    table2 = table2.filter(pl.col(id_column2).is_in(population))
    print(f"Table rows matching population IDs: {len(table2)} after filtering on {id_column2}")

    # Join tables on the ID columns
    joined = table1.join(table2, left_on=id_column1, right_on=id_column2, how="inner")

    # Calculate absolute difference in days
    joined = joined.with_columns(
        diff_days=((pl.col(date_column1).str.to_date() - pl.col(date_column2).str.to_date()).dt.total_days().abs())
    )
    # Filter to find records where difference equals target days
    matching = joined.filter((min_diff < pl.col("diff_days")) & (pl.col("diff_days") < max_diff))

    # Get IDs that have the target difference
    matching_ids = set(matching[id_column1])
    return matching_ids


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


def update_population(population, subset, action):
    if action == "exclude":
        discards = subset
        population.difference_update(subset)
    elif action == "include":
        discards = population.difference(subset)
        population = population.intersection(subset)
    else:
        raise NotImplementedError(f"unexpected action: {action}")
    return population, discards
