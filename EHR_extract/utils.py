import polars as pl
import json


def check_duplicates(table, population_column, allow_duplicates=False):
    duplicates = table[population_column].value_counts().filter(pl.col("count") > 1)
    if duplicates.height > 0:
        if not allow_duplicates:
            raise ValueError(f"Duplicate entries for key column {population_column}. Examples: {duplicates.head(5)}")
        else:
            table = table.group_by(population_column).agg(pl.col("*").first())
            assert(len(table[population_column].unique()) == len(table[population_column]))
    return table

def take_latest_row(table, key_column, date_col):
    table = table.sort([key_column, date_col])
    table = table.group_by(key_column).agg(pl.col("*").last())
    return table
    
def load_table_path(path, strict=True, n_rows=None, has_header=True, null_values=None):
    if strict:
        ignore_errors = False
    else:
        ignore_errors = True

    if path.endswith(".csv"):
        try:
            return pl.read_csv(path, ignore_errors=ignore_errors, n_rows=n_rows, has_header=has_header, null_values=null_values)
        except pl.exceptions.ComputeError:
            return pl.read_csv(
                path, ignore_errors=ignore_errors, infer_schema_length=1000000, n_rows=n_rows, has_header=has_header, null_values=null_values
            )
    else:
        raise NotImplementedError(f"Unknown file type for path: {path}. Did you remember to add the file extension?")


def expr_startswith(col: pl.Expr, val) -> pl.Expr:
    s = col.cast(pl.String)
    return pl.any_horizontal(
        [s.str.starts_with(p) for p in val]
    )

def load_table(
    table_cfg,
    strict=True,
    n_rows=None,
    has_header=True,
    null_values=None,
    _join_depth: int = 0,
):
    if isinstance(table_cfg, str):
        return load_table_path(
            table_cfg,
            strict=strict,
            n_rows=n_rows,
            has_header=has_header,
            null_values=null_values,
        )
    table1 = load_table(
        table_cfg["table1"],
        strict=strict,
        n_rows=n_rows,
        has_header=has_header,
        null_values=null_values,
        _join_depth=_join_depth + 1,
    )
    table2 = load_table(
        table_cfg["table2"],
        strict=strict,
        n_rows=n_rows,
        has_header=has_header,
        null_values=null_values,
        _join_depth=_join_depth + 1,
    )
    left_on = table_cfg["left_on"]
    right_on = table_cfg["right_on"]
    # Use a unique suffix per nested join so `_right` from an inner join
    # does not collide when an outer join also has overlapping columns.
    suffix = f"_join{_join_depth}"
    return table1.join(
        table2,
        left_on=left_on,
        right_on=right_on,
        how="left",
        suffix=suffix,
    )

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
        return lambda col, val: col.cast(pl.Float64, strict=False) > val
    elif operator_str == "<":
        return lambda col, val: col.cast(pl.Float64, strict=False) < val
    elif operator_str == ">=":
        return lambda col, val: col.cast(pl.Float64, strict=False) >= val
    elif operator_str == "<=":
        return lambda col, val: col.cast(pl.Float64, strict=False) <= val
    elif operator_str == "between":
        # Inclusive range: value is [low, high]
        return lambda col, val: col.cast(pl.Float64, strict=False).is_between(
            val[0], val[1], closed="both"
        )
    elif operator_str == "not_null":
        return lambda col, val: col.is_not_null()
    elif operator_str == "startswith":
        return expr_startswith
    elif operator_str == "is_true":
        return lambda col, val: col.cast(pl.Boolean, strict=False) == True
    elif operator_str == "is_false":
        return lambda col, val: col.cast(pl.Boolean, strict=False) == False
    else:
        raise NotImplementedError(f"Unknown operator: {operator_str}")


def filter_numeric_rows(table, column):
    table = table.with_columns(parsed=pl.col(column).cast(pl.Float64, strict=False))
    table = table.filter(pl.col("parsed").is_not_null())
    return table


def update_population(population, key, subset, action):
    pre_discard_population = len(population)
    population_set = set(population[key])
    if action == "exclude":
        discards = subset
        population_set.difference_update(subset)
    elif action == "include":
        discards = population_set.difference(subset)
        population_set = population_set.intersection(subset)
    else:
        raise NotImplementedError(f"unexpected action: {action}")
    population = population.filter(pl.col(key).is_in(population_set))
    return population, discards, len(discards), pre_discard_population


def write_imaging_metadata_to_formats(imaging_dataframe, output_formats, path):
    for output_format in output_formats:
        if output_format == "csv":
            imaging_dataframe.write_csv(path + ".csv")
        elif output_format == "json":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"funky output arg: {output_format}")


def merge_population_tables(table_cfgs: list):
    population = pl.DataFrame()
    for table_cfg in table_cfgs:
        tab = load_table(table_cfg.table)
        tab = tab.select(list(table_cfg.columns.values()))
        tab = tab.rename({v: k for k, v in table_cfg.columns.items()})
        population = population.vstack(tab)
    return population
    
def dtype_from_cfg(dtype):
    if dtype == "string":
        return pl.String
    elif dtype == "integer":
        return pl.Int64
    elif dtype == "float":
        return pl.Float64
    elif dtype == "boolean":
        return pl.Boolean
    elif dtype == "date":
        return pl.Date
    elif dtype == "datetime":
        return pl.Datetime
    else:
        raise NotImplementedError(f"Unknown dtype: {dtype}")

def convert_to_date(
    name: str,
    date_format: str = "%Y-%m-%d",
    datetime_format: str | None = "%Y-%m-%d %H:%M:%S%.f",
) -> pl.Expr:
    """Force a column to `pl.Date` (optionally accepting datetimes and dropping time)."""
    s = pl.col(name)
    s_str = s.cast(pl.String)
    typed = s.cast(pl.Date, strict=False)
    parsed_date = s_str.str.strptime(pl.Date, date_format, strict=False)
    if datetime_format is None:
        return pl.coalesce([typed, parsed_date])
    parsed_dt_as_date = s_str.str.strptime(pl.Datetime, datetime_format, strict=False).dt.date()
    return pl.coalesce([typed, parsed_date, parsed_dt_as_date])

def convert_to_datetime(
    name: str,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
) -> pl.Expr:
    """Force a column to `pl.Datetime` using an explicit format."""
    s = pl.col(name)
    s_str = s.cast(pl.String)
    typed = s.cast(pl.Datetime, strict=False)
    parsed_dt = s_str.str.strptime(pl.Datetime, datetime_format, strict=False)
    return pl.coalesce([typed, parsed_dt])

def date_bound_expr(date_col=None, offset_days=0) -> pl.Expr | None:
    """Use as date_bound_expr(**cfg.time_conditionals.<window>.min_date) (YAML: column + offset_days)."""
    if date_col is None:
        return None
    off = int(offset_days) if offset_days is not None else 0
    base = convert_to_date(date_col, date_format="%Y-%m-%d")
    if off == 0:
        return base
    return base + pl.duration(days=off)


def warn_time_filter_dates(
    df: pl.DataFrame,
    *,
    label: str,
    event_col: str,
    event_d: pl.Expr,
    lo: pl.Expr | None,
    hi: pl.Expr | None,
    min_date_cfg: dict | None = None,
    max_date_cfg: dict | None = None,
) -> None:
    """Print warnings for date-parse and time-window issues before applying filters."""
    prefix = f"WARNING [{label}]"

    if event_col not in df.columns:
        print(f"{prefix}: event column '{event_col}' missing after join")
        return

    work = df.with_columns(event_d.alias("__event_d"))
    if lo is not None:
        work = work.with_columns(lo.alias("__lo"))
    if hi is not None:
        work = work.with_columns(hi.alias("__hi"))

    matched = work.filter(pl.col(event_col).is_not_null())
    n_matched = matched.height
    if n_matched == 0:
        print(f"{prefix}: no joined rows with non-null '{event_col}' (join may have failed)")
        return

    n_parsed = matched.filter(pl.col("__event_d").is_not_null()).height
    n_unparsed = n_matched - n_parsed
    if n_unparsed > 0:
        pct = 100 * n_unparsed / n_matched
        samples = (
            matched.filter(pl.col("__event_d").is_null())
            .select(event_col)
            .head(5)
            .to_series()
            .to_list()
        )
        print(
            f"{prefix}: {n_unparsed}/{n_matched} ({pct:.1f}%) non-null '{event_col}' "
            f"values failed date parse; sample raw values: {samples}"
        )

    for bound_name, bound_cfg in (("min", min_date_cfg), ("max", max_date_cfg)):
        if not bound_cfg:
            continue
        bound_col = bound_cfg.get("date_col")
        if not bound_col:
            continue
        if bound_col not in work.columns:
            print(f"{prefix}: {bound_name} bound column '{bound_col}' missing from joined table")
            continue
        n_null_bound = matched.filter(pl.col(bound_col).is_null()).height
        if n_null_bound > 0:
            print(
                f"{prefix}: {n_null_bound}/{n_matched} matched rows have null "
                f"{bound_name} bound column '{bound_col}'"
            )

    parsed = matched.filter(pl.col("__event_d").is_not_null())
    if parsed.height == 0:
        return

    if lo is not None:
        n_pass_lo = parsed.filter(pl.col("__event_d") >= pl.col("__lo")).height
        if n_pass_lo == 0:
            print(
                f"{prefix}: 0/{parsed.height} parsed events pass lower bound "
                "(check date formats on bound columns or offset_days)"
            )
    if hi is not None:
        n_pass_hi = parsed.filter(pl.col("__event_d") <= pl.col("__hi")).height
        if n_pass_hi == 0:
            print(
                f"{prefix}: 0/{parsed.height} parsed events pass upper bound "
                "(check date formats on bound columns or offset_days)"
            )
    if lo is not None and hi is not None:
        n_inverted = parsed.filter(pl.col("__lo") > pl.col("__hi")).height
        if n_inverted > 0:
            print(f"{prefix}: {n_inverted} rows have min bound > max bound")


def safe_save_df(df: pl.DataFrame) -> pl.DataFrame:
    """Polars CSV writer rejects Object columns; serialize them as JSON strings."""
    exprs = []
    for name in df.columns:
        if df.schema[name] == pl.Object:
            exprs.append(
                pl.col(name).map_elements(
                    lambda x: json.dumps(x, default=str, ensure_ascii=False),
                    return_dtype=pl.String,
                ).alias(name)
            )
    return df.with_columns(exprs) if exprs else df
