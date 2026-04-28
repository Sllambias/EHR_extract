import polars as pl
import json

def load_table_path(path, strict=True, n_rows=None, has_header=True):
    if strict:
        ignore_errors = False
    else:
        ignore_errors = True

    if path.endswith(".csv"):
        try:
            return pl.read_csv(path, ignore_errors=ignore_errors, n_rows=n_rows, has_header=has_header)
        except pl.exceptions.ComputeError:
            return pl.read_csv(
                path, ignore_errors=ignore_errors, infer_schema_length=1000000, n_rows=n_rows, has_header=has_header
            )
    else:
        raise NotImplementedError(f"Unknown file type for path: {path}. Did you remember to add the file extension?")


def expr_startswith(col: pl.Expr, val) -> pl.Expr:
    s = col.cast(pl.String)
    return pl.any_horizontal(
        [s.str.starts_with(p) for p in val]
    )

def load_table(table_cfg, strict=True, n_rows=None, has_header=True):
    if isinstance(table_cfg, str):
        return load_table_path(table_cfg, strict=strict, n_rows=n_rows, has_header=has_header)
    else:
        table1 = load_table(table_cfg["table1"], strict=strict, n_rows=n_rows, has_header=has_header)
        table2 = load_table(table_cfg["table2"], strict=strict, n_rows=n_rows, has_header=has_header)
        left_on = table_cfg["left_on"]
        right_on = table_cfg["right_on"]
        if isinstance(left_on, list):
            left_on = [left_on]
        if isinstance(right_on, list):
            right_on = [right_on]
        return table1.join(table2, left_on=left_on, right_on=right_on, how="left")

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
    elif operator_str == "startswith":
        return expr_startswith
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
