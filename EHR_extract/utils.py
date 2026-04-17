import polars as pl


def load_table(path, strict=True, n_rows=None):
    if strict:
        ignore_errors = False
    else:
        ignore_errors = True

    if path.endswith(".csv"):
        try:
            return pl.read_csv(path, ignore_errors=ignore_errors, n_rows=n_rows)
        except pl.exceptions.ComputeError:
            return pl.read_csv(path, ignore_errors=ignore_errors, infer_schema_length=1000000, n_rows=n_rows)
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
