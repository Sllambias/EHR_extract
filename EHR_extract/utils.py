import polars as pl


def load_table(path, strict=True, n_rows=None):
    if strict:
        ignore_errors = False
    else:
        ignore_errors = True

    if path.endswith(".csv"):
        return pl.read_csv(path, ignore_errors=ignore_errors, n_rows=n_rows)
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


def update_population(population, subset, action):
    pre_discard_population = len(population)
    if action == "exclude":
        discards = subset
        population.difference_update(subset)
    elif action == "include":
        discards = population.difference(subset)
        population = population.intersection(subset)
    else:
        raise NotImplementedError(f"unexpected action: {action}")
    return population, discards, len(discards), pre_discard_population
