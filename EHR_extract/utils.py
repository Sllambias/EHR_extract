import polars as pl
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from EHR_extract.paths import get_config_path
import os


def load_table(path, strict=True, n_rows=None, has_header=True):
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


def get_python_operator(operator_str):
    if operator_str == "in":
        return lambda col, val: col.is_in(val)
    elif operator_str == "not_in":
        raise NotImplementedError("NOT IN is not implemented as it should not be used. Be precise and use the IN operator.")
    elif operator_str == "startswith":
        return lambda col, val: col.str.starts_with(val)
    elif operator_str == "missing":
        return lambda col, val: col.is_null()
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
    table = table.drop("parsed")
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


def merge_population_tables(table_cfgs: list, population, strict=True):
    for table_cfg in table_cfgs:
        tab = load_table(table_cfg.table, strict=strict)
        tab = tab.select(list(table_cfg.columns.values()))
        tab = tab.rename({v: k for k, v in table_cfg.columns.items()})
        tab = tab.select(sorted(tab.columns))
        population = population.vstack(tab)
    return population


def merge_composed_population_tables(population, population_merge_on, composed_table_cfgs: list, format_SP_GA=False):
    for composition_cfg in composed_table_cfgs:
        tables = [load_table(table_cfg.table) for table_cfg in composition_cfg.tables]
        tables = [
            tab.select(list(table_cfg.columns.values())).rename({v: k for k, v in table_cfg.columns.items()})
            for tab, table_cfg in zip(tables, composition_cfg.tables)
        ]
        merged_table = tables[0]
        for tab in tables[1:]:
            merged_table = merged_table.join(tab, on=composition_cfg.merge_on)
        merged_table = merged_table.select(sorted(merged_table.columns))
        if composition_cfg.format_SP_GA:
            merged_table = merged_table.with_columns(pl.col("GA").map_elements(format_sp_ga))
        population = population.vstack(merged_table)

    return population


def format_sp_ga(ga_str):
    # Assuming GA is in the format "XX weeks YY days"
    if isinstance(ga_str, str):
        parts = ga_str.split()
        if len(parts) == 0 or not parts[0].isnumeric():
            return ""
        if len(parts) == 1:
            return str(int(parts[0]) * 7)
        weeks = int(parts[0])
        days = int(parts[1][0])
        return str(weeks * 7 + days)
    else:
        return ""


def format_criterion(criterion):
    crit_str = f"{criterion.action} IF: "
    s = [
        f"{cond.get('condition', None)} {cond.get('column', None)} {cond.get('operator', None)} {cond.get('value', None)}"
        for cond in criterion.conditions
    ]
    output = crit_str + " ".join(s)
    return output


class RecursiveSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in os.listdir(get_config_path()):
            if os.path.isdir(os.path.join(get_config_path(), path)):
                search_path.append(
                    provider="recursive-searchpath-plugin", path="file://" + os.path.join(get_config_path(), path)
                )
