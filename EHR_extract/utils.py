import os
import polars as pl
import json
from EHR_extract.paths import get_config_path
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


def check_duplicates(table, population_column, allow_duplicates=False):
    duplicates = table[population_column].value_counts().filter(pl.col("count") > 1)
    if duplicates.height > 0:
        if not allow_duplicates:
            raise ValueError(f"Duplicate entries for key column {population_column}. Examples: {duplicates.head(5)}")
        else:
            table = table.group_by(population_column).agg(pl.col("*").first())
            assert len(table[population_column].unique()) == len(table[population_column])
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
            return pl.read_csv(
                path, ignore_errors=ignore_errors, n_rows=n_rows, has_header=has_header, null_values=null_values
            )
        except pl.exceptions.ComputeError:
            return pl.read_csv(
                path,
                ignore_errors=ignore_errors,
                infer_schema_length=10000000,
                n_rows=n_rows,
                has_header=has_header,
                null_values=null_values,
            )
    else:
        raise NotImplementedError(f"Unknown file type for path: {path}. Did you remember to add the file extension?")


def expr_startswith(col: pl.Expr, val) -> pl.Expr:
    s = col.cast(pl.String)
    return pl.any_horizontal([s.str.starts_with(p) for p in val])


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
        return lambda col, val: ~col.is_in(val)
    elif operator_str == "startswith":
        return lambda col, val: col.str.starts_with(val)
    elif operator_str == "missing":
        return lambda col, val: col.is_null()
    elif operator_str == "not missing":
        return lambda col, val: col.is_not_null()
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
        return lambda col, val: col.cast(pl.Float64, strict=False).is_between(val[0], val[1], closed="both")
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


def deduplicate_on_key(population, population_key):
    missing_count = pl.concat_list([pl.col(column).is_null().cast(pl.UInt32) for column in population.columns]).list.sum()
    population = population.with_columns(_missing_count=missing_count)
    population = population.sort([population_key, "_missing_count"], descending=[False, True])
    population = population.unique(subset=[population_key], keep="first").drop("_missing_count")
    return population


def merge_population_tables(table_cfgs: list, population, strict=True):
    for table_cfg in table_cfgs:
        tab = load_table(table_cfg.table, strict=strict)
        tab = tab.select(list(table_cfg.columns.values()))
        tab = tab.rename({v: k for k, v in table_cfg.columns.items()})
        tab = tab.select(sorted(tab.columns))
        if "GA" in tab.columns:
            tab = filter_numeric_rows(tab, "GA")
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
    parsed_iso_dt = s_str.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False).dt.date()
    parsed_iso_dt_frac = s_str.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False).dt.date()
    return pl.coalesce([typed, parsed_date, parsed_dt_as_date, parsed_iso_dt, parsed_iso_dt_frac])


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
                pl.col(name)
                .map_elements(
                    lambda x: json.dumps(x, default=str, ensure_ascii=False),
                    return_dtype=pl.String,
                )
                .alias(name)
            )
    return df.with_columns(exprs) if exprs else df


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
    def get_standard_format(condition):
        return f"{condition.get('standard', None)} {condition.get('column', None)} {condition.get('operator', None)} {condition.get('value', None)}"

    def get_custom_format(condition):
        return f"{condition.get('custom', None)} {condition.get('function', None)} {condition['args'].get('operator', None)} {condition['args'].get('value', None)}"

    crit_str = f"{criterion.action} IF: "
    s = [get_standard_format(cond) if "standard" in cond else get_custom_format(cond) for cond in criterion.conditions]
    output = crit_str + " ".join(s)
    return output


class RecursiveSearchpathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for path in os.listdir(get_config_path()):
            if os.path.isdir(os.path.join(get_config_path(), path)):
                search_path.append(
                    provider="recursive-searchpath-plugin", path="file://" + os.path.join(get_config_path(), path)
                )


def get_physical_deltas_post_PN_processing(
    image,
    physical_delta_x,
    physical_delta_y,
    region_location_min_x0,
    region_location_max_x1,
    region_location_min_y0,
    region_location_max_y1,
):
    x, y = image.size
    assert x == region_location_max_x1, f"{x} != {region_location_max_x1}"

    resampled_x = 224
    resampled_y = 224

    resampling_ratio_x = x / resampled_x
    resampled_physical_delta_x = physical_delta_x * resampling_ratio_x

    y_crop = abs(region_location_min_y0 - region_location_max_y1)
    y_cropped = y - y_crop

    resampling_ratio_y = y_cropped / resampled_y
    resampled_physical_delta_y = physical_delta_y * resampling_ratio_y

    return resampled_physical_delta_x, resampled_physical_delta_y


def calculate_region_y_ratio(image, region_location_max_y1):
    _, y = image.size
    return region_location_max_y1 / y


def downsample_segmentation_and_insert_black_bar(segmentation, y_downsample_ratio):
    x, y = segmentation.size
    canvas = torch.zeros((3, x, y))

    y_downsampled = int(y * (1 - y_downsample_ratio))

    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Resize(
                (x, y_downsampled), interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT
            ),
        ]
    )

    resampled_seg = tf(segmentation)
    canvas[:, :, y - y_downsampled :] = resampled_seg
    return canvas


if __name__ == "__main__":
    import argparse
    import os
    import torch
    import torchvision
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/all_images_X.csv")
    args = parser.parse_args()

    for idx, row in enumerate(pl.read_csv(args.path).iter_rows(named=True)):
        if idx > 10:
            break
        if not os.path.isfile(row["no_ocr_preprocessed_file_path"]):
            continue

        try:
            image = Image.open(row["no_ocr_preprocessed_file_path"])
            resampled_physical_delta_x, resampled_physical_delta_y = get_physical_deltas_post_PN_processing(
                image=image,
                physical_delta_x=float(row["physical_delta_y"]),
                physical_delta_y=float(row["physical_delta_y"]),
                region_location_min_x0=int(row["region_location_min_x0"]),
                region_location_max_x1=int(row["region_location_max_x1"]),
                region_location_min_y0=int(row["region_location_min_y0"]),
                region_location_max_y1=int(row["region_location_max_y1"]),
            )
            top_bar_ratio = calculate_region_y_ratio(image=image, region_location_max_y1=int(row["region_location_max_y1"]))
            x = downsample_segmentation_and_insert_black_bar(image, top_bar_ratio)
        except ValueError as e:  # Will skip when deltas or reg locs are lists
            print(f"skipping case. Due to error: {e}")

        print(resampled_physical_delta_x, resampled_physical_delta_y)
