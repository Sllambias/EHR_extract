import hydra
import json
import polars as pl
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import find_close_births, find_scantime_ga, find_images_and_timedeltas
from EHR_extract.paths import get_config_path
from EHR_extract.utils import (
    filter_numeric_rows,
    get_python_operator,
    load_table,
    update_population,
)
from omegaconf import DictConfig, OmegaConf

load_dotenv()

custom_functions = {
    "find_close_births": find_close_births,
    "find_images_and_timedeltas": find_images_and_timedeltas,
    "find_scantime_ga": find_scantime_ga,
}


def get_column_distribution(column):
    stats = {}
    if column.dtype.is_numeric():
        stats["mean"] = column.mean()
        stats["max"] = column.max()
        stats["min"] = column.min()
        stats["NA/NULL"] = column.null_count()
        stats["NOT NA/NULL"] = len(column) - stats["NA/NULL"]
    else:
        stats["unique"] = list(column.unique())
        stats["NA/NULL"] = column.null_count()
        stats["NOT NA/NULL"] = len(column) - stats["NA/NULL"]
    return stats


def summary_from_cfg(cfg):
    all_dists = {}
    population = set(load_table(cfg.base_population.table)[cfg.base_population.column])
    print("Population size:", len(population))

    for criterion in cfg.conditional_criteria:
        for condition in criterion.conditions:
            if all_dists.get(condition.table) is None:
                all_dists[condition.table] = {}
            table = load_table(condition.table, strict=cfg.strict)
            print(f"Table rows total: {len(table)} for table: {condition.table}")
            all_dists[condition.table][condition.column] = get_column_distribution(column=table.get_column(condition.column))

    print("\n ### Applying standard criteria ### \n")
    for table_cfg in cfg.get("standard_criteria", []):
        table = load_table(table_cfg.table, strict=cfg.strict)
        all_dists[table_cfg.table] = {}
        print(f"Table rows total: {len(table)} for table: {table_cfg.table}")
        table = table.filter(pl.col(table_cfg.match_on).is_in(population))
        print(f"Table rows matching population IDs: {len(table)} after filtering on {table_cfg.match_on}")
        for criteria in table_cfg.get("criteria", []):
            tmp_table = table.clone()
            all_dists[table_cfg.table][criteria.column] = get_column_distribution(column=tmp_table.get_column(criteria.column))

        for multicolumn_criteria in table_cfg.get("multicolumn_criteria", []):
            tmp_table = table.clone()
            for criterion in criteria.criteria:
                all_dists[table_cfg.table][criteria.column] = get_column_distribution(
                    column=tmp_table.get_column(criteria.column)
                )

    return all_dists

def get_summary(main_table: pl.DataFrame, ignore_columns, n_samples=10_000):
    sampled_table = main_table.drop(ignore_columns)
    n_draw = min(int(n_samples), sampled_table.height)
    sampled_table = sampled_table.sample(n=n_draw, shuffle=True)
    n_row = sampled_table.height
    rows = []
    for col in sampled_table.columns:
        series = sampled_table.get_column(col)
        dtype = series.dtype
        if getattr(dtype, "is_float", lambda: False)():
            missing = series.is_null() | series.is_nan()
            nan_count = int(missing.sum())
            nan_pct = float(missing.mean() * 100.0)
        else:
            nan_count = int(series.null_count())
            nan_pct = float(series.is_null().mean() * 100.0)

        if dtype == pl.Boolean:
            col_summary = float(series.sum() / n_row * 100.0)
        elif dtype.is_numeric():
            m = series.mean()
            col_summary = float(m) if m is not None else float("nan")
        else:
            vc = series.value_counts(sort=True)
            val_col, count_col = vc.columns[0], "count"
            col_summary = {
                str(row[val_col]): round(float(row[count_col]) / n_row * 100.0, 3)
                for row in vc.iter_rows(named=True)
            }
        rows.append(
            {
                "column": col,
                "nan_count": nan_count,
                "nan_pct": nan_pct,
                "summary": col_summary,
            }
        )
    summary = pl.DataFrame(
        {
            "column": [r["column"] for r in rows],
            "nan_count": [r["nan_count"] for r in rows],
            "nan_pct": [r["nan_pct"] for r in rows],
            "summary": pl.Series("summary", [r["summary"] for r in rows], dtype=pl.Object),
        }
    )
    return summary


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    dist = summary_from_cfg(cfg)
    with open(cfg.paths.distribution_save_path, "w") as fp:
        json.dump(dist, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
