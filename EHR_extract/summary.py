import hydra
import json
import polars as pl
from dotenv import load_dotenv
from EHR_extract.custom_find_functions import find_close_siblings, find_scantime_ga, find_images_and_timedeltas
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
    "find_close_siblings": find_close_siblings,
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
