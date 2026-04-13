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
    return stats


def extract_from_cfg(cfg):
    all_discards = []
    all_dists = {}
    imaging_metadata = {}
    population = set(load_table(cfg.base_population.table)[cfg.base_population.column])
    print("Population size:", len(population))

    print("\n ### Applying standard criteria ### \n")
    for table_cfg in cfg.get("standard_criteria", []):
        table = load_table(table_cfg.table, strict=cfg.strict)
        all_dists[table_cfg.table] = {}
        print(f"Table rows total: {len(table)} for table: {table_cfg.table}")
        table = table.filter(pl.col(table_cfg.match_on).is_in(population))
        print(f"Table rows matching population IDs: {len(table)} after filtering on {table_cfg.match_on}")
        for criteria in table_cfg.get("criteria", []):
            tmp_table = table.clone()
            all_dists[table_cfg.table][criteria.column] = (
                get_column_distribution(column=tmp_table.get_column(criteria.column)) if cfg.distribution_save_path else {}
            )

            py_operator = get_python_operator(criteria.operator)
            if criteria.operator in [">", "<", ">=", "<="]:
                tmp_table = filter_numeric_rows(tmp_table, criteria.column)
            tmp_table = tmp_table.filter(py_operator(pl.col(criteria.column), criteria.value))

            population, discards = update_population(
                population=population,
                subset=set(tmp_table[table_cfg.match_on]),
                action=criteria.action,
            )
            all_discards.append([OmegaConf.to_container(criteria), list(discards)])
            print(f"Population size: {len(population)} after filtering on criteria {criteria}")

        for multicolumn_criteria in table_cfg.get("multicolumn_criteria", []):
            tmp_table = table.clone()
            for criterion in criteria.criteria:
                py_operator = get_python_operator(criterion.operator)
                dist = get_dist() if cfg.get_distribution(tmp_table.select(criterion.column)) else {}
                if criterion.operator in [">", "<", ">=", "<="]:
                    tmp_table = filter_numeric_rows(tmp_table, criterion.column)
                tmp_table = tmp_table.filter(py_operator(pl.col(criterion.column), criterion.value))
                population, discards = update_population(
                    population=population,
                    subset=set(tmp_table[table_cfg.match_on]),
                    action=criteria.action,
                )
                all_discards.append([OmegaConf.to_container(criteria), list(discards)])
                print(f"Population size: {len(population)} after filtering on multicolumn criteria {criterion}")
        print("---")

    print("\n ### Applying custom criteria ### \n")
    for custom_cfg in cfg.get("custom_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        set_of_matches = fn(**args, population=population)
        population, discards = update_population(
            population=population,
            subset=set_of_matches,
            action=custom_cfg.action,
        )
        all_discards.append([OmegaConf.to_container(custom_cfg), list(discards)])

        print(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function}")
        print("---")

    print("\n ### Applying imaging matching criteria ### \n")
    for custom_cfg in cfg.get("imaging_matching_criteria", {}):
        fn = custom_functions[custom_cfg.function]
        args = custom_cfg.args
        set_of_matches, imaging_metadata = fn(**args, population=population)
        population, discards = update_population(
            population=population,
            subset=set_of_matches,
            action=custom_cfg.action,
        )
        all_discards.append([OmegaConf.to_container(custom_cfg), list(discards)])

        print(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function}")
        print("---")
    return population, imaging_metadata, all_discards, all_dists


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population, metadata, discards, dist = extract_from_cfg(cfg)

    d = {}
    for i in range(len(discards)):
        d[i] = {
            "criteria": discards[i][0],
            "number_of_discards": len(discards[i][1]),
            "discards": discards[i][1],
        }

    with open(cfg.discards_save_path, "w") as fp:
        json.dump(d, fp, indent=4)
    with open(cfg.population_save_path, "w") as fp:
        json.dump(list(population), fp, indent=4)
    with open(cfg.distribution_save_path, "w") as fp:
        json.dump(dist, fp, indent=4)
    metadata.write_csv(cfg.imaging_metadata_save_path)


if __name__ == "__main__":
    main()
