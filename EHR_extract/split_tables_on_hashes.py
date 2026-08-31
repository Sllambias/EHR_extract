import os
import polars as pl
import xlsxwriter
import yaml
from pathlib import Path
import os
import hydra
from pathlib import Path
from dotenv import load_dotenv
from EHR_extract.utils.paths import get_config_path
from EHR_extract.utils.utils import RecursiveSearchpathPlugin, load_table, safe_save_df
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

load_dotenv()
Plugins.instance().register(RecursiveSearchpathPlugin)


def sheet_name(table_name):
    return table_name.replace(" - CPMI", "").strip()[:31]


def collect_hash_matches(cfg, population):
    """Collect configured columns for rows matching hashes from hashes.csv."""
    patients = population[cfg["population_id_column"]].str.strip_chars().drop_nulls().to_list()

    if cfg.max_ids is not None:
        population = population.sample(n=cfg.max_ids, shuffle=True, seed=4215)

    output_dir = Path(cfg.paths.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    for patient in patients:
        print(f"\nprocessing id: {patient}")
        with xlsxwriter.Workbook(output_dir / f"{patient}.xlsx") as workbook:
            for table_cfg in cfg.tables:
                table_path = table_cfg.table
                table_name = Path(table_cfg.table).name
                print("processing table: ", table_name)
                df = pl.read_csv(table_path)

                df = df.with_columns(pl.col(table_cfg["id_col"]).str.strip_chars())
                df = df.filter(pl.col(table_cfg["id_col"]) == patient)
                if table_cfg.get("columns", None) is not None:
                    cols = [table_cfg["id_col"]]
                    if table_cfg.get("time_col", None) is not None:
                        cols.append(table_cfg["time_col"])
                    cols += table_cfg["columns"]
                    df = df.select(cols)

                if table_cfg.get("time_col", None) is not None:
                    df = df.sort(table_cfg["time_col"], descending=True, nulls_last=True)

                df.write_excel(workbook=workbook, worksheet=sheet_name(table_name))
                del df


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population = load_table(cfg.paths.population_table)

    collect_hash_matches(cfg, population)


if __name__ == "__main__":
    main()
