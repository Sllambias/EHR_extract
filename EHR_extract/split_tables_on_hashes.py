import hydra
import os
import polars as pl
import xlsxwriter
from dotenv import load_dotenv
from EHR_extract.utils.paths import get_config_path
from EHR_extract.utils.utils import (
    RecursiveSearchpathPlugin,
    convert_to_date,
    date_bound_expr,
    get_python_operator,
    load_table,
)
from hydra.core.plugins import Plugins
from omegaconf import DictConfig
from pathlib import Path

load_dotenv()
Plugins.instance().register(RecursiveSearchpathPlugin)


def sheet_name(table_name):
    return table_name.replace(" - CPMI", "").strip()[:31]


def filter_on_time_window(df, time_col, patient_rows, time_window):
    """Keep rows whose time_col falls inside the time window of any of the patient's population rows."""
    lo = date_bound_expr(**time_window.min_date)
    hi = date_bound_expr(**time_window.max_date)
    df = df.with_row_index("row_nr")
    matches = df.select("row_nr", convert_to_date(time_col).alias("event_date")).join(patient_rows, how="cross")
    if lo is not None:
        matches = matches.filter(pl.col("event_date") >= lo)
    if hi is not None:
        matches = matches.filter(pl.col("event_date") <= hi)
    return df.filter(pl.col("row_nr").is_in(matches["row_nr"].implode())).drop("row_nr")


def collect_hash_matches(cfg, population):
    """Collect configured columns for rows matching hashes from hashes.csv."""
    # maintain_order keeps the seeded sample below reproducible across runs.
    patients = population[cfg["population_id_column"]].str.strip_chars().drop_nulls().unique(maintain_order=True)

    max_ids = cfg.get("max_ids", None)
    if max_ids is not None:
        patients = patients.sample(n=min(max_ids, patients.len()), shuffle=True, seed=4215)

    patients = patients.to_list()

    output_dir = Path(cfg.paths.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    for patient in patients:
        print(f"\nprocessing id: {patient}")
        patient_rows = population.filter(pl.col(cfg["population_id_column"]).str.strip_chars() == patient)
        with xlsxwriter.Workbook(output_dir / f"{patient}.xlsx") as workbook:
            for table_cfg in cfg.tables:
                table_path = table_cfg.table
                table_name = Path(table_cfg.table).name
                print("processing table: ", table_name)
                df = pl.read_csv(table_path, infer_schema_length=100000000)

                df = df.with_columns(pl.col(table_cfg["id_col"]).str.strip_chars())
                df = df.filter(pl.col(table_cfg["id_col"]) == patient)

                if table_cfg.get("filters", None) is not None:
                    for row_filter in table_cfg["filters"]:
                        py_operator = get_python_operator(row_filter.operator)
                        df = df.filter(py_operator(pl.col(row_filter.column), row_filter.value))

                if table_cfg.get("time_window", None) is not None:
                    if table_cfg.get("time_col", None) is None:
                        raise ValueError(f"{table_name}: time_window needs a time_col to filter on")
                    time_window = cfg.time_conditionals[table_cfg["time_window"]]
                    df = filter_on_time_window(df, table_cfg["time_col"], patient_rows, time_window)

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
