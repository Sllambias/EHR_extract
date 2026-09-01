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


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population = load_table(cfg.paths.population_table)
    if cfg.max_ids is not None:
        population = population.sample(n=cfg.max_ids, shuffle=True, seed=4215)
    for table_cfg in cfg.tables:
        table = load_table(table_cfg.table)
        table = table.join(population, left_on=table_cfg.id_col, right_on=cfg.population_id_column)
        table_name = Path(table_cfg.table).name
        safe_save_df(table, fp=os.path.join(cfg.paths.output_dir, table_name))
        print(table_name, table)


if __name__ == "__main__":
    main()
