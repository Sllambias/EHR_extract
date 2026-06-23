import hydra
import logging
import numpy as np
import os
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
from EHR_extract.paths import get_config_path
from EHR_extract.utils import RecursiveSearchpathPlugin, merge_population_tables
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

Plugins.instance().register(RecursiveSearchpathPlugin)

load_dotenv()


def create_splits(
    population_cfg, update_train_split: bool = False, update_test_split: bool = False, holdout_frac=None, seed=None
):
    population = pl.DataFrame()
    rng = np.random.default_rng(seed=seed)

    full_population = merge_population_tables(population_cfg.tables, population=population, strict=False)
    full_population = set(full_population[population_cfg.population_key])

    if update_train_split or update_test_split:
        prev_train_population = pl.read_csv(update_train_split)
        prev_test_population = pl.read_csv(update_test_split)

        full_population.difference_update(set(prev_train_population[population_cfg.population_key]))
        full_population.difference_update(set(prev_test_population[population_cfg.population_key]))
        if len(full_population) == 0:
            logging.warning("NO NEW SAMPLES IN UPDATE")

    n_unique_ids = len(full_population)
    holdout_size = int(n_unique_ids * holdout_frac)

    test_population = set(rng.choice(list(full_population), size=holdout_size, replace=False))
    train_population = full_population
    train_population.difference_update(test_population)

    print(f"Made train split of len: {len(train_population)} and test split of len {len(test_population)}")
    assert len(test_population & train_population) == 0, "polution between train & test"

    train_df = pl.DataFrame({population_cfg.population_key: list(train_population)})
    test_df = pl.DataFrame({population_cfg.population_key: list(test_population)})

    if update_train_split or update_test_split:
        train_df = train_df.vstack(prev_train_population)
        test_df = test_df.vstack(prev_test_population)

    return train_df, test_df


@hydra.main(
    config_path=get_config_path(),
    config_name="split_V3",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = create_splits(
        population_cfg=cfg.population,
        update_train_split=cfg.get("update_train_split", False),
        update_test_split=cfg.get("update_test_split", False),
        holdout_frac=cfg.holdout_frac,
        seed=cfg.seed,
    )

    timestamp = datetime.today().strftime("%Y-%m-%d")

    train_output_path = os.path.join(cfg.paths.output_dir, f"train_split_{cfg.holdout_frac}_{timestamp}.csv")
    test_output_path = os.path.join(cfg.paths.output_dir, f"test_split_{cfg.holdout_frac}_{timestamp}.csv")

    if os.path.exists(train_output_path) or os.path.exists(test_output_path):
        logging.warning(
            "SPLITS WITH IDENTICAL NAMES ALREADY EXIST. WILL NOT SAVE THE CURRENTLY GENERATED SPLITS."
            "If this is intended manually delete the old splits or change the name/version of the current."
        )
    else:
        train_df.write_csv(train_output_path, separator=",")
        test_df.write_csv(test_output_path, separator=",")


if __name__ == "__main__":
    main()
