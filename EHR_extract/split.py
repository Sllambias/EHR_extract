import hydra
import numpy as np
import os
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
from EHR_extract.paths import get_config_path
from EHR_extract.utils import merge_population_tables
from omegaconf import DictConfig

load_dotenv()


def create_splits(population_cfg, holdout_frac=None, seed=None):
    population = pl.DataFrame()
    rng = np.random.default_rng(seed=seed)

    full_population = merge_population_tables(population_cfg.tables, population=population, strict=False)
    full_population = set(full_population[population_cfg.population_key])

    n_unique_ids = len(full_population)
    holdout_size = int(n_unique_ids * holdout_frac)

    test_population = set(rng.choice(list(full_population), size=holdout_size, replace=False))
    train_population = full_population
    train_population.difference_update(test_population)

    print(f"Made train split of len: {len(train_population)} and test split of len {len(test_population)}")
    assert len(test_population & train_population) == 0, "polution between train & test"

    train_df = pl.DataFrame({population_cfg.population_key: list(train_population)})
    test_df = pl.DataFrame({population_cfg.population_key: list(test_population)})
    return train_df, test_df


@hydra.main(
    config_path=get_config_path(),
    config_name="splits/split_V3",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    train_df, test_df = create_splits(
        population_cfg=cfg.population,
        holdout_frac=cfg.holdout_frac,
        seed=cfg.seed,
    )

    timestamp = datetime.today().strftime("%Y-%m-%d")

    train_df.write_csv(os.path.join(cfg.paths.output_dir, f"train_split_{cfg.holdout_frac}_{timestamp}.csv"), separator=",")
    test_df.write_csv(os.path.join(cfg.paths.output_dir, f"test_split_{cfg.holdout_frac}_{timestamp}.csv"), separator=",")


if __name__ == "__main__":
    main()
