import hydra
from dotenv import load_dotenv
from EHR_extract.paths import get_config_path
from EHR_extract.utils import load_table
from omegaconf import DictConfig

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(f"### Generating table summary for {cfg.table} with head: ###")
    print(load_table(cfg.table, n_rows=5).head())
    print("### Loading full table ###")
    table = load_table(cfg.table)


if __name__ == "__main__":
    main()


# %%


# %%
