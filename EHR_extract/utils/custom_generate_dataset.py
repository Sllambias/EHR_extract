import polars as pl


def ultradino_train1():
    cervix_preds = "/projects/users/data/UCPH/DeepFetal/projects/common/cervix_classification/cervix_preds_v3.csv"
    cervix_preds = pl.read_csv(cervix_preds)

    img_db_snapshot = "/projects/users/data/UCPH/DeepFetal/ultrasound/tables/database_snapshots/all_images_2026-08-24.csv"
    img_db = pl.read_csv(img_db_snapshot)

    img_db_with_preds = cervix_preds.join(img_db, on="file_path")

    train_split = (
        "/projects/users/data/UCPH/DeepFetal/projects/common/splits/split_preterm_custom1/train_split_0_2026-08-24.csv"
    )
    train_mom_cpr = pl.read_csv(train_split)
    print(train_mom_cpr)
    train_imb_db_with_preds = img_db_with_preds.join(train_mom_cpr, left_on="phair_hash", right_on="CPR_MOR")
    train_imb_db_with_preds = train_imb_db_with_preds.select(["file_path", "is_cervix", "no_ocr_preprocessed_file_path"])
    print(train_imb_db_with_preds.null_count())
    train_imb_db_with_preds = train_imb_db_with_preds.drop_nulls()

    train_cervix_imgs = train_imb_db_with_preds.filter(pl.col("is_cervix") == 1)
    train_noncervix_imgs = train_imb_db_with_preds.filter(pl.col("is_cervix") == 0)
    print(len(train_cervix_imgs), len(train_noncervix_imgs))

    train_cervix_samples = train_cervix_imgs.sample(n=2500000, with_replacement=True, shuffle=True, seed=21498)
    train_noncervix_samples = train_noncervix_imgs.sample(n=2500000, with_replacement=False, shuffle=True, seed=42910)

    train_samples = (
        train_cervix_samples["no_ocr_preprocessed_file_path"].to_list()
        + train_noncervix_samples["no_ocr_preprocessed_file_path"].to_list()
    )

    df = pl.DataFrame(train_samples)
    df.write_csv("ultradino_train1.csv", include_header=False)
    print(len(train_samples))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fn")
    args = parser.parse_args()
    if args.fn == "ultradino_train1":
        ultradino_train1()
    else:
        print(args.fn)
