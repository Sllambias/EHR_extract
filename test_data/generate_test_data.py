import polars
import random
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import os


def generate_test_csv(num_rows, output_path):
    # Read CPR_BARN values from the source file
    source_df = polars.read_csv(
        "/Users/zcr545/Desktop/Projects/repos/ehr2meds/data/raw/fetal_data/SDS_AND_SP_from_population/population.csv"
    )
    cpr_mor_values = source_df["MOR_CPR"].to_list()

    # Generate random date between 2020-01-01 and 2024-12-31
    start_date = datetime(1970, 1, 1)
    end_date = datetime(2024, 12, 31)

    def random_study_date():
        random_days = random.randint(0, (end_date - start_date).days)
        return (start_date + timedelta(days=random_days)).strftime("%Y%m%d")

    def random_birth_date():
        random_days = random.randint(0, (end_date - start_date).days)
        return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

    data = {
        "phair_hash": [random.choice(cpr_mor_values) for _ in range(num_rows)],
        "file_path": [
            f"images/study_{random.randint(1000, 9999)}/series_{random.randint(1, 10)}/image_{random.randint(1, 100)}.dcm"
            for _ in range(num_rows)
        ],
        "study_date": [random_study_date() for _ in range(num_rows)],
        "physical_delta_x": [random.uniform(0.1, 1.0) for _ in range(num_rows)],
        "physical_delta_y": [random.uniform(0.1, 1.0) for _ in range(num_rows)],
        "region_location_min_x0": 0,
        "region_location_min_y0": 0,
        "region_location_max_x1": [random.randint(50, 800) for _ in range(num_rows)],
        "region_location_max_y1": 30,
        "no_ocr_preprocessed_file_path": [
            f"/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/preprocessed_images/study_{random.randint(1000, 9999)}_series_{random.randint(1, 10)}_image_{random.randint(1, 100)}.png"
            for _ in range(num_rows)
        ],
    }

    # Create DataFrame and save to CSV
    df = polars.DataFrame(data)
    df.write_csv(output_path)

    return df


def generate_img_type_csv(num_rows, output_path, sample_from):
    # Generate test data to get file paths
    test_df = polars.read_csv(sample_from)

    # Sample file paths from the generated data
    file_paths = test_df["file_path"].to_list()
    # Create img_type data with random class from 1-30
    data = {
        "file_path": [random.choice(file_paths) for _ in range(num_rows)],
        "pred": [random.randint(1, 30) for _ in range(num_rows)],
        "is_cervix": [random.randint(0, 2) for _ in range(num_rows)],
    }

    # Create DataFrame and save to CSV
    df = polars.DataFrame(data)
    df.write_csv(output_path)

    return df


def generate_holdout_csv(num_rows, output_path, sample_from):
    # Generate test data to get file paths
    test_df = polars.read_csv(sample_from)

    # Sample file paths from the generated data
    ids = test_df["phair_hash"].to_list()
    # Create img_type data with random class from 1-30
    data = [random.choice(ids) for _ in range(num_rows)]

    # Create DataFrame and save to CSV
    df = polars.DataFrame({"CPR_MOR": data})
    df.write_csv(output_path)


def generate_test_images(n, path_table):
    path_table = polars.read_csv(path_table)
    dir = path_table["no_ocr_preprocessed_file_path"][0].rsplit("/", 1)[0]
    os.makedirs(dir, exist_ok=True)
    for i in range(n):
        x = path_table["region_location_max_x1"][i]
        y = np.random.randint(50, 1200)
        data = np.zeros((y, x))
        data = Image.fromarray(data)
        data = data.convert("RGB")
        data.save(path_table["no_ocr_preprocessed_file_path"][i])


if __name__ == "__main__":
    output_csv_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/all_images_X.csv"
    generate_test_csv(5000, output_csv_path)

    img_type_output_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/img_type.csv"
    generate_img_type_csv(
        5000, img_type_output_path, sample_from="/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/all_images_X.csv"
    )

    holdout_output_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/holdout.csv"
    generate_holdout_csv(
        50, holdout_output_path, sample_from="/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/all_images_X.csv"
    )

    generate_test_images(5000, output_csv_path)
