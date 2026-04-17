import polars
import random
from datetime import datetime, timedelta


def generate_test_csv(num_rows, output_path):
    # Read CPR_BARN values from the source file
    source_df = polars.read_csv(
        "/Users/zcr545/Desktop/Projects/repos/ehr2meds/data/raw/fetal_data/SDS_AND_SP_from_population/population.csv"
    )
    cpr_mor_values = source_df["CPR_MOR"].to_list()
    cpr_child_values = source_df["CPR_BARN"].to_list()

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
        "cpr_mother": [random.choice(cpr_mor_values[:50]) for _ in range(num_rows)],
        "cpr_child": [random.choice(cpr_child_values[:50]) for _ in range(num_rows)],
        "file_path": [
            f"/images/study_{random.randint(1000, 9999)}/series_{random.randint(1, 10)}/image_{random.randint(1, 100)}.dcm"
            for _ in range(num_rows)
        ],
        "study_date": [random_study_date() for _ in range(num_rows)],
        "Birthdate": [random_birth_date() for _ in range(num_rows)],
        "GA_days": [random.randint(100, 300) for _ in range(num_rows)],
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
    print(file_paths)
    # Create img_type data with random class from 1-30
    data = {
        "file_path": [random.choice(file_paths) for _ in range(num_rows)],
        "class": [random.randint(1, 30) for _ in range(num_rows)],
    }

    # Create DataFrame and save to CSV
    df = polars.DataFrame(data)
    df.write_csv(output_path)

    return df


if __name__ == "__main__":
    output_csv_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/test_table.csv"
    generate_test_csv(1000, output_csv_path)

    img_type_output_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/img_type.csv"
    generate_img_type_csv(
        1000, img_type_output_path, sample_from="/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/test_table.csv"
    )
