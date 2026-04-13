import polars
import random
from datetime import datetime, timedelta


def generate_test_csv(num_rows, output_path):
    # Read CPR_BARN values from the source file
    source_df = polars.read_csv(
        "/Users/zcr545/Desktop/Projects/repos/ehr2meds/data/raw/fetal_data/SDS_and_SP_from_population/population.csv"
    )
    cpr_mor_values = source_df["CPR_MOR"].to_list()

    # Generate random date between 2020-01-01 and 2024-12-31
    start_date = datetime(1970, 1, 1)
    end_date = datetime(2024, 12, 31)

    def random_date():
        random_days = random.randint(0, (end_date - start_date).days)
        return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

    data = {
        "CPR_MOR": [random.choice(cpr_mor_values[:50]) for _ in range(num_rows)],
        "image_path": [
            f"/images/study_{random.randint(1000, 9999)}/series_{random.randint(1, 10)}/image_{random.randint(1, 100)}.dcm"
            for _ in range(num_rows)
        ],
        "studydate": [random_date() for _ in range(num_rows)],
        "GA_at_studydate_in_days": [random.randint(70, 280) for _ in range(num_rows)],
    }

    # Create DataFrame and save to CSV
    df = polars.DataFrame(data)
    df.write_csv(output_path)

    return df


if __name__ == "__main__":
    output_csv_path = "/Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/test_table.csv"
    generate_test_csv(1000, output_csv_path)
