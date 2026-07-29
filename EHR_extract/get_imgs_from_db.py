import pandas as pd
import sqlite3
from datetime import datetime

db_path = "/projects/users/data/UCPH/DeepFetal/ultrasound/tables/DeepFetal_image_database_250526.sqlite"

date = datetime.today().strftime("%Y-%m-%d")

conn = sqlite3.connect(db_path)

query = """
   SELECT t1.file_path, t1.no_ocr_preprocessed_file_path, t2.phair_hash, t3.study_date,
          t3.physical_delta_x, t3.physical_delta_y, t3.region_location_min_x0, t3. region_location_min_y0,
          t3.region_location_max_x1, region_location_max_y1
   FROM path_table t1
   JOIN cpr_hashes t2 ON t1.file_hash = t2.xxhash
   JOIN dicom_metadata_table t3 ON t1.sop_instance_uid = t3.sop_instance_uid
"""


df = pd.read_sql_query(query, conn)

df.to_csv(f"all_images_{date}.csv", index=False)

conn.close()
