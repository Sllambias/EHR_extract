import pandas as pd
import sqlite3
from datetime import datetime

db_path = "/projects/users/data/UCPH/DeepFetal/ultrasound/ultrasound_metadata_db.sqlite"

date = datetime.today().strftime("%Y-%m-%d")

conn = sqlite3.connect(db_path)

query = """
    SELECT t1.file_path, t1.study_date, t2.phair_hash
    FROM metadata_cache t1
    JOIN cpr_hashes t2 ON t1.file_hash = t2.xxhash
"""

df = pd.read_sql_query(query, conn)

df.to_csv(f"all_images_{date}.csv", index=False)

conn.close()
