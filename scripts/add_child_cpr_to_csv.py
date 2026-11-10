import pandas as pd
from pathlib import Path

CSV_FILE = "/projects/users/data/UCPH/DeepFetal/projects/heart_seg/datanationalhjerte_phair.csv"
REGISTERS = ['/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/nyfoedte.csv', '/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/mfr.csv']
MAX_DATE_DIFFERENCE_DAYS = 150

def add_child_cpr_to_csv(csv_file, registers):
    df = pd.read_csv(csv_file)
    df_subset = df[['CPR mor', 'Termin']].drop_duplicates().copy()
    df_subset['_termin_date'] = pd.to_datetime(df_subset['Termin'], errors='coerce', dayfirst=True)

    register_dfs = []
    for register in registers:
        reg_df = pd.read_csv(register)

        # differentiate between mfr and nyfoedte
        if register.endswith("mfr.csv"):
            reg_df = reg_df[['CPR_MODER', 'FOEDSELSDATO', 'CPR_BARN']].rename(
                columns={
                    'CPR_MODER': 'CPR mor',
                    'FOEDSELSDATO': 'birth_date',
                    'CPR_BARN': 'CPR barn'
                }
            )
        elif register.endswith("nyfoedte.csv"):
            reg_df = reg_df[['CPRnummer_Mor', 'FoedselsDato_Barn', 'CPRnummer_Barn']].rename(
                columns={
                    'CPRnummer_Mor': 'CPR mor',
                    'FoedselsDato_Barn': 'birth_date',
                    'CPRnummer_Barn': 'CPR barn'
                }
            )
        else:
            continue

        register_dfs.append(reg_df)

    if register_dfs:
        reg_df = pd.concat(register_dfs, ignore_index=True)
        reg_df['birth_date'] = pd.to_datetime(reg_df['birth_date'], errors='coerce', dayfirst=True)
        matches = df_subset.merge(reg_df, on='CPR mor', how='left')
        matches['day_difference'] = (matches['birth_date'] - matches['_termin_date']).abs().dt.days
        matches = matches[matches['day_difference'] <= MAX_DATE_DIFFERENCE_DAYS]
        matches = matches.sort_values('day_difference').drop_duplicates(['CPR mor', 'Termin'])

        # Update the main dataframe with matched CPR barn values
        for _, row in matches.iterrows():
            mask = (df['CPR mor'] == row['CPR mor']) & (df['Termin'] == row['Termin'])
            if pd.notna(row['CPR barn']):
                df.loc[mask, 'CPR barn'] = row['CPR barn']

    df.to_csv(csv_file.replace(".csv", "_updated.csv"), index=False)

if __name__ == "__main__":
    add_child_cpr_to_csv(CSV_FILE, REGISTERS)
