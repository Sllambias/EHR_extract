import pandas as pd
from pathlib import Path

CSV_FILE = "/projects/users/data/UCPH/DeepFetal/projects/heart_seg/datanationalhjerte_phair.csv"
REGISTERS = ['/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/nyfoedte.csv', '/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/mfr.csv']

def add_child_cpr_to_csv(csv_file, registers):
    df = pd.read_csv(csv_file)
    # fix ga_fødsel column to be consistent with gestational age in days
    df['ga_fødsel'] = df['ga_fødsel'].astype(int)
    
    df_subset = df[['CPR mor', 'ga_fødsel']].drop_duplicates()
    for register in registers:
        reg_df = pd.read_csv(register)

        # differentiate between mfr and nyfoedte
        if register.endswith("mfr.csv"):
            matches = df_subset.merge(
                reg_df[['CPR_MODER', 'GESTATIONSALDER_DAGE', 'CPR_BARN']],
                left_on=['CPR mor', 'ga_fødsel'],
                right_on=['CPR_MODER', 'GESTATIONSALDER_DAGE'],
                how='left'
            )
            cpr_barn_col = 'CPR_BARN'
        elif register.endswith("nyfoedte.csv"):
            matches = df_subset.merge(
                reg_df[['CPRnummer_Mor', 'Gestationsalder', 'CPRnummer_Barn']],
                left_on=['CPR mor', 'ga_fødsel'],
                right_on=['CPRnummer_Mor', 'Gestationsalder'],
                how='left'
            )
            cpr_barn_col = 'CPRnummer_Barn'
        else:
            continue
            
        # Update the main dataframe with matched CPR barn values
        for _, row in matches.iterrows():
            mask = (df['CPR mor'] == row['CPR mor']) & (df['ga_fødsel'] == row['ga_fødsel'])
            if pd.notna(row[cpr_barn_col]):
                df.loc[mask, 'CPR barn'] = row[cpr_barn_col]
    
    df.to_csv(csv_file.replace(".csv", "_updated.csv"), index=False)

if __name__ == "__main__":
    add_child_cpr_to_csv(CSV_FILE, REGISTERS)
