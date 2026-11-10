import pandas as pd
from pathlib import Path

CSV_FILE = "/projects/users/data/UCPH/DeepFetal/projects/heart_seg/datanationalhjerte_phair.csv"
REGISTERS = ['/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/nyfoedte.csv', '/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/MFR.csv']

def add_child_cpr_to_csv(csv_file, registers):
    df = pd.read_csv(csv_file)
    df_subset = df[['CPR mor', 'Termin']].drop_duplicates()
    
    for register in registers:
        reg_df = pd.read_csv(register)

        # differentiate between mfr and nyfoedte
        if register.endswith("MFR.csv"):
            matches = df_subset.merge(
                reg_df[['CPR_MODER', 'FOEDSELSDATO', 'CPR_BARN']],
                left_on=['CPR mor', 'Termin'],
                right_on=['CPR_MODER', 'FOEDSELSDATO'],
                how='left'
            )
            cpr_barn_col = 'CPR_BARN'
        elif register.endswith("nyfoedte.csv"):
            matches = df_subset.merge(
                reg_df[['CPRnummer_Mor', 'FoedselsDato_Barn', 'CPRnummer_Barn']],
                left_on=['CPR mor', 'Termin'],
                right_on=['CPRnummer_Mor', 'FoedselsDato_Barn'],
                how='left'
            )
            cpr_barn_col = 'CPRnummer_Barn'
        else:
            continue
            
        # Update the main dataframe with matched CPR barn values
        for _, row in matches.iterrows():
            mask = (df['CPR mor'] == row['CPR mor']) & (df['Termin'] == row['Termin'])
            if pd.notna(row[cpr_barn_col]):
                df.loc[mask, 'CPR barn'] = row[cpr_barn_col]
    
    df.to_csv(csv_file.replace(".csv", "_updated.csv"), index=False)

if __name__ == "__main__":
    add_child_cpr_to_csv(CSV_FILE, REGISTERS)
