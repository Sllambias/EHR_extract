import pandas as pd
from pathlib import Path

CSV_FILE = "/projects/users/data/UCPH/DeepFetal/projects/heart_seg/datanationalhjerte_phair.csv"
REGISTERS = ['/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/nyfoedte.csv', '/storage/archive/UCPH/DeepFetal/SDS/registers_pseudonymized/mfr.csv']


def _read_register(register):
    if register.lower().endswith("mfr.csv"):
        columns = {
            "CPR_MODER": "CPR mor",
            "GESTATIONSALDER_DAGE": "ga_fødsel",
            "CPR_BARN": "CPR barn",
            "FOEDSELSDATO": "birth_date",
        }
    elif register.lower().endswith("nyfoedte.csv"):
        columns = {
            "CPRnummer_Mor": "CPR mor",
            "Gestationsalder": "ga_fødsel",
            "CPRnummer_Barn": "CPR barn",
            "FoedselsDato_Barn": "birth_date",
        }
    else:
        raise ValueError(f"Unsupported register: {register}")

    raw_columns = list(columns)
    reg_df = pd.read_csv(
        register,
        dtype={column: "string" for column in raw_columns if column.startswith("CPR")},
        usecols=raw_columns,
    )
    reg_df = reg_df.rename(columns=columns)
    reg_df["ga_fødsel"] = pd.to_numeric(reg_df["ga_fødsel"], errors="coerce")
    reg_df["birth_date"] = pd.to_datetime(reg_df["birth_date"], errors="coerce", dayfirst=True).dt.normalize()
    return reg_df.dropna(subset=["CPR mor", "CPR barn"]).drop_duplicates()


def _closest_child_cpr(candidates, termin, context):
    if pd.isna(termin):
        raise ValueError(f"{context}: Termin is missing, so the birth date cannot be matched.")

    dated_candidates = candidates.dropna(subset=["birth_date"]).copy()
    if dated_candidates.empty:
        raise ValueError(f"{context}: matching register records have no valid birth dates.")

    dated_candidates["day_difference"] = (dated_candidates["birth_date"] - termin).abs().dt.days
    closest = dated_candidates[dated_candidates["day_difference"] == dated_candidates["day_difference"].min()]
    if closest["day_difference"].iloc[0] > 90:
        days = int(closest["day_difference"].iloc[0])
        raise ValueError(f"{context}: closest register birth is {days} days from Termin (maximum 90).")

    child_cprs = closest["CPR barn"].drop_duplicates()
    if len(child_cprs) != 1:
        raise ValueError(f"{context}: multiple children have equally close register birth dates.")
    return child_cprs.iloc[0]


def add_child_cpr_to_csv(csv_file, registers):
    df = pd.read_csv(csv_file, dtype={"CPR mor": "string", "CPR barn": "string"})
    df["ga_fødsel"] = pd.to_numeric(df["ga_fødsel"], errors="coerce")
    df["Termin"] = pd.to_datetime(df["Termin"], errors="coerce", dayfirst=True).dt.normalize()

    register_df = pd.concat([_read_register(register) for register in registers], ignore_index=True)
    errors = []

    for index, row in df.iterrows():
        context = f"CSV row {index + 2} (CPR mor={row['CPR mor']!r}, Termin={row['Termin']})"
        if pd.isna(row["CPR mor"]):
            errors.append(f"{context}: mother CPR is missing.")
            continue
        candidates = register_df[register_df["CPR mor"] == row["CPR mor"]]
        if candidates.empty:
            errors.append(f"{context}: no matching mother CPR in the registers.")
            continue

        ga_candidates = candidates[candidates["ga_fødsel"] == row["ga_fødsel"]]
        if pd.notna(row["ga_fødsel"]) and not ga_candidates.empty:
            child_cprs = ga_candidates["CPR barn"].drop_duplicates()
            if len(child_cprs) == 1:
                df.loc[index, "CPR barn"] = child_cprs.iloc[0]
                continue
            candidates = ga_candidates

        try:
            df.loc[index, "CPR barn"] = _closest_child_cpr(candidates, row["Termin"], context)
        except ValueError as error:
            errors.append(str(error))

    if errors:
        raise ValueError("Could not match CPR barn for all CSV rows:\n" + "\n".join(errors))

    output_path = Path(csv_file).with_name(f"{Path(csv_file).stem}_updated.csv")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    add_child_cpr_to_csv(CSV_FILE, REGISTERS)
