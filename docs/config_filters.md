
# Previously implemented parameters

### GA Available at birth
```bash
- action: exclude
  conditions: 
    - standard: 
      table: ${paths.input_dir_SDS}/mfr.csv
      match_on: CPR_BARN
      column: GESTATIONSALDER_DAGE
      operator: "missing"
      value: 
    - standard: 
      table: ${paths.input_dir_SDS}/nyfoedte.csv
      match_on: CPRnummer_Barn
      column: Gestationsalder
      operator: "missing"
      value: 
    - standard: 
      table: ${paths.input_dir_SP}/Barn - Fødselsinfo.csv
      match_on: BABY_CPR
      column: Gestationsalder
      operator: "missing"
      value: 
```