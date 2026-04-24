### The population
The ```population:``` defines the base population and the dataframe that will be used to filter the population. The required keys are 
- `population.population_key` which defines the key/column which defines the population
- `population.file_path_key` which defines the column name we will use to store the image paths (Currently required because splits are defined on path level. Will be removed in the future when splits are on IDs)
- `population.tables` which contains a list of dicts with the keys `table` and `columns`. The value of the table key should be the path to the table and the value of the columns key should be a dict with a k,v mapping specifying column renaming patterns with keys being the new name and values being the old name. This is required to ensure that columns with different names but similar contents can be joined across tables.

In the example below the primary population is each child (i.e. each (hashed) CPR_CHILD). The population is comprised of all children in either `mfr` or `nyfoedte` and the resulting table contains the columns `CPR_BARN | CPR_MOR | GA | BIRTHDAY` with rows equalling the sum of all rows in the tables in the list.

```
population: 
  population_key: CPR_BARN
  file_path_key: FILE_PATH
  tables: 
    - table: ${paths.input_dir_SDS}/mfr.csv
      columns:
        CPR_BARN: CPR_BARN
        CPR_MOR: CPR_MODER
        GA: GESTATIONSALDER_DAGE
        BIRTHDAY: FOEDSELSDATO
    - table: ${paths.input_dir_SDS}/nyfoedte.csv
      columns:
        CPR_BARN: CPRnummer_Barn
        CPR_MOR: CPRnummer_Mor
        GA: Gestationsalder
        BIRTHDAY: FoedselsDato_Barn
```

### Matching images with children
By default images are matched with the mother but not the child so, to match images with children, we require an extracted imaging csv containing at least the three columns (1) ID of the mother, (2) the image path and (3) the date of the examination. Like in the population definition the columns are a k,v mapping of `NEW_COLUMN_NAME`:`OLD_COLUMN_NAME` and the column of the Mother's ID must match the columns of the mother's ID specified in the population, as this will be the key that the tables will be joined on.

Internally, this means that each child will be matched with all the mother's images at first. Afterwards, for each child, we will remove all images with study dates outside the range of the [birthday - GA, birthday]. In the end the table will have a row for each valid image for each child. This means the same image may appear in rows for multiple children, as will be the case for twin pregnancies, and that each child ID will appear in as many rows as the mother had images in the duration of that pregnancy.

```
imaging_table:
  table: /Users/zcr545/Desktop/Projects/repos/EHR_extract/test_data/test_table.csv
  columns:
    CPR_MOR: cpr_mother
    FILE_PATH: file_path
    STUDY_DATE: study_date
```