import logging
import polars as pl
from EHR_extract.utils.utils import (
    check_duplicates,
    convert_to_date,
    date_bound_expr,
    dtype_from_cfg,
    filter_numeric_rows,
    get_python_operator,
    load_table,
    select_present,
    take_latest_row,
)


def match_value_on_birthdate(
    population, value_time_column, population_birthdate_column, population_gestational_age_column, include_days_after_birth=0
):
    population = filter_numeric_rows(population, population_gestational_age_column)
    population = population.with_columns(
        conception_date=pl.col(population_birthdate_column).str.to_datetime()
        - pl.duration(days=pl.col(population_gestational_age_column).cast(pl.Int64))
    )
    population = population.filter(
        (pl.col(value_time_column).str.to_datetime() >= pl.col("conception_date"))
        & (
            pl.col(value_time_column).str.to_datetime()
            <= pl.col(population_birthdate_column).str.to_datetime().dt.offset_by(f"{include_days_after_birth}d")
        )
    )
    return population


def match_value_with_child_cpr_on_lpr_id_to_mom_cpr_to_birthdate(
    operator,
    value,
    value_table_path,
    value_column,
    value_time_column,
    value_id_column,
    mapping_table_path,
    mapping_table_id_column,
    mapping_table_mom_cpr_column,
    population,
    population_mom_cpr_column,
    population_child_cpr_column,
    population_birth_column,
    population_gestational_age_column,
    population_key_column,
    include_days_after_birth=0,
):
    """
    This function takes tables A, B and C and matches a Value in Table A with a child CPR in Table C by:
    Finding the values, LPR_ID and value_timestamps in Table A
    Then matching the LPR_ID to the mom_CPR in Table B
    Then matching the mom_CPR to child_CPR in Table C
    and finally filtering the child_CPR if the value_timestamps fall within their pregnancy
    """
    # Includes population's columns: same-named columns in these tables take precedence in the joins.
    columns = [
        value_column,
        value_time_column,
        value_id_column,
        mapping_table_id_column,
        mapping_table_mom_cpr_column,
        population_mom_cpr_column,
        population_child_cpr_column,
        population_birth_column,
        population_gestational_age_column,
        population_key_column,
    ]
    value_table = load_table(value_table_path, columns=columns)
    py_operator = get_python_operator(operator[0])
    value_table = value_table.with_columns(positive=py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path, columns=columns)
    joined = value_table.join(
        mapping_table,
        left_on=value_id_column,
        right_on=mapping_table_id_column,
        how="inner",
    )
    joined = joined.join(
        population,
        left_on=mapping_table_mom_cpr_column,
        right_on=population_mom_cpr_column,
        how="inner",
    )

    joined = match_value_on_birthdate(
        population=joined,
        value_time_column=value_time_column,
        population_birthdate_column=population_birth_column,
        population_gestational_age_column=population_gestational_age_column,
        include_days_after_birth=include_days_after_birth,
    )

    # Get the unique child CPRs
    if operator[1] == "any":
        joined = joined.filter(pl.col("positive").any().over(population_child_cpr_column))
    elif operator[1] == "all":
        joined = joined.filter(pl.col("positive").all().over(population_child_cpr_column))
    population = population.filter(pl.col(population_key_column).rechunk().is_in(set(joined[population_child_cpr_column])))
    matches = set(population[population_child_cpr_column].unique())
    return matches


def match_value_with_child_cpr_on_birth_id(
    operator,
    value,
    value_table_path,
    value_column,
    value_table_birth_id_column,
    mapping_table_path,
    mapping_table_birth_id_column,
    mapping_table_child_cpr_column,
    population,
    population_key_column,
):
    # Includes population's columns: same-named columns in these tables take precedence in the joins.
    columns = [
        value_column,
        value_table_birth_id_column,
        mapping_table_birth_id_column,
        mapping_table_child_cpr_column,
        population_key_column,
    ]
    value_table = load_table(value_table_path, columns=columns)

    py_operator = get_python_operator(operator[0])
    value_table = value_table.with_columns(positive=py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path, columns=columns)
    mapping_table = mapping_table.filter(
        pl.col(mapping_table_child_cpr_column).rechunk().is_in(set(population[population_key_column]))
    )

    joined = value_table.join(
        mapping_table,
        left_on=value_table_birth_id_column,
        right_on=mapping_table_birth_id_column,
        how="inner",
    )
    joined = joined.join(population, left_on=mapping_table_child_cpr_column, right_on=population_key_column, how="inner")

    # Get the unique child CPRs
    if operator[1] == "any":
        joined = joined.filter(pl.col("positive").any().over(mapping_table_child_cpr_column))
    elif operator[1] == "all":
        joined = joined.filter(pl.col("positive").all().over(mapping_table_child_cpr_column))
    population = population.filter(
        pl.col(population_key_column).rechunk().is_in(set(joined[mapping_table_child_cpr_column]))
    )
    matches = set(population[population_key_column].unique())

    return matches


def match_value_with_child_cpr_on_birthdate(
    operator,
    value,
    value_table_path,
    value_column,
    value_time_column,
    value_mother_cpr_column,
    population,
    population_mother_cpr_column,
    population_child_cpr_column,
    population_birth_column,
    population_gestational_age_column,
    population_key_column,
    include_days_after_birth=0,
):
    # Includes population's columns: same-named columns in this table take precedence in the join.
    columns = [
        value_column,
        value_time_column,
        value_mother_cpr_column,
        population_mother_cpr_column,
        population_child_cpr_column,
        population_birth_column,
        population_gestational_age_column,
        population_key_column,
    ]
    value_table = load_table(value_table_path, columns=columns)

    # Filter based on operator and value
    py_operator = get_python_operator(operator[0])
    value_table = value_table.with_columns(positive=py_operator(pl.col(value_column), value))

    # Join on mother's CPR
    joined = value_table.join(population, left_on=value_mother_cpr_column, right_on=population_mother_cpr_column, how="inner")

    joined = match_value_on_birthdate(
        population=joined,
        value_time_column=value_time_column,
        population_birthdate_column=population_birth_column,
        population_gestational_age_column=population_gestational_age_column,
        include_days_after_birth=include_days_after_birth,
    )

    # Get the unique child CPRs
    if operator[1] == "any":
        joined = joined.filter(pl.col("positive").any().over(population_child_cpr_column))
    elif operator[1] == "all":
        joined = joined.filter(pl.col("positive").all().over(population_child_cpr_column))

    population = population.filter(
        pl.col(population_child_cpr_column).rechunk().is_in(set(joined[population_child_cpr_column]))
    )
    matches = set(population[population_child_cpr_column].unique())

    return matches


def match_years_with_child_cpr_on_birthdate(
    date_start,
    date_end,
    value_table_path,
    value_time_column,
    value_child_cpr_column,
    population,
    population_key_column,
):
    value_table = load_table(value_table_path, columns=[value_time_column, value_child_cpr_column, population_key_column])

    start = pl.lit(date_start).str.to_date("%d%m%Y")
    end = pl.lit(date_end).str.to_date("%d%m%Y")
    value_table = value_table.filter(pl.col(value_time_column).str.to_datetime(strict=False).dt.date().is_between(start, end))

    joined = population.join(value_table, left_on=population_key_column, right_on=value_child_cpr_column, how="inner")
    matches = set(joined[population_key_column].unique())
    return matches


def match_images_with_child(
    population, table_cfg, study_date_key="STUDY_DATE", mom_key="CPR_MOR", birthday_key="BIRTHDAY", ga_key="GA"
):
    """
    Barn CPR periode fra  fødselsdato - GA i dage til fødselsdato og så er alle billeder fra mor i den periode tilskrevet
    barns CPR. Så kan tvillinger også få tildelt samme billeder.
    """
    table = load_table(table_cfg.table, strict=False, columns=list(table_cfg.columns.values()))
    table = table.select(list(table_cfg.columns.values()))
    table = table.rename({v: k for k, v in table_cfg.columns.items()})
    table = table.join(population, left_on=mom_key, right_on=mom_key)
    # The birth date and GA are first turned into text (.cast(pl.String)): depending on how the population
    # was read they can arrive as numbers, and the .str functions below only work on text.
    table = table.with_columns(pl.col(birthday_key).cast(pl.String).str.to_datetime())
    table = table.with_columns(pl.col(study_date_key).cast(pl.String).str.to_date("%Y%m%d", strict=False))
    # GA as a whole number of days (a value that is not a number becomes empty)
    table = table.with_columns(pl.col(ga_key).cast(pl.String).str.to_integer(strict=False))
    table = table.unique()
    table = table.with_columns(
        image_during_pregnancy=pl.col(study_date_key).is_between(
            pl.col(birthday_key) - pl.duration(days=pl.col(ga_key)), pl.col(birthday_key)
        )
    )

    table = table.filter(pl.col("image_during_pregnancy"))
    logging.info(f"Valid images: {len(table)} after matching image + EHR matching.  \n")
    return table


def merge_population_on(population, table, merge_key, population_key_column):
    logging.info(f"Merging population of size {len(population)} with {table}")
    table = load_table(table)
    logging.info(logging.info(f"Merging population with table of size {len(table)}"))
    population = population.join(table, left_on=population_key_column, right_on=merge_key)
    logging.info(f"Population size after merge: {len(population)}")
    return population


def find_images_within_time_windows(
    scan_date_column,
    image_path_column,
    min_diff_days_scan_to_delivery,
    max_diff_days_scan_to_delivery,
    min_ga_in_days_at_scan,
    max_ga_in_days_at_scan,
    population,
    population_key_column,
    population_delivery_date_column="BIRTHDAY",
    population_ga_in_days_at_delivery_column="GA",
    add_columns=False,
):
    """Keep images inside both windows. add_columns keeps the columns scan_to_delivery and ga_at_scan (days)."""
    discard_stats = {"n_population_before_discard": len(population)}
    # Calculate absolute difference in days
    population = population.with_columns(
        scan_to_delivery=((pl.col(population_delivery_date_column) - pl.col(scan_date_column)).dt.total_days())
    )

    population = population.filter(
        (min_diff_days_scan_to_delivery < pl.col("scan_to_delivery"))
        & (pl.col("scan_to_delivery") < max_diff_days_scan_to_delivery)
    )

    population = filter_numeric_rows(population, population_ga_in_days_at_delivery_column)
    population = population.with_columns(
        ga_at_scan=(pl.col(population_ga_in_days_at_delivery_column)).cast(pl.Float64) - pl.col("scan_to_delivery")
    )
    population = population.filter(
        (min_ga_in_days_at_scan < pl.col("ga_at_scan")) & (pl.col("ga_at_scan") < max_ga_in_days_at_scan)
    )
    if add_columns:
        population = population.with_columns(pl.col("ga_at_scan").cast(pl.Int64))
    else:
        population = population.drop(["ga_at_scan", "scan_to_delivery"])
    discard_stats.update(
        {
            "criteria": "find_images_and_timedeltas",
            "discards": "N/A",
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def find_images_with_one_region(columns, population, population_key_column):
    """Drop images where any of the columns holds a list of values, such as "[0.01, 0.02]" for flow images."""
    discard_stats = {"n_population_before_discard": len(population)}
    for column in columns:
        n_values = pl.col(column).cast(pl.String).str.strip_chars("[] ").str.split(",").list.len()
        population = population.filter(n_values.fill_null(1) == 1)

    discard_stats.update(
        {
            "criteria": "find_images_with_one_region",
            "discards": "N/A",
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def find_images_with_predicted_classes(
    table,
    classes,
    class_column,
    image_path_column,
    population,
    population_image_path_column,
    population_key_column,
):
    discard_stats = {"n_population_before_discard": len(population)}

    table_path = table
    table = load_table(table, columns=[class_column, image_path_column])
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    matched_paths = table.filter(pl.col(class_column).is_in(classes))[image_path_column]
    logging.debug(f"Table rows matching predicted classes: {len(matched_paths)}")

    population = population.filter(pl.col(population_image_path_column).rechunk().is_in(matched_paths.implode()))
    logging.debug(f"Table rows matching population: {len(population)}")

    discard_stats.update(
        {
            "criteria": "find_images_with_predicted_classes",
            "discards": "N/A",
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def _parse_date(column):
    """Text or date column -> Date. Accepts 2019-03-01, 2019-03-01 10:00:00, 20190301 and 01MAR2019."""
    # Read the value as text and remove spaces around it
    c = pl.col(column).cast(pl.String).str.strip_chars()
    # Try the formats one after the other; the first one that gives a date is used:
    #   2019-03-01 (also with a time after it) | 20190301 | 01MAR2019
    return pl.coalesce(
        c.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        c.str.to_date("%Y%m%d", strict=False),
        c.str.to_date("%d%b%Y", strict=False),
    )


def _load_event_table(src, columns):
    """The table with the event codes. The date column may be missing here (then it is taken from the
    mapping table, e.g. LPR2: D_INDDTO is in t_adm, not in t_diag)."""
    # Read only the needed columns: the codes, the date (if any) and the linking ID
    table = load_table(src["table"], columns=columns)
    # Stop with a clear message if the code column or the linking ID column is missing
    for c in (src["code_column"], columns[-1]):
        if c not in table.columns:
            raise ValueError(f"find_images_before_event: column '{c}' not found in {src['table']}. "
                             f"Columns: {pl.read_csv(src['table'], n_rows=0).columns}")
    return table


def _load_mapping_table(src, columns, date_col, event_table):
    """The linking table (t_adm / nyfoedte); also reads the date column when the event table does not have it."""
    # If the date is not in the event table, read it from the linking table instead
    # (LPR2: the admission date D_INDDTO is in t_adm, not in t_diag)
    if date_col and date_col not in event_table.columns:
        columns = columns + [date_col]
    return load_table(src["mapping_table"], columns=columns)


def _check_date_column(src, date_col, table):
    # After joining the two tables, the date column must exist in one of them
    if date_col and date_col not in table.columns:
        raise ValueError(f"find_images_before_event: date column '{date_col}' found neither in {src['table']} "
                         f"nor in {src['mapping_table']}. Check date_column in the YAML.")


def find_images_before_event(
    sources,
    population,
    population_key_column,
    keep_same_day=False,
    study_date_column="STUDY_DATE",
    birthday_column="BIRTHDAY",
    ga_column="GA",
    mother_column="CPR_MOR",
    event_name="event",
):
    """Keep only the images taken BEFORE an event (e.g. a cervical cerclage) in the same pregnancy.

    For every child (pregnancy) the events are collected from `sources` (a list of dicts):
      table, code_column, codes, operator (in | == | startswith | startswith_any, default in)
      link: "lpr"      -> id_column (V_RECNUM), mapping_table (t_adm), mapping_id_column (K_RECNUM),
                          mapping_mother_column (V_CPR); events are matched to the mother's pregnancy
                          by date: pregnancy start - lookback_days <= event date <= birth date
            "birth_id" -> birth_id_column (DW_EK_Foedsel), mapping_table (nyfoedte), mapping_birth_id_column,
                          mapping_child_column (CPRnummer_Barn)
      date_column:       column with the event date (None/omitted = the source has no date)
      date_is_placement: true  -> the date is the moment the event started (e.g. cerclage placement)
                         false -> the date is not the start (e.g. removal of the stitch, an admission date)
      lookback_days:     (lpr only) also count events up to this many days BEFORE the pregnancy started
                         (e.g. an abdominal cerclage placed before the pregnancy), default 0
    Per pregnancy:
      - with >= 1 dated placement event: keep only images taken before the EARLIEST placement date
        (images on the same day are removed unless keep_same_day is true)
      - with events but no dated placement event: ALL images of that pregnancy are removed
        (there was an event, but it is unknown when it started)
      - without events: all images are kept
    """
    discard_stats = {"n_population_before_discard": len(population)}
    # --- Step 1: one row per child (= per pregnancy) with its time window:
    #     _birth = birth date, _start = first day of the pregnancy (birth date - GA)
    children = (population.select([population_key_column, mother_column, birthday_column, ga_column])
                          .unique(subset=population_key_column)
                          .with_columns(_parse_date(birthday_column).alias("_birth"),
                                        pl.col(ga_column).cast(pl.String).cast(pl.Float64, strict=False).alias("_ga"))
                          .with_columns((pl.col("_birth") - pl.duration(days=pl.col("_ga"))).alias("_start")))
    # --- Step 2: collect the events of every source in `sources` (from the YAML).
    #     Each source gives rows: child, event date, and whether that date is a placement date.
    events = []
    for src in sources:
        src = dict(src)
        # how the codes are compared: in (default), ==, startswith or startswith_any
        op = get_python_operator(src.get("operator", "in"))
        date_col = src.get("date_column")
        # placement = the date of this source is the START of the event (e.g. the cerclage was put in).
        # A source without a date can never give a placement date.
        placement = bool(src.get("date_is_placement", True)) and date_col is not None
        cols = [src["code_column"]] + ([date_col] if date_col else [])
        # a list of codes from the YAML becomes a Python list; a single code stays text
        codes = list(src["codes"]) if not isinstance(src["codes"], str) else src["codes"]
        # LPR sources (t_sksopr, t_diag): record number -> mother (via t_adm) -> her pregnancy
        # that contains the event date
        if src["link"] == "lpr":
            table = _load_event_table(src, cols + [src["id_column"]])
            # keep only the rows with one of the event codes
            table = table.filter(op(pl.col(src["code_column"]), codes))
            mapping = _load_mapping_table(src, [src["mapping_id_column"], src["mapping_mother_column"]], date_col, table)
            # record number -> mother's ID (and the date, if it is in t_adm)
            table = table.join(mapping, left_on=src["id_column"], right_on=src["mapping_id_column"])
            _check_date_column(src, date_col, table)
            table = table.with_columns((_parse_date(date_col) if date_col else pl.lit(None, pl.Date)).alias("_date"))
            # mother -> all her children (pregnancies) in the population
            table = table.join(children, left_on=src["mapping_mother_column"], right_on=mother_column)
            # keep the event only if its date falls inside THIS pregnancy:
            #   pregnancy start - lookback_days <= event date <= birth date
            # (lookback_days > 0 also catches e.g. an abdominal cerclage placed before the pregnancy)
            lookback = pl.duration(days=int(src.get("lookback_days", 0)))
            table = table.filter((pl.col("_date") >= pl.col("_start") - lookback) & (pl.col("_date") <= pl.col("_birth")))
            child_col = population_key_column
        # Birth-linked sources (mor_procedurer, mor_diagnoser): birth ID -> the child directly,
        # so no date window is needed
        elif src["link"] == "birth_id":
            table = _load_event_table(src, cols + [src["birth_id_column"]])
            table = table.filter(op(pl.col(src["code_column"]), codes))
            mapping = _load_mapping_table(src, [src["mapping_birth_id_column"], src["mapping_child_column"]], date_col, table)
            table = table.join(mapping, left_on=src["birth_id_column"], right_on=src["mapping_birth_id_column"])
            _check_date_column(src, date_col, table)
            table = table.with_columns((_parse_date(date_col) if date_col else pl.lit(None, pl.Date)).alias("_date"))
            child_col = src["mapping_child_column"]
        else:
            # only 'lpr' and 'birth_id' are known
            raise ValueError(f"find_images_before_event: unknown link '{src['link']}' (use 'lpr' or 'birth_id')")
        # log how many children this source found
        logging.info(f"{event_name}: {table.select(child_col).n_unique()} children with an event in "
                     f"{src['table']} ({src['code_column']})")
        # keep: child, event date (empty if unknown) and whether that date is a placement date
        events.append(table.select(pl.col(child_col).cast(pl.String).alias("_child"), pl.col("_date"),
                                   pl.lit(placement).alias("_placement")))

    # --- Step 3: one row per child that had an event, with its EARLIEST placement date
    #     (empty when only events without a placement date were found)
    schema = {"_child": pl.String, "_date": pl.Date, "_placement": pl.Boolean}
    per_child = pl.concat(events) if events else pl.DataFrame(schema=schema)
    # only children that are in the population
    per_child = per_child.filter(pl.col("_child").is_in(children[population_key_column].cast(pl.String).implode()))
    per_child = per_child.group_by("_child").agg(
        pl.col("_date").filter(pl.col("_placement") & pl.col("_date").is_not_null()).min().alias("_first_placement"))

    # --- Step 4: decide for every image row
    # before = the scan was taken before the placement date (on the same day only if keep_same_day)
    scan = _parse_date(study_date_column)
    before = (scan <= pl.col("_first_placement")) if keep_same_day else (scan < pl.col("_first_placement"))
    # attach each child's first placement date to its image rows
    population = population.with_columns(pl.col(population_key_column).cast(pl.String).alias("_child"))
    population = population.join(per_child, on="_child", how="left")
    # keep an image if its pregnancy had no event at all, OR if a placement date is known and the scan was
    # before it ( | means OR, & means AND ). A pregnancy with an event but no placement date loses all images.
    no_event = ~pl.col("_child").is_in(per_child["_child"].implode())
    population = population.filter(no_event | (pl.col("_first_placement").is_not_null() & before))
    # remove the helper columns
    population = population.drop(["_child", "_first_placement"])

    # --- Step 5: report in the log and in the _discards file
    n_children = per_child.height
    n_undated = per_child.filter(pl.col("_first_placement").is_null()).height
    logging.info(f"{event_name}: {n_children} children in the population had an event; {n_undated} of them without a "
                 f"dated placement (all their images removed); the others keep only images before the event")
    discard_stats.update(
        {
            "criteria": f"find_images_before_event ({event_name})",
            "discards": {"children_with_event": n_children, "children_without_dated_placement": n_undated},
            "n_discards": discard_stats["n_population_before_discard"] - len(population),
        }
    )
    return population, discard_stats


def find_close_births(
    value,
    operator,
    table,
    match_on,
    mom_column,
    birth_id_column,
    delivery_date_column,
    population,
    population_key_column,
):
    # Sort by mother and birth date
    population = set(population.get_column(population_key_column))
    py_operator = get_python_operator(operator)
    table_path = table
    table = load_table(table, columns=[match_on, mom_column, birth_id_column, delivery_date_column])
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    table = table.with_columns(pl.col(delivery_date_column).str.to_date())
    table = table.sort([mom_column, delivery_date_column])

    # Calculate difference between consecutive births for each mother
    table = table.with_columns(
        diff=pl.col(delivery_date_column).diff().over(mom_column),
        prev_child_ID=pl.col(match_on).shift(1).over(mom_column),
        prev_child_birth_ID=pl.col(birth_id_column).shift(1).over(mom_column),
    )
    # Filter to find children with siblings born less than 40 weeks apart
    # 40 weeks = 280 days
    close_siblings = table.filter(
        (py_operator(pl.col("diff").dt.total_days(), value)) & (pl.col(birth_id_column) != pl.col("prev_child_birth_ID"))
    )
    close_siblings = close_siblings.filter(pl.col(match_on).rechunk().is_in(population))

    # Get the CPR_BARN values to exclude
    siblings_to_exclude = set(close_siblings[match_on]) | set(close_siblings["prev_child_ID"])
    return siblings_to_exclude


def find_duplicated_ids(table, match_on, id_columns, population, population_key_column, min_count=2):
    """IDs (match_on, e.g. the child CPR) whose group in id_columns (e.g. birth ID + mother) contains at least
    min_count different IDs. min_count=2 (default): every multiple birth (twins, triplets, ...), same as before.
    min_count=3: only triplets or more (twins are NOT returned)."""
    population = set(population.get_column(population_key_column))
    table_path = table
    table = load_table(table, columns=[match_on, id_columns])
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")
    # id_columns may be one column name or a list of names: make it a list
    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    # ignore rows with an empty birth ID or mother: they cannot be grouped into a birth
    table = table.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in id_columns]))
    # count the DIFFERENT children per birth (id_columns) and keep the births with at least min_count of them:
    #   min_count=2 -> twins, triplets, ... (all multiple births)   min_count=3 -> triplets or more only
    duplicated_ids = table.filter(pl.col(match_on).n_unique().over(id_columns) >= min_count)
    # only children that are in the population
    duplicated_ids = duplicated_ids.filter(pl.col(match_on).rechunk().is_in(population))
    logging.debug(
        f"Table rows / unique IDs matching population IDs: {len(table)} / {table[match_on].n_unique()} \
            after filtering on {match_on}"
    )
    duplicated_ids = set(duplicated_ids[match_on])
    return duplicated_ids


def find_pregnancy_start(table, birth_date_col, GA_days_col, pregnancy_start_col):
    table = table.with_columns(
        (
            pl.col(birth_date_col).cast(pl.Date, strict=False)
            - pl.duration(days=pl.col(GA_days_col).cast(pl.Int64, strict=False))
        ).alias(pregnancy_start_col)
    )
    return table


def find_GA_days(table, GA_weeks_col, GA_days_col):
    weeks = pl.col(GA_weeks_col).cast(pl.String).str.extract(r"(?i)(\d+)\s*w", group_index=1).cast(pl.Int64, strict=False)
    days = (
        pl.col(GA_weeks_col)
        .cast(pl.String)
        .str.extract(r"(?i)w\s*(\d+)\s*d", group_index=1)
        .cast(pl.Int64, strict=False)
        .fill_null(0)
    )
    table = table.with_columns((weeks * 7 + days).alias(GA_days_col))
    return table


def find_GA_weeks(table, GA_days_col, GA_weeks_col):
    table = table.with_columns((pl.col(GA_days_col).cast(pl.Int64, strict=False) / 7).alias(GA_weeks_col))
    return table


def find_date_at_GA(table, birth_date_col, GA_days_col, GA_number, date_col):
    GA_difference = pl.col(GA_days_col).cast(pl.Int64, strict=False) - int(GA_number)
    table = table.with_columns(
        (pl.col(birth_date_col).cast(pl.Date, strict=False) - pl.duration(days=GA_difference)).alias(date_col)
    )
    return table


def find_GA_at_date(table, birth_date_col, GA_days_col, study_date_col, GA_at_date_col):
    """GA in days at `study_date_col`, from GA at birth and birth date."""
    birth_d = convert_to_date(birth_date_col)
    study_d = convert_to_date(study_date_col)
    days_to_birth = (birth_d - study_d).dt.total_days()
    table = table.with_columns((pl.col(GA_days_col).cast(pl.Int64, strict=False) - days_to_birth).alias(GA_at_date_col))
    return table


def find_maternal_age(
    table,
    m_table_path,
    maternal_birth_date_col: str,
    maternal_id_col: str,
    baby_birth_date_col: str,
    key_column: str,
    maternal_age_col: str,
    population_maternal_id_col: str = "m_cpr",
):
    base_cols = table.columns
    m_table = load_table(m_table_path, columns=[maternal_id_col, maternal_birth_date_col]).select(
        [maternal_id_col, maternal_birth_date_col]
    )
    m_table = m_table.unique(subset=[maternal_id_col], keep="first")

    merged = table.join(m_table, left_on=population_maternal_id_col, right_on=maternal_id_col, how="left")

    # Normalize both to `Date` (accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"; drop time).
    baby_d = convert_to_date(baby_birth_date_col)
    mom_d = convert_to_date(maternal_birth_date_col)

    years = baby_d.dt.year() - mom_d.dt.year()
    had_birthday = (baby_d.dt.month() > mom_d.dt.month()) | (
        (baby_d.dt.month() == mom_d.dt.month()) & (baby_d.dt.day() >= mom_d.dt.day())
    )
    merged = merged.with_columns((years - (~had_birthday).cast(pl.Int64)).cast(pl.Int64, strict=False).alias(maternal_age_col))
    # Keep only original columns + the newly created age column.
    return merged.select(base_cols + [maternal_age_col])


def extract_filtered_values_from_source(
    main_table,
    *,
    table,
    left_on,
    right_on,
    target_col,
    date_col,
    min_date,
    max_date,
    filters,
    new_col_name,
    dtype,
    allow_duplicates=False,
):
    filter_cols = [f.column for f in filters or []]
    used = [left_on, right_on, target_col, date_col, min_date.get("date_col"), max_date.get("date_col"), *filter_cols]
    table = load_table(table, strict=False, columns=used)

    for filter in filters or []:
        py_operator = get_python_operator(filter.operator)
        table = table.filter(py_operator(pl.col(filter.column), filter.value))

    tmp_table = select_present(main_table, used).join(
        table,
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    event_d = convert_to_date(date_col)
    lo = date_bound_expr(**min_date)
    if lo is not None:
        tmp_table = tmp_table.filter(event_d >= lo)
    hi = date_bound_expr(**max_date)
    if hi is not None:
        tmp_table = tmp_table.filter(event_d <= hi)

    pl_dtype = dtype_from_cfg(dtype)
    tmp_table = tmp_table.filter(pl.col(target_col).cast(pl_dtype, strict=False).is_not_null())

    tmp_table = take_latest_row(tmp_table, left_on, date_col)
    tmp_table = tmp_table.select([left_on, target_col]).rename({target_col: new_col_name})
    if not allow_duplicates:
        return check_duplicates(tmp_table, left_on)
    return tmp_table


def merge_source_specs(table=None, sources=None, **shared):
    """Build per-source arg dicts; `sources` overrides shared fields per entry."""
    keys = ("table", "left_on", "right_on", "target_col", "date_col", "filters")
    if sources is not None:
        specs = []
        for src in sources:
            spec = {k: shared.get(k) for k in keys}
            for k in keys:
                if k in src and src[k] is not None:
                    spec[k] = src[k]
            specs.append(spec)
        return specs
    if table is None:
        raise ValueError("extract_filtered_values requires `table` or `sources`")
    return [{k: shared.get(k) for k in keys} | {"table": table}]


def extract_filtered_values(
    main_table,
    left_on,
    right_on=None,
    target_col=None,
    date_col=None,
    min_date=None,
    max_date=None,
    filters=None,
    new_col_name=None,
    dtype=None,
    allow_duplicates=False,
    table=None,
    sources=None,
):

    specs = merge_source_specs(
        table=table,
        sources=sources,
        left_on=left_on,
        right_on=right_on,
        target_col=target_col,
        date_col=date_col,
        filters=filters,
    )
    for i, spec in enumerate(specs):
        chunk = extract_filtered_values_from_source(
            main_table,
            left_on=left_on,
            right_on=spec["right_on"],
            target_col=spec["target_col"],
            date_col=spec["date_col"],
            min_date=min_date,
            max_date=max_date,
            filters=spec["filters"],
            new_col_name=new_col_name,
            dtype=dtype,
            allow_duplicates=allow_duplicates,
            table=spec["table"],
        )
        fb_col = f"__{new_col_name}_fb"
        if i == 0:
            if new_col_name in main_table.columns:
                main_table = main_table.drop(new_col_name)
            main_table = main_table.join(chunk, on=left_on, how="left")
        else:
            main_table = main_table.join(
                chunk.rename({new_col_name: fb_col}),
                on=left_on,
                how="left",
            )
            main_table = main_table.with_columns(pl.coalesce([pl.col(new_col_name), pl.col(fb_col)]).alias(new_col_name)).drop(
                fb_col
            )
        if not allow_duplicates:
            main_table = check_duplicates(main_table, left_on)
    return main_table


def extract_filtered_conditional_values(
    main_table,
    left_on,
    right_on,
    key_column,
    table,
    target_col,
    date_col,
    min_date,
    max_date,
    filters,
    new_col_name,
    dtype,
    conditions,
    allow_duplicates=False,
):
    filtered_table = extract_filtered_values(
        main_table=main_table,
        table=table,
        left_on=left_on,
        right_on=right_on,
        target_col=target_col,
        date_col=date_col,
        min_date=min_date,
        max_date=max_date,
        filters=filters,
        new_col_name="target_col",
        dtype=dtype,
        allow_duplicates=True,
    )

    # Convert that extracted value into a boolean using `condition`.
    condition_matches = set()
    for condition in conditions:
        py_operator = get_python_operator(condition.operator)
        tmp_table = filtered_table.filter(py_operator(pl.col(condition.column), condition.value))
        print("After filter", len(tmp_table))
        if condition.condition is None:
            last_condition = set(tmp_table[key_column])
        elif condition.condition == "and":
            last_condition = last_condition.intersection(set(tmp_table[key_column]))
        elif condition.condition == "or":
            condition_matches = condition_matches.union(last_condition)
            last_condition = set(tmp_table[key_column])
        else:
            print("wow, weird condition")
    condition_matches = condition_matches.union(last_condition)
    tmp_table = tmp_table.with_columns(pl.col(key_column).rechunk().is_in(list(condition_matches)).alias(new_col_name))
    main_table = main_table.join(
        tmp_table.select([key_column, new_col_name]),
        on=key_column,
        how="left",
    )  # .with_columns(
    #     pl.coalesce([pl.col(f"{new_col_name}_new"), pl.col(new_col_name)])
    #     .alias(new_col_name)
    # ) #.drop(f"{new_col_name}_new")

    # Check for duplicates
    duplicates = main_table[key_column].value_counts().filter(pl.col("count") > 1)
    if duplicates.height > 0 and not allow_duplicates:
        raise ValueError(f"Duplicate entries for key column {key_column}. Examples: {duplicates.head(5)}")
    else:
        main_table = main_table.group_by(key_column).agg(pl.col("*").first())
        assert len(main_table[key_column].unique()) == len(main_table[key_column])
    return main_table


def extract_latest_value_from_source(
    main_table,
    *,
    table,
    left_on,
    right_on,
    target_col,
    new_col_name,
    date_col,
    min_date,
    max_date,
    dtype,
):
    used = [left_on, right_on, target_col, date_col, min_date.get("date_col"), max_date.get("date_col")]
    table = load_table(table, strict=False, columns=used)

    tmp_table = select_present(main_table, used).join(
        table,
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    event_d = convert_to_date(date_col)
    lo = date_bound_expr(**min_date)
    if lo is not None:
        tmp_table = tmp_table.filter(event_d >= lo)
    hi = date_bound_expr(**max_date)
    if hi is not None:
        tmp_table = tmp_table.filter(event_d <= hi)

    pl_dtype = dtype_from_cfg(dtype)
    tmp_table = tmp_table.filter(pl.col(target_col).cast(pl_dtype, strict=False).is_not_null())

    tmp_table = take_latest_row(tmp_table, left_on, date_col)
    return tmp_table.select([left_on, target_col]).rename({target_col: new_col_name})


def extract_latest_value(
    main_table,
    left_on,
    right_on=None,
    target_col=None,
    new_col_name=None,
    date_col=None,
    min_date=None,
    max_date=None,
    dtype=None,
    allow_duplicates=False,
    table=None,
    sources=None,
):
    specs = merge_source_specs(
        table=table,
        sources=sources,
        left_on=left_on,
        right_on=right_on,
        target_col=target_col,
        date_col=date_col,
        filters=[],
    )
    for i, spec in enumerate(specs):
        chunk = extract_latest_value_from_source(
            main_table,
            table=spec["table"],
            left_on=left_on,
            right_on=spec["right_on"],
            target_col=spec["target_col"],
            new_col_name=new_col_name,
            date_col=spec["date_col"],
            min_date=min_date,
            max_date=max_date,
            dtype=dtype,
        )
        fb_col = f"__{new_col_name}_fb"
        if i == 0:
            if new_col_name in main_table.columns:
                main_table = main_table.drop(new_col_name)
            main_table = main_table.join(chunk, on=left_on, how="left")
        else:
            main_table = main_table.join(
                chunk.rename({new_col_name: fb_col}),
                on=left_on,
                how="left",
            )
            main_table = main_table.with_columns(pl.coalesce([pl.col(new_col_name), pl.col(fb_col)]).alias(new_col_name)).drop(
                fb_col
            )
    return main_table


def find_images_with_min_mother_age_at_scan(
    population,
    population_key_column,
    min_age=18,
    mother_birthday_column="MOTHER_BIRTHDAY",
    scan_date_column="STUDY_DATE",
):
    """Remove images taken while the mother was younger than min_age (age ON THE DAY OF THE SCAN).
    Only possible where the mother's birth date is known (nyfoedte). Images of mothers without a known
    birth date are kept here: their age AT THE BIRTH is checked by a criterion on MOTHER_AGE."""
    discard_stats = {"n_population_before_discard": len(population)}
    # the mother's birth date and the scan date, as dates
    birthday = convert_to_date(mother_birthday_column)
    scan = convert_to_date(scan_date_column)
    # had_birthday = on the scan day, had her birthday of that year already come?
    #   yes if the scan is in a later month than her birthday, or in the same month on or after the day
    had_birthday = (scan.dt.month() > birthday.dt.month()) | (
        (scan.dt.month() == birthday.dt.month()) & (scan.dt.day() >= birthday.dt.day())
    )
    age_at_scan = scan.dt.year() - birthday.dt.year() - pl.when(had_birthday).then(0).otherwise(1)  # scan year - birth year - 1 (if mother's birthday has not happened yet this year)
    # keep the image if her birth date is unknown (her age AT THE BIRTH was checked by the MOTHER_AGE criterion)
    # OR if she was at least min_age on the day of the scan   ( | means OR )
    population = population.filter(birthday.is_null() | (age_at_scan >= min_age))
    # report in the _discards file how many image rows were removed
    discard_stats.update({"criteria": f"find_images_with_min_mother_age_at_scan (>= {min_age})", "discards": "N/A",
                          "n_discards": discard_stats["n_population_before_discard"] - len(population)})
    return population, discard_stats


def find_images_in_scan_years(
    population,
    population_key_column,
    year_from=None,
    year_to=None,
    scan_date_column="STUDY_DATE"
):
    """Keep only images with a scan date from year_from to year_to (both included; None = no limit)."""
    discard_stats = {"n_population_before_discard": len(population)}
    # the year of each scan
    year = convert_to_date(scan_date_column).dt.year()
    # keep scans from year_from onwards (None = no lower limit)
    if year_from is not None:
        population = population.filter(year >= int(year_from))
    # keep scans up to and including year_to (None = no upper limit)
    if year_to is not None:
        population = population.filter(year <= int(year_to))
    # report in the _discards file how many image rows were removed
    discard_stats.update({"criteria": f"find_images_in_scan_years ({year_from}-{year_to})", "discards": "N/A",
                          "n_discards": discard_stats["n_population_before_discard"] - len(population)})
    return population, discard_stats


def keep_one_row_per_image(population, population_key_column, image_path_column="FILE_PATH"):
    """Twins share their mother's images, so a shared image appears once per twin. Keep it once: the row
    of the child whose ID sorts first (twins have the same GA at birth, so the label is the same)."""
    discard_stats = {"n_population_before_discard": len(population)}
    # sort so that, for every image, the twin whose ID sorts first comes first,
    # then keep only that first row of each image
    population = (population.sort([image_path_column, population_key_column])
                            .unique(subset=image_path_column, keep="first", maintain_order=True))
    # report in the _discards file how many (duplicate) rows were removed
    discard_stats.update({"criteria": "keep_one_row_per_image", "discards": "N/A",
                          "n_discards": discard_stats["n_population_before_discard"] - len(population)})
    return population, discard_stats


def add_label_by_ga_at_birth(
    population,
    population_key_column,
    positive_max_ga_at_birth=224,
    ga_column="GA",
    label_column="label",
    positive="positive",
    negative="negative"
):
    """Add a label column: positive if GA at birth <= positive_max_ga_at_birth (days), else negative.
    Removes nothing."""
    # GA at birth as a number: <= positive_max_ga_at_birth (224 days = 32+0) -> positive, otherwise negative
    # (criterion 1 already removed children without a GA)
    population = population.with_columns(
        pl.when(pl.col(ga_column).cast(pl.Float64, strict=False) <= positive_max_ga_at_birth)
          .then(pl.lit(positive)).otherwise(pl.lit(negative)).alias(label_column))
    # nothing is removed, so n_discards = 0
    return population, {"n_population_before_discard": len(population), "criteria": "add_label_by_ga_at_birth",
                        "discards": "N/A", "n_discards": 0}
