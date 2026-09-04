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
    take_latest_row,
)


def match_value_on_birthdate(population, value_time_column, population_birthdate_column, population_gestational_age_column):
    population = filter_numeric_rows(population, population_gestational_age_column)
    population = population.with_columns(
        conception_date=pl.col(population_birthdate_column).str.to_datetime()
        - pl.duration(days=pl.col(population_gestational_age_column).cast(pl.Int64))
    )
    population = population.filter(
        (pl.col(value_time_column).str.to_datetime() >= pl.col("conception_date"))
        & (pl.col(value_time_column).str.to_datetime() <= pl.col(population_birthdate_column).str.to_datetime())
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
):
    """
    This function takes tables A, B and C and matches a Value in Table A with a child CPR in Table C by:
    Finding the values, LPR_ID and value_timestamps in Table A
    Then matching the LPR_ID to the mom_CPR in Table B
    Then matching the mom_CPR to child_CPR in Table C
    and finally filtering the child_CPR if the value_timestamps fall within their pregnancy
    """
    value_table = load_table(value_table_path)
    py_operator = get_python_operator(operator[0])
    value_table = value_table.with_columns(positive=py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path)
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
    )

    # Get the unique child CPRs
    if operator[1] == "any":
        joined = joined.filter(pl.col("positive").any().over(population_child_cpr_column))
    elif operator[1] == "all":
        joined = joined.filter(pl.col("positive").all().over(population_child_cpr_column))
    population = population.filter(pl.col(population_key_column).is_in(set(joined[population_child_cpr_column])))
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
    value_table = load_table(value_table_path)

    py_operator = get_python_operator(operator[0])
    value_table = value_table.with_columns(positive=py_operator(pl.col(value_column), value))

    mapping_table = load_table(mapping_table_path)
    mapping_table = mapping_table.filter(pl.col(mapping_table_child_cpr_column).is_in(set(population[population_key_column])))

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
    population = population.filter(pl.col(population_key_column).is_in(set(joined[mapping_table_child_cpr_column])))
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
):
    value_table = load_table(value_table_path)

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
    )

    # Get the unique child CPRs
    if operator[1] == "any":
        joined = joined.filter(pl.col("positive").any().over(population_child_cpr_column))
    elif operator[1] == "all":
        joined = joined.filter(pl.col("positive").all().over(population_child_cpr_column))

    population = population.filter(pl.col(population_child_cpr_column).is_in(set(joined[population_child_cpr_column])))
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
    value_table = load_table(value_table_path)

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
    table = load_table(table_cfg.table, strict=False)
    table = table.select(list(table_cfg.columns.values()))
    table = table.rename({v: k for k, v in table_cfg.columns.items()})
    table = table.join(population, left_on=mom_key, right_on=mom_key)
    table = table.with_columns(pl.col(birthday_key).str.to_datetime())
    table = table.with_columns(pl.col(study_date_key).cast(pl.String).str.to_date("%Y%m%d"))
    table = table.with_columns(pl.col(ga_key).str.to_integer(strict=False))
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
):
    discard_stats = {"n_population_before_discard": len(population)}
    # Calculate absolute difference in days
    population = population.with_columns(
        diff_in_days_scan_to_delivery=((pl.col(population_delivery_date_column) - pl.col(scan_date_column)).dt.total_days())
    )

    population = population.filter(
        (min_diff_days_scan_to_delivery < pl.col("diff_in_days_scan_to_delivery"))
        & (pl.col("diff_in_days_scan_to_delivery") < max_diff_days_scan_to_delivery)
    )

    population = filter_numeric_rows(population, population_ga_in_days_at_delivery_column)
    population = population.with_columns(
        GA_in_days_at_scantime=(pl.col(population_ga_in_days_at_delivery_column)).cast(pl.Float64)
        - pl.col("diff_in_days_scan_to_delivery")
    )
    population = population.filter(
        (min_ga_in_days_at_scan < pl.col("GA_in_days_at_scantime"))
        & (pl.col("GA_in_days_at_scantime") < max_ga_in_days_at_scan)
    )
    population = population.drop(["GA_in_days_at_scantime", "diff_in_days_scan_to_delivery"])
    discard_stats.update(
        {
            "criteria": "find_images_and_timedeltas",
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
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")

    matched_paths = table.filter(pl.col(class_column).is_in(classes))[image_path_column]
    logging.debug(f"Table rows matching predicted classes: {len(matched_paths)}")

    population = population.filter(pl.col(population_image_path_column).is_in(matched_paths))
    logging.debug(f"Table rows matching population: {len(population)}")

    discard_stats.update(
        {
            "criteria": "find_images_with_predicted_classes",
            "discards": "N/A",
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
    table = load_table(table)
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
    close_siblings = close_siblings.filter(pl.col(match_on).is_in(population))

    # Get the CPR_BARN values to exclude
    siblings_to_exclude = set(close_siblings[match_on]) | set(close_siblings["prev_child_ID"])
    return siblings_to_exclude


def find_duplicated_ids(table, match_on, id_columns, population, population_key_column):
    population = set(population.get_column(population_key_column))
    table_path = table
    table = load_table(table)
    logging.debug(f"Table rows total: {len(table)} for table: {table_path}")
    duplicated_ids = table.filter(table[id_columns].is_duplicated())
    duplicated_ids = duplicated_ids.filter(pl.col(match_on).is_in(population))
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
    m_table = load_table(m_table_path).select([maternal_id_col, maternal_birth_date_col])
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
    table = load_table(table, strict=False)

    for filter in filters or []:
        py_operator = get_python_operator(filter.operator)
        table = table.filter(py_operator(pl.col(filter.column), filter.value))

    tmp_table = main_table.join(
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
            main_table = check_duplicates(main_table, left_on, allow_duplicates=allow_duplicates)
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
    tmp_table = tmp_table.with_columns(pl.col(key_column).is_in(list(condition_matches)).alias(new_col_name))
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
    table = load_table(table, strict=False)

    tmp_table = main_table.join(
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
